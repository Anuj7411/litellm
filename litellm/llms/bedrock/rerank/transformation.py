"""
Translates from Cohere's `/v1/rerank` input format to Bedrock's `/rerank` input format.

Why separate file? Make it easy to see how transformation works
"""

import math
from typing import Final

from litellm._uuid import uuid
from litellm.types.llms.bedrock import (
    BedrockRerankBedrockRerankingConfiguration,
    BedrockRerankConfiguration,
    BedrockRerankInlineDocumentSource,
    BedrockRerankModelConfiguration,
    BedrockRerankQuery,
    BedrockRerankRequest,
    BedrockRerankSource,
    BedrockRerankTextDocument,
    BedrockRerankTextQuery,
)
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankRequest,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class BedrockRerankConfig:
    # Bedrock bills one search unit per up to 100 document chunks in a query,
    # but the Rerank response never reports how many units were actually billed.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-pricing.html
    MAX_DOCUMENT_CHUNKS_PER_SEARCH_UNIT: Final[int] = 100

    def _transform_sources(self, documents: list[str | dict]) -> list[BedrockRerankSource]:
        """
        Transform the sources from RerankRequest format to Bedrock format.
        """
        _sources: Final = []
        for document in documents:
            if isinstance(document, str):
                _sources.append(
                    BedrockRerankSource(
                        inlineDocumentSource=BedrockRerankInlineDocumentSource(
                            textDocument=BedrockRerankTextDocument(text=document),
                            type="TEXT",
                        ),
                        type="INLINE",
                    )
                )
            else:
                _sources.append(
                    BedrockRerankSource(
                        inlineDocumentSource=BedrockRerankInlineDocumentSource(jsonDocument=document, type="JSON"),
                        type="INLINE",
                    )
                )
        return _sources

    def _transform_request(self, request_data: RerankRequest) -> BedrockRerankRequest:
        """
        Transform the request from RerankRequest format to Bedrock format.
        """
        _sources: Final = self._transform_sources(request_data.documents)

        return BedrockRerankRequest(
            queries=[
                BedrockRerankQuery(
                    textQuery=BedrockRerankTextQuery(text=request_data.query),
                    type="TEXT",
                )
            ],
            rerankingConfiguration=BedrockRerankConfiguration(
                bedrockRerankingConfiguration=BedrockRerankBedrockRerankingConfiguration(
                    modelConfiguration=BedrockRerankModelConfiguration(modelArn=request_data.model),
                    numberOfResults=request_data.top_n or len(request_data.documents),
                ),
                type="BEDROCK_RERANKING_MODEL",
            ),
            sources=_sources,
        )

    def transform_request(
        self,
        request_data: RerankRequest,
    ) -> BedrockRerankRequest:
        return self._transform_request(request_data)

    def _default_search_units(self, num_document_chunks: int | None) -> int:
        if num_document_chunks is None or num_document_chunks <= 0:
            return 1
        return math.ceil(num_document_chunks / self.MAX_DOCUMENT_CHUNKS_PER_SEARCH_UNIT)

    def _transform_response(self, response: dict, num_document_chunks: int | None = None) -> RerankResponse:
        """
        Transform the response from Bedrock into the RerankResponse format.

        example input:
        {"results":[{"index":0,"relevanceScore":0.6847912669181824},{"index":1,"relevanceScore":0.5980774760246277}]}
        """
        usage: Final = response.get("usage")
        _billed_units = (
            RerankBilledUnits(**usage)
            if usage is not None
            else RerankBilledUnits(search_units=self._default_search_units(num_document_chunks))
        )
        _tokens: Final = RerankTokens(**(usage or {}))
        rerank_meta: Final = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: list[RerankResponseResult] | None = None

        bedrock_results: Final = response.get("results")
        if bedrock_results:
            _results = [
                RerankResponseResult(
                    index=result.get("index"),
                    relevance_score=result.get("relevanceScore"),
                )
                for result in bedrock_results
            ]

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=_results,
            meta=rerank_meta,
        )  # Return response

    def transform_response(
        self,
        response: dict[str, object],  # mutable-ok: mirrors override contract
        num_document_chunks: int | None = None,
    ) -> RerankResponse:
        return self._transform_response(response, num_document_chunks=num_document_chunks)
