from typing import Final

import litellm
from litellm.types.rerank import RerankBilledUnits
from litellm.types.utils import ModelInfo

# together_ai, novita (and any provider that doesn't override calculate_rerank_cost)
# resolve to this config class for rerank cost calculation.
_cost_config: Final = litellm.CohereRerankConfig()


def test_token_priced_model_bills_by_tokens_when_query_price_is_absent():
    """
    Regression test: a rerank model priced only via 'input_cost_per_token' (e.g.
    together_ai/Salesforce/Llama-Rank-V1, novita/baai/bge-reranker-v2-m3) must still be
    billed, using its reported 'total_tokens'. Before the fix, calculate_rerank_cost
    only knew about 'input_cost_per_query' + 'search_units' and silently returned
    (0.0, 0.0) for every such model, regardless of actual usage.
    """
    model_info: Final = ModelInfo(input_cost_per_token=1e-07)
    billed_units: Final = RerankBilledUnits(total_tokens=5000)

    prompt_cost, completion_cost = _cost_config.calculate_rerank_cost(
        model="together_ai/Salesforce/Llama-Rank-V1",
        custom_llm_provider="together_ai",
        billed_units=billed_units,
        model_info=model_info,
    )

    assert prompt_cost == 5000 * 1e-07
    assert completion_cost == 0.0


def test_query_priced_model_still_bills_by_query():
    """Existing query-based billing (bedrock, azure_ai, cohere, vertex_ai) is unchanged."""
    model_info: Final = ModelInfo(input_cost_per_query=0.002, input_cost_per_token=1e-06)
    billed_units: Final = RerankBilledUnits(search_units=3)

    prompt_cost, completion_cost = _cost_config.calculate_rerank_cost(
        model="rerank-english-v3.0",
        custom_llm_provider="cohere",
        billed_units=billed_units,
        model_info=model_info,
    )

    assert prompt_cost == 3 * 0.002
    assert completion_cost == 0.0


def test_dual_priced_model_falls_back_to_tokens_when_search_units_missing():
    """
    A model priced with both metrics (e.g. nvidia_nim) but whose response only reports
    'total_tokens' (no 'search_units') must bill from the token price, not silently zero.
    """
    model_info: Final = ModelInfo(input_cost_per_query=0.002, input_cost_per_token=2e-06)
    billed_units: Final = RerankBilledUnits(total_tokens=10)

    prompt_cost, completion_cost = _cost_config.calculate_rerank_cost(
        model="nvidia_nim/nvidia/llama-3.2-nv-rerankqa-1b-v2",
        custom_llm_provider="nvidia_nim",
        billed_units=billed_units,
        model_info=model_info,
    )

    assert prompt_cost == 10 * 2e-06
    assert completion_cost == 0.0


def test_no_applicable_pricing_or_usage_returns_zero():
    """No model_info, no billed_units, or neither metric present -> cost stays zero."""
    assert _cost_config.calculate_rerank_cost(model="x") == (0.0, 0.0)
    assert _cost_config.calculate_rerank_cost(model="x", billed_units=RerankBilledUnits(search_units=3)) == (0.0, 0.0)
    assert _cost_config.calculate_rerank_cost(
        model="x", model_info=ModelInfo(), billed_units=RerankBilledUnits(search_units=3)
    ) == (0.0, 0.0)
