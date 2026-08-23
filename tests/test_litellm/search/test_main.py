"""
Tests for litellm/search/main.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "test-api-key")


def _mock_perplexity_response(num_results: int) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": "..."} for i in range(num_results)
        ],
    }
    return mock_response


@pytest.mark.asyncio
async def test_multi_query_search_bills_per_query():
    """
    Regression: litellm.asearch(query=[...]) must bill one input_cost_per_query
    per element of the query list, not a single flat query.

    Perplexity's /search endpoint natively accepts a list of queries in one
    request (billed per query by the provider), so a 3-item query list must
    cost 3x a single query, not 1x.
    """
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _mock_perplexity_response(num_results=3)

        single_query_response = await litellm.asearch(
            query="latest AI news",
            search_provider="perplexity",
        )
        multi_query_response = await litellm.asearch(
            query=["latest AI news", "AI safety research", "AI regulation 2026"],
            search_provider="perplexity",
        )

    single_query_cost = single_query_response._hidden_params["response_cost"]
    multi_query_cost = multi_query_response._hidden_params["response_cost"]

    assert single_query_cost == pytest.approx(0.005)
    assert multi_query_cost == pytest.approx(3 * single_query_cost)
