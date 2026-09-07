from typing import Final

from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult


def _search_response() -> SearchResponse:
    return SearchResponse(
        results=[SearchResult(title="t", url="u", snippet="s", date=None, last_updated=None)],
        object="search",
    )


def test_tavily_advanced_search_bills_the_advanced_rate() -> None:
    """
    The cost map prices tavily/search-advanced at 0.016, double the base
    tavily/search rate of 0.008. A request sent with search_depth="advanced"
    must be billed at the advanced rate, not silently fall back to the base one.
    """
    base_cost: Final[float] = completion_cost(
        completion_response=_search_response(),
        model="tavily/search",
        custom_llm_provider="tavily",
        call_type="search",
        optional_params={"search_depth": "basic"},
    )
    advanced_cost: Final[float] = completion_cost(
        completion_response=_search_response(),
        model="tavily/search",
        custom_llm_provider="tavily",
        call_type="search",
        optional_params={"search_depth": "advanced"},
    )

    assert base_cost == 0.008
    assert advanced_cost == 0.016


def test_linkup_deep_search_bills_the_deep_rate() -> None:
    """
    linkup/search-deep (0.05867) is roughly 10x linkup/search (0.00587).
    depth="deep" must resolve to the deep-search entry.
    """
    standard_cost: Final[float] = completion_cost(
        completion_response=_search_response(),
        model="linkup/search",
        custom_llm_provider="linkup",
        call_type="search",
        optional_params={"depth": "standard"},
    )
    deep_cost: Final[float] = completion_cost(
        completion_response=_search_response(),
        model="linkup/search",
        custom_llm_provider="linkup",
        call_type="search",
        optional_params={"depth": "deep"},
    )

    assert standard_cost == 0.00587
    assert deep_cost == 0.05867


def test_unknown_mode_value_falls_back_to_base_price() -> None:
    """A mode value with no matching cost-map entry must not raise or silently
    resolve to some other model; it should use the base per-provider price."""
    cost: Final[float] = completion_cost(
        completion_response=_search_response(),
        model="tavily/search",
        custom_llm_provider="tavily",
        call_type="search",
        optional_params={"search_depth": "nonexistent-mode"},
    )

    assert cost == 0.008
