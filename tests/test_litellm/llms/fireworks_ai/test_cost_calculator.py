import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    """Force local model cost map usage for all tests in this file."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    litellm.model_cost = get_model_cost_map(url=litellm.model_cost_map_url)


def test_cost_per_token_discounts_cached_tokens():
    """
    Regression test for cached prompt tokens being billed at the full
    input_cost_per_token rate instead of the discounted
    cache_read_input_token_cost rate.

    fireworks_ai/accounts/fireworks/models/glm-4p7 has
    input_cost_per_token=6e-07 and cache_read_input_token_cost=3e-07, so an
    800-of-1000 cache hit must cost less than billing all 1000 tokens at the
    full rate, and must match the expected blended rate exactly.
    """
    import litellm

    model = "fireworks_ai/accounts/fireworks/models/glm-4p7"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="fireworks_ai")
    input_cost_per_token = model_info["input_cost_per_token"]
    cache_read_input_token_cost = model_info["cache_read_input_token_cost"]
    assert cache_read_input_token_cost is not None
    assert cache_read_input_token_cost < input_cost_per_token

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
    )

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    uncached_tokens = 1000 - 800
    expected_prompt_cost = (
        uncached_tokens * input_cost_per_token + 800 * cache_read_input_token_cost
    )
    full_price_prompt_cost = 1000 * input_cost_per_token

    assert prompt_cost == pytest.approx(expected_prompt_cost)
    # The bug billed every prompt token at the full rate, ignoring the cache
    # discount entirely - assert we are strictly below that to catch a
    # regression back to that behavior.
    assert prompt_cost < full_price_prompt_cost
    assert completion_cost == pytest.approx(100 * model_info["output_cost_per_token"])


def test_cost_per_token_without_cache_hit_unaffected():
    """Models/requests with no cached tokens should be billed at the full input rate."""
    import litellm

    model = "fireworks_ai/accounts/fireworks/models/glm-4p7"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="fireworks_ai")

    usage = Usage(prompt_tokens=500, completion_tokens=50, total_tokens=550)

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx(500 * model_info["input_cost_per_token"])
    assert completion_cost == pytest.approx(50 * model_info["output_cost_per_token"])


def test_cost_per_token_unmapped_model_falls_back_to_pricing_category():
    """Unmapped fireworks_ai models should still resolve pricing via the
    parameter-count category fallback (e.g. '...-70b-...' -> 41.1b-80b tier)."""
    usage = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)

    prompt_cost, completion_cost = cost_per_token(
        model="fireworks_ai/some-unmapped-70b-model", usage=usage
    )

    assert prompt_cost > 0
    assert completion_cost > 0
