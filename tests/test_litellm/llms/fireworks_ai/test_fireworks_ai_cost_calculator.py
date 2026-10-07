"""
Test suite for Fireworks AI cost calculation functionality.

Covers:
- Flat per-token pricing (no prompt caching).
- Discounting cache-hit tokens at `cache_read_input_token_cost` for models
  that support Fireworks AI prompt caching (e.g. kimi-k2p5, glm-4p7).
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.fireworks_ai.cost_calculator import (
    cost_per_token as fireworks_ai_cost_per_token,
)
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.fixture(autouse=True)
def setup_model_cost_map():
    """Load the local model cost map so pricing lookups do not hit the network."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")


def test_fireworks_ai_flat_pricing_no_cache():
    """
    Without any cached tokens, the full prompt token count is billed at the
    regular `input_cost_per_token` rate.
    """
    model = "fireworks_ai/accounts/fireworks/models/kimi-k2p5"
    usage = Usage(prompt_tokens=1000, completion_tokens=200)

    prompt_cost, completion_cost = fireworks_ai_cost_per_token(model=model, usage=usage)

    model_info = litellm.get_model_info(model)
    expected_prompt_cost = 1000 * model_info["input_cost_per_token"]
    expected_completion_cost = 200 * model_info["output_cost_per_token"]

    assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
    assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)


def test_fireworks_ai_discounts_cached_tokens():
    """
    Regression test: Fireworks AI bills cache-hit tokens at the discounted
    `cache_read_input_token_cost` rate instead of the full `input_cost_per_token`
    rate. kimi-k2p5 prices cached reads at a fraction of the regular input price,
    so billing all prompt tokens at the full rate overcharges the caller.
    """
    model = "fireworks_ai/accounts/fireworks/models/kimi-k2p5"
    prompt_tokens = 1000
    cached_tokens = 400
    completion_tokens = 200

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )

    prompt_cost, completion_cost = fireworks_ai_cost_per_token(model=model, usage=usage)

    model_info = litellm.get_model_info(model)
    input_cost_per_token = model_info["input_cost_per_token"]
    cache_read_input_token_cost = model_info["cache_read_input_token_cost"]
    assert cache_read_input_token_cost is not None
    # Sanity check this model actually prices cache reads below the base rate,
    # otherwise this test would not catch the bug.
    assert cache_read_input_token_cost < input_cost_per_token

    billable_tokens = prompt_tokens - cached_tokens
    expected_prompt_cost = (
        billable_tokens * input_cost_per_token
        + cached_tokens * cache_read_input_token_cost
    )
    expected_completion_cost = completion_tokens * model_info["output_cost_per_token"]

    assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
    assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    # The pre-fix behavior billed every prompt token at the full input rate,
    # ignoring the cache discount entirely. Make sure we are strictly cheaper
    # than that now.
    uncached_prompt_cost = prompt_tokens * input_cost_per_token
    assert prompt_cost < uncached_prompt_cost


def test_fireworks_ai_no_cache_pricing_falls_back_to_flat_rate():
    """
    Models without a `cache_read_input_token_cost` entry should still bill
    cache-hit tokens at the regular input rate (no discount available), matching
    pre-existing behavior for models that don't support prompt caching pricing.
    """
    model = "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"
    model_info = litellm.get_model_info(model)
    assert model_info.get("cache_read_input_token_cost") is None

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400),
    )

    prompt_cost, _ = fireworks_ai_cost_per_token(model=model, usage=usage)
    expected_prompt_cost = 1000 * model_info["input_cost_per_token"]

    assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
