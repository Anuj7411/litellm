"""
Test suite for Fireworks AI cost calculation functionality.

Tests the cost calculation for Fireworks AI models including:
- Flat-rate pricing for models without prompt-caching pricing.
- Discounting cached tokens at 'cache_read_input_token_cost' instead of the
  full 'input_cost_per_token' rate.
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


class TestFireworksAICostCalculator:
    """Test suite for Fireworks AI cost calculation functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing by loading it locally."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_fireworks_ai_flat_pricing_without_cache(self):
        """
        Without any cached tokens, cost should simply be prompt/completion
        tokens multiplied by their respective flat rates.
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        prompt_cost, completion_cost = fireworks_ai_cost_per_token(
            model="fireworks_ai/glm-4p7", usage=usage
        )

        model_info = litellm.get_model_info("fireworks_ai/glm-4p7")
        expected_prompt_cost = 1000 * model_info["input_cost_per_token"]
        expected_completion_cost = 500 * model_info["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_fireworks_ai_discounts_cached_tokens(self):
        """
        Regression test: glm-4p7 defines a discounted 'cache_read_input_token_cost'.
        Cached tokens must be billed at that discounted rate, not at the full
        'input_cost_per_token' rate.

        Before the fix, cost_per_token multiplied the full prompt_tokens count
        (cached + uncached) by input_cost_per_token, ignoring the cache
        discount entirely and overbilling callers whose requests hit cache.
        """
        usage = Usage(
            prompt_tokens=1000,  # 400 cached + 600 new
            completion_tokens=200,
            total_tokens=1200,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400),
        )

        prompt_cost, completion_cost = fireworks_ai_cost_per_token(
            model="fireworks_ai/glm-4p7", usage=usage
        )

        model_info = litellm.get_model_info("fireworks_ai/glm-4p7")
        input_cost_per_token = model_info["input_cost_per_token"]
        cache_read_cost = model_info["cache_read_input_token_cost"]
        assert cache_read_cost is not None and cache_read_cost < input_cost_per_token

        expected_prompt_cost = (600 * input_cost_per_token) + (400 * cache_read_cost)
        expected_completion_cost = 200 * model_info["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

        # Explicitly pin down the overbilling the bug caused: billing every
        # cached token at the full input rate costs strictly more.
        buggy_prompt_cost = 1000 * input_cost_per_token
        assert prompt_cost < buggy_prompt_cost

    def test_fireworks_ai_unmapped_model_falls_back_to_base_model_category(self):
        """
        An unmapped model name (e.g. a raw <n>b parameter-count model) should
        fall back to the generic size-based pricing category instead of
        raising, and still go through the cache-aware calculator.
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        prompt_cost, completion_cost = fireworks_ai_cost_per_token(
            model="some-unmapped-7b-model", usage=usage
        )

        model_info = litellm.get_model_info(
            "fireworks-ai-4.1b-to-16b", custom_llm_provider="fireworks_ai"
        )
        expected_prompt_cost = 1000 * model_info["input_cost_per_token"]
        expected_completion_cost = 500 * model_info["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)
