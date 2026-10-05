"""
Test suite for Fireworks AI cost calculation functionality.
"""

import math

from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


class TestFireworksAICostCalculator:
    """Test suite for Fireworks AI cost calculation functionality."""

    def test_basic_cost_calculation(self):
        """No cache hits: cost should be prompt_tokens * input rate."""
        usage = Usage(prompt_tokens=10000, completion_tokens=500, total_tokens=10500)

        prompt_cost, completion_cost = cost_per_token(
            model="fireworks_ai/accounts/fireworks/models/glm-4p7", usage=usage
        )

        expected_prompt_cost = 10000 * 6e-07
        expected_completion_cost = 500 * 2.2e-06

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_cached_tokens_are_discounted(self):
        """
        Regression test: Fireworks previously billed every prompt token at the
        full input rate, ignoring prompt_tokens_details.cached_tokens and the
        model's cache_read_input_token_cost. This overcharged any request with
        cache hits. Cached tokens must be billed at the discounted cache-read
        rate, not the full input rate.
        """
        usage = Usage(
            prompt_tokens=10000,
            completion_tokens=500,
            total_tokens=10500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=8000),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="fireworks_ai/accounts/fireworks/models/glm-4p7", usage=usage
        )

        # glm-4p7: input=6e-7/tok, cache_read=3e-7/tok (half price), output=2.2e-6/tok
        # 2000 non-cached tokens at full rate + 8000 cached tokens at the discounted rate
        expected_prompt_cost = 2000 * 6e-07 + 8000 * 3e-07
        expected_completion_cost = 500 * 2.2e-06

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

        # Confirms the discount was actually applied: billing all prompt tokens at the
        # full rate (the pre-fix behavior) would produce a higher, wrong cost.
        undiscounted_prompt_cost = 10000 * 6e-07
        assert prompt_cost < undiscounted_prompt_cost

    def test_unmapped_model_falls_back_to_base_model_pricing(self):
        """Unknown model names fall back to the parameter-count pricing category."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        prompt_cost, completion_cost = cost_per_token(
            model="fireworks_ai/some-custom-13b-model", usage=usage
        )

        # "13b" falls into the "fireworks-ai-4.1b-to-16b" pricing category.
        expected_prompt_cost = 100 * 2e-07
        expected_completion_cost = 50 * 2e-07

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)
