import pytest

from litellm.llms.vertex_ai.cost_calculator import _handle_128k_pricing
from litellm.types.utils import ModelInfo, Usage


def test_handle_128k_pricing_long_context_output_uses_prompt_token_threshold():
    """
    Regression test: the long-context (>128k) pricing tier for Vertex AI/Gemini
    models using the legacy `*_above_128k_tokens` fields must be gated by the
    size of the prompt (request context), for both input AND output pricing.

    Previously, the output cost branch incorrectly checked `completion_tokens`
    against the 128k threshold instead of `prompt_tokens`. Since completion
    tokens are almost always far smaller than the long-context threshold, this
    silently billed long-context output at the base (non-premium) rate even
    when the request's prompt clearly exceeded 128k tokens.
    """
    model_info = ModelInfo(
        key="test-model",
        input_cost_per_token=1e-6,
        input_cost_per_token_above_128k_tokens=2e-6,
        output_cost_per_token=4e-6,
        output_cost_per_token_above_128k_tokens=8e-6,
    )
    usage = Usage(prompt_tokens=150_000, completion_tokens=500, total_tokens=150_500)

    prompt_cost, completion_cost = _handle_128k_pricing(model_info=model_info, usage=usage)

    assert prompt_cost == pytest.approx(150_000 * 2e-6)
    assert completion_cost == pytest.approx(500 * 8e-6)


def test_handle_128k_pricing_short_context_uses_base_rate():
    """When the prompt is below the 128k threshold, both input and output
    costs should use the base (non-premium) per-token rate."""
    model_info = ModelInfo(
        key="test-model",
        input_cost_per_token=1e-6,
        input_cost_per_token_above_128k_tokens=2e-6,
        output_cost_per_token=4e-6,
        output_cost_per_token_above_128k_tokens=8e-6,
    )
    usage = Usage(prompt_tokens=1_000, completion_tokens=500, total_tokens=1_500)

    prompt_cost, completion_cost = _handle_128k_pricing(model_info=model_info, usage=usage)

    assert prompt_cost == pytest.approx(1_000 * 1e-6)
    assert completion_cost == pytest.approx(500 * 4e-6)
