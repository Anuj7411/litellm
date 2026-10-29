import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_character
from litellm.types.utils import Usage


@pytest.fixture
def above_128k_character_priced_model():
    model = "test-vertex-above-128k-character-pricing-model"
    litellm.register_model(
        {
            f"vertex_ai/{model}": {
                "max_tokens": 2097152,
                "max_input_tokens": 2097152,
                "max_output_tokens": 8192,
                "input_cost_per_character": 0.00000125,
                "input_cost_per_character_above_128k_tokens": 0.0000025,
                "output_cost_per_character": 0.000005,
                "output_cost_per_character_above_128k_tokens": 0.00001,
                "litellm_provider": "vertex_ai-language-models",
                "mode": "chat",
            }
        }
    )
    yield model


def test_get_model_info_returns_input_cost_per_character_above_128k_tokens(
    above_128k_character_priced_model,
):
    """
    Regression test: get_model_info() must surface
    'input_cost_per_character_above_128k_tokens' for a model whose cost map
    entry sets it, the same way it already surfaces the sibling
    'output_cost_per_character_above_128k_tokens' field. Before the fix,
    get_model_info() silently dropped the input-side field.
    """
    model_info = litellm.get_model_info(
        model=above_128k_character_priced_model, custom_llm_provider="vertex_ai"
    )

    assert model_info.get("input_cost_per_character_above_128k_tokens") == 0.0000025


def test_cost_per_character_prompt_above_128k_uses_character_tier_rate(
    above_128k_character_priced_model,
):
    """
    Regression test for the user-visible effect of the bug above: when the
    prompt is long enough to hit Vertex AI's >128k-token character-pricing
    tier, cost_per_character() must bill it at
    input_cost_per_character_above_128k_tokens, not silently fall back to
    token-based pricing (which is what happened when get_model_info()
    dropped the field, since the assertion checking for it would fail and
    the code would catch that failure and recompute the prompt cost from
    usage.prompt_tokens via cost_per_token() instead).
    """
    model = above_128k_character_priced_model
    prompt_characters = 600_000  # > 128_000 * 4, so the >128k tier applies
    prompt_tokens = 150_000  # deliberately different from the token-based cost
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
        total_tokens=prompt_tokens + 10,
    )

    prompt_cost, _ = cost_per_character(
        model=model,
        custom_llm_provider="vertex_ai",
        usage=usage,
        prompt_characters=prompt_characters,
        completion_characters=10,
    )

    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")
    expected_prompt_cost = (
        prompt_characters * model_info["input_cost_per_character_above_128k_tokens"]
    )

    assert prompt_cost == pytest.approx(expected_prompt_cost)
