import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_character
from litellm.types.utils import Usage


def test_cost_per_character_above_128k_uses_characters_not_tokens():
    """
    For character-priced Vertex AI models, output cost above the 128k-token
    threshold must scale with `completion_characters`, not `usage.completion_tokens`.

    Regression test: the above-128k branch previously multiplied the
    per-character rate by `usage.completion_tokens`, silently under-billing
    long responses whenever a token count differs from the character count
    (which it always does).
    """
    model = "vertex-ai-char-pricing-above-128k-test-model"
    litellm.register_model(
        {
            model: {
                "input_cost_per_character": 0.0000025,
                "input_cost_per_character_above_128k_tokens": 0.000005,
                "output_cost_per_character": 0.0000075,
                "output_cost_per_character_above_128k_tokens": 0.000015,
                "litellm_provider": "vertex_ai",
                "mode": "chat",
            }
        }
    )

    # completion_characters * 4 > 128_000 tokens triggers the above-128k tier.
    completion_characters = 40_000
    # A token count far below the character count, matching what a real
    # tokenizer produces (~4 characters per token) and making the two
    # possible calculations (characters vs tokens) clearly distinguishable.
    usage = Usage(prompt_tokens=250, completion_tokens=10_000)

    _, completion_cost = cost_per_character(
        model=model,
        custom_llm_provider="vertex_ai",
        usage=usage,
        prompt_characters=1_000,
        completion_characters=completion_characters,
    )

    expected_completion_cost = completion_characters * 0.000015
    assert completion_cost == expected_completion_cost

    # Guard against the regression directly: the buggy calculation
    # (tokens * rate) must NOT match the returned cost.
    buggy_completion_cost = usage.completion_tokens * 0.000015
    assert completion_cost != buggy_completion_cost
