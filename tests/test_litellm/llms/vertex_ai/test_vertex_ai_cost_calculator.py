import os

import pytest

import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_character
from litellm.types.utils import Usage


def test_cost_per_character_128k_threshold_uses_tokens_not_characters():
    """
    cost_per_character() decides whether a request crosses the 128k-token pricing
    threshold by converting the character count to an estimated token count first
    (1 token ~= 4 characters, so tokens = characters / 4), then compares that
    against 128_000.

    Regression test for a bug where the conversion multiplied by 4 instead of
    dividing, so any prompt over ~32,000 characters (only ~8,000 tokens) was
    incorrectly treated as being over the 128k-token threshold - 16x too
    sensitive. For "medlm-large", which prices per-character but has no
    "above_128k" character rate configured, this made the above-128k pricing
    assertion fail and the cost silently fall back to token-based pricing (which
    medlm-large also does not define), billing the request as $0 instead of the
    correct per-character cost.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_info = litellm.get_model_info("medlm-large")
    assert model_info["input_cost_per_character"] > 0
    assert model_info.get("input_cost_per_character_above_128k_tokens") is None

    prompt_characters = 40_000  # ~10,000 tokens - well under the 128k-token threshold
    usage = Usage(prompt_tokens=10_000, completion_tokens=0, total_tokens=10_000)

    prompt_cost, _ = cost_per_character(
        model="medlm-large",
        custom_llm_provider="vertex_ai",
        usage=usage,
        prompt_characters=prompt_characters,
        completion_characters=0,
    )

    assert prompt_cost == pytest.approx(prompt_characters * model_info["input_cost_per_character"])
