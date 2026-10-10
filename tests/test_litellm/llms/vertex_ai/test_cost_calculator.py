import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_token
from litellm.types.utils import Usage


def test_cost_per_token_bills_output_at_above_128k_tier_when_prompt_crosses_threshold(
    monkeypatch,
):
    """Google selects the >128k pricing tier from the prompt length alone, and that one
    decision governs both the input and output rate for the whole request
    (https://ai.google.dev/gemini-api/docs/pricing). A long prompt with a short reply
    must still bill the completion at the above-128k output rate.

    Before the fix, ``_handle_128k_pricing`` picked the output tier from
    ``completion_tokens`` instead of ``prompt_tokens``, so a short reply to a
    >128k-token prompt was silently billed at the cheaper base output rate.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            "vertex_ai/fake-128k-model": {
                "litellm_provider": "vertex_ai",
                "mode": "chat",
                "input_cost_per_token": 1.25e-6,
                "output_cost_per_token": 5e-6,
                "input_cost_per_token_above_128k_tokens": 2.5e-6,
                "output_cost_per_token_above_128k_tokens": 1e-5,
            },
        },
    )

    usage = Usage(prompt_tokens=200_000, completion_tokens=500, total_tokens=200_500)
    prompt_cost, completion_cost = cost_per_token(
        model="fake-128k-model",
        custom_llm_provider="vertex_ai",
        usage=usage,
    )

    assert prompt_cost == 200_000 * 2.5e-6
    assert completion_cost == 500 * 1e-5


def test_cost_per_token_bills_output_at_base_tier_when_prompt_under_threshold(
    monkeypatch,
):
    """A prompt under the 128k threshold keeps both rates at the base tier, even when a
    large completion alone would have crossed 128k under the old (buggy) per-leg check."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            "vertex_ai/fake-128k-model": {
                "litellm_provider": "vertex_ai",
                "mode": "chat",
                "input_cost_per_token": 1.25e-6,
                "output_cost_per_token": 5e-6,
                "input_cost_per_token_above_128k_tokens": 2.5e-6,
                "output_cost_per_token_above_128k_tokens": 1e-5,
            },
        },
    )

    usage = Usage(prompt_tokens=1_000, completion_tokens=200_000, total_tokens=201_000)
    prompt_cost, completion_cost = cost_per_token(
        model="fake-128k-model",
        custom_llm_provider="vertex_ai",
        usage=usage,
    )

    assert prompt_cost == 1_000 * 1.25e-6
    assert completion_cost == 200_000 * 5e-6
