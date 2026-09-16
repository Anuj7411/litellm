import pytest

from litellm.llms.bedrock.image_generation.amazon_stability1_transformation import (
    AmazonStabilityConfig,
)
from litellm.types.utils import ImageResponse


def _single_image_response() -> ImageResponse:
    response = ImageResponse()
    response.data = [{"b64_json": "abc"}]
    return response


@pytest.mark.parametrize("size", ["1024x1024", "1024-x-1024"])
def test_cost_calculator_normalizes_openai_style_size(size):
    """Regression test: an OpenAI-style size (e.g. "1024x1024") must price the same
    as the model_prices_and_context_window.json key format ("1024-x-1024"), not raise
    "model isn't mapped" and silently drop the charge."""
    cost = AmazonStabilityConfig.cost_calculator(
        model="stability.stable-diffusion-xl-v1",
        image_response=_single_image_response(),
        size=size,
        optional_params={},
    )
    assert cost == pytest.approx(0.04)


@pytest.mark.parametrize("size", ["512x512", "512-x-512"])
def test_cost_calculator_normalizes_size_for_max_steps_tier(size):
    cost = AmazonStabilityConfig.cost_calculator(
        model="stability.stable-diffusion-xl-v0",
        image_response=_single_image_response(),
        size=size,
        optional_params={"steps": 60},
    )
    assert cost == pytest.approx(0.036)
