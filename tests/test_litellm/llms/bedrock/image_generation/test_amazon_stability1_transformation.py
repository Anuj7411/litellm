from openai.types.image import Image

from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.bedrock.image_generation.amazon_stability1_transformation import (
    AmazonStabilityConfig,
)
from litellm.types.utils import ImageResponse


def _one_image_response() -> ImageResponse:
    return ImageResponse(data=[Image(b64_json="abc")])


def test_stability_xl_v0_default_size_bills_512():
    """
    stability.stable-diffusion-xl-v0 is only priced in model_prices_and_context_window.json
    at 512-x-512 and max-x-max; there is no 1024-x-1024 entry. A request that doesn't specify
    a size must still resolve to a priced bucket instead of raising.
    """
    cost = AmazonStabilityConfig.cost_calculator(
        model="stability.stable-diffusion-xl-v0",
        image_response=_one_image_response(),
    )
    assert cost == 0.018


def test_stability_xl_v0_explicit_size_still_respected():
    cost = AmazonStabilityConfig.cost_calculator(
        model="stability.stable-diffusion-xl-v0",
        image_response=_one_image_response(),
        size="max-x-max",
    )
    assert cost == 0.036


def test_stability_xl_v1_default_size_still_bills_1024():
    """v1 is only priced at 1024-x-1024, so its default must stay unchanged."""
    cost = AmazonStabilityConfig.cost_calculator(
        model="stability.stable-diffusion-xl-v1",
        image_response=_one_image_response(),
    )
    assert cost == 0.04


def test_route_image_generation_cost_calculator_bedrock_v0_no_size():
    """End-to-end: a real bedrock/stability.stable-diffusion-xl-v0 image_generation() call
    with no size kwarg must not raise while calculating cost."""
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="stability.stable-diffusion-xl-v0",
        completion_response=_one_image_response(),
        custom_llm_provider="bedrock",
    )
    assert cost == 0.018
