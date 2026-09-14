"""
Vertex AI Image Edit Cost Calculator
"""

from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    calculate_image_response_web_search_cost,
)
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Vertex AI image edit cost calculator.

    Mirrors vertex_ai image generation pricing: use provider token usage
    (`usageMetadata`) when the model bills per token, otherwise fall back to
    the flat per-image rate (`output_cost_per_image`).
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    model_info: Final = litellm.get_model_info(
        model=model,
        custom_llm_provider="vertex_ai",
    )

    web_search_cost: Final = calculate_image_response_web_search_cost(
        image_response=image_response,
        custom_llm_provider="vertex_ai",
        model_info=model_info,
    )

    token_based_cost: Final = calculate_image_response_cost_from_usage(
        model=model,
        image_response=image_response,
        custom_llm_provider="vertex_ai",
    )
    if token_based_cost is not None:
        return token_based_cost + web_search_cost

    output_cost_per_image: Final[float] = model_info.get("output_cost_per_image") or 0.0
    num_images: Final = len(image_response.data or [])
    return output_cost_per_image * num_images + web_search_cost
