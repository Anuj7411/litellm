"""
Transformation logic for Amazon Titan Image Generation.
"""

import re
import types
from typing import List, Optional, Tuple

from openai.types.image import Image

from litellm.utils import get_model_info
from litellm.types.llms.bedrock import (
    AmazonNovaCanvasImageGenerationConfig,
    AmazonTitanImageGenerationRequestBody,
    AmazonTitanTextToImageParams,
)
from litellm.types.utils import ImageResponse

PIXEL_TIER_COST_FIELD_PATTERN = re.compile(
    r"output_cost_per_image_above_(\d+)_and_(\d+)_pixels$"
)


class AmazonTitanImageGenerationConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0
    """

    cfg_scale: Optional[int] = None
    seed: Optional[float] = None
    steps: Optional[List[str]] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(
        self,
        cfg_scale: Optional[int] = None,
        seed: Optional[float] = None,
        steps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @classmethod
    def _is_titan_model(cls, model: Optional[str] = None) -> bool:
        """
        Returns True if the model is a Titan model

        Titan models follow this pattern:

        """
        if model and "amazon.titan" in model:
            return True
        return False

    @classmethod
    def get_supported_openai_params(cls, model: Optional[str] = None) -> List:
        return ["size", "n", "quality"]

    @classmethod
    def map_openai_params(
        cls,
        non_default_params: dict,
        optional_params: dict,
    ):
        from typing import Any, Dict

        image_generation_config: Dict[str, Any] = {}
        for k, v in non_default_params.items():
            if k == "size" and v is not None:
                width, height = v.split("x")
                image_generation_config["width"] = int(width)
                image_generation_config["height"] = int(height)
            elif k == "n" and v is not None:
                image_generation_config["numberOfImages"] = v
            elif (
                k == "quality" and v is not None
            ):  # 'auto', 'hd', 'standard', 'high', 'medium', 'low'
                if v in ("hd", "premium", "high"):
                    image_generation_config["quality"] = "premium"
                elif v in ("standard", "medium", "low"):
                    image_generation_config["quality"] = "standard"

        if image_generation_config:
            optional_params["imageGenerationConfig"] = image_generation_config
        return optional_params

    @classmethod
    def transform_request_body(
        cls,
        text: str,
        optional_params: dict,
    ) -> AmazonTitanImageGenerationRequestBody:
        from typing import Any, Dict

        image_generation_config = optional_params.pop("imageGenerationConfig", {})
        negative_text = optional_params.pop("negativeText", None)
        text_to_image_params: Dict[str, Any] = {"text": text}
        if negative_text:
            text_to_image_params["negativeText"] = negative_text
        task_type = optional_params.pop("taskType", "TEXT_IMAGE")
        user_specified_image_generation_config = optional_params.pop(
            "imageGenerationConfig", {}
        )
        image_generation_config = {
            **image_generation_config,
            **user_specified_image_generation_config,
        }
        return AmazonTitanImageGenerationRequestBody(
            taskType=task_type,
            textToImageParams=AmazonTitanTextToImageParams(**text_to_image_params),  # type: ignore
            imageGenerationConfig=AmazonNovaCanvasImageGenerationConfig(
                **image_generation_config
            ),
        )

    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        image_list: List[Image] = []
        for image in response_dict["images"]:
            _image = Image(b64_json=image)
            image_list.append(_image)

        model_response.data = image_list

        return model_response

    @classmethod
    def _get_requested_dimensions(
        cls, size: Optional[str], image_generation_config: dict
    ) -> Optional[Tuple[int, int]]:
        width = image_generation_config.get("width")
        height = image_generation_config.get("height")
        if width is not None and height is not None:
            return width, height
        if size:
            try:
                str_width, str_height = size.split("x")
                return int(str_width), int(str_height)
            except ValueError:
                return None
        return None

    @classmethod
    def _resolve_cost_field(
        cls, model_info: dict, size: Optional[str], optional_params: Optional[dict]
    ) -> str:
        """
        Titan bills a higher rate for premium quality, and again once the image exceeds the
        model's pixel threshold (512x512 for v1, 1024x1024 for v2) - see the `output_cost_per_image_*`
        fields for these models in model_prices_and_context_window.json.
        """
        image_generation_config = (optional_params or {}).get(
            "imageGenerationConfig"
        ) or {}
        is_premium = image_generation_config.get("quality") == "premium"
        dimensions = cls._get_requested_dimensions(size, image_generation_config)

        if dimensions is not None:
            width, height = dimensions
            for field_name, field_value in model_info.items():
                if field_value is None:
                    continue
                match = PIXEL_TIER_COST_FIELD_PATTERN.fullmatch(field_name)
                if not match:
                    continue
                if width > int(match.group(1)) and height > int(match.group(2)):
                    premium_field_name = f"{field_name}_and_premium_image"
                    if is_premium and model_info.get(premium_field_name) is not None:
                        return premium_field_name
                    return field_name

        if (
            is_premium
            and model_info.get("output_cost_per_image_premium_image") is not None
        ):
            return "output_cost_per_image_premium_image"
        return "output_cost_per_image"

    @classmethod
    def cost_calculator(
        cls,
        model: str,
        image_response: ImageResponse,
        size: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> float:
        if not image_response.data:
            return 0.0
        model_info = get_model_info(model=model)
        cost_field = cls._resolve_cost_field(model_info, size, optional_params)
        output_cost_per_image = model_info.get(cost_field) or 0.0
        num_images = len(image_response.data)
        return output_cost_per_image * num_images
