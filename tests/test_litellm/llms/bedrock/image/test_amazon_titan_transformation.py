import pytest

import litellm
from litellm.llms.bedrock.image_generation.amazon_titan_transformation import (
    AmazonTitanImageGenerationConfig,
)
from litellm.types.utils import ImageResponse
from openai.types.image import Image as ImageObject


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """Amazon Titan's tiered pricing fields aren't in every published cost map snapshot -
    force the local backup map so tier lookups in this file are deterministic."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.mark.parametrize(
    "model, size, quality, expected_cost",
    [
        # v1: base rate below the 512x512 threshold
        ("amazon.titan-image-generator-v1", "512x512", "standard", 0.008),
        # v1: premium rate below the 512x512 threshold
        ("amazon.titan-image-generator-v1", "512x512", "premium", 0.01),
        # v1: base rate above the 512x512 threshold
        ("amazon.titan-image-generator-v1", "1024x1024", "standard", 0.01),
        # v1: premium rate above the 512x512 threshold
        ("amazon.titan-image-generator-v1", "1024x1024", "premium", 0.012),
        # v2: base rate below the 1024x1024 threshold
        ("amazon.titan-image-generator-v2:0", "1024x1024", "standard", 0.008),
        # v2: premium rate below the 1024x1024 threshold
        ("amazon.titan-image-generator-v2:0", "1024x1024", "premium", 0.01),
        # v2: base rate above the 1024x1024 threshold
        ("amazon.titan-image-generator-v2:0", "2048x2048", "standard", 0.01),
        # v2: premium rate above the 1024x1024 threshold
        ("amazon.titan-image-generator-v2:0", "2048x2048", "premium", 0.012),
    ],
)
def test_cost_calculator_bills_quality_and_pixel_tiers(
    model, size, quality, expected_cost
):
    """
    Regression test for undercharging Titan image generations: the cost calculator
    used to ignore quality/size and always bill the flat `output_cost_per_image` rate,
    even though AWS (and litellm's own pricing map) charge more for premium quality
    and for images above the model's pixel threshold.
    """
    image_response = ImageResponse(data=[ImageObject(b64_json="base64_image")])
    optional_params = {"imageGenerationConfig": {"quality": quality}}

    cost = AmazonTitanImageGenerationConfig.cost_calculator(
        model=model,
        image_response=image_response,
        size=size,
        optional_params=optional_params,
    )

    assert cost == pytest.approx(expected_cost)


def test_cost_calculator_scales_with_number_of_images():
    image_response = ImageResponse(
        data=[
            ImageObject(b64_json="base64_image_1"),
            ImageObject(b64_json="base64_image_2"),
        ]
    )

    cost = AmazonTitanImageGenerationConfig.cost_calculator(
        model="amazon.titan-image-generator-v1",
        image_response=image_response,
        size="1024x1024",
        optional_params={"imageGenerationConfig": {"quality": "premium"}},
    )

    assert cost == pytest.approx(0.012 * 2)


def test_cost_calculator_defaults_to_base_rate_without_optional_params():
    image_response = ImageResponse(data=[ImageObject(b64_json="base64_image")])

    cost = AmazonTitanImageGenerationConfig.cost_calculator(
        model="amazon.titan-image-generator-v1",
        image_response=image_response,
    )

    assert cost == pytest.approx(0.008)


def test_cost_calculator_no_images_is_free():
    image_response = ImageResponse(data=[])

    cost = AmazonTitanImageGenerationConfig.cost_calculator(
        model="amazon.titan-image-generator-v1",
        image_response=image_response,
        size="1024x1024",
        optional_params={"imageGenerationConfig": {"quality": "premium"}},
    )

    assert cost == 0.0
