import pytest
from litellm.llms.bedrock.image_generation.amazon_nova_canvas_transformation import (
    AmazonNovaCanvasConfig,
)
from litellm.types.utils import ImageObject, ImageResponse


def test_transform_request_body_text_to_image():
    params = {
        "imageGenerationConfig": {
            "cfgScale": 7,
            "seed": 42,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
            "textToImageParams": {"negativeText": "blurry"},
        }
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert isinstance(req, dict)
    assert "textToImageParams" in req
    assert req["textToImageParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512


def test_transform_request_body_color_guided():
    params = {
        "taskType": "COLOR_GUIDED_GENERATION",
        "imageGenerationConfig": {
            "cfgScale": 7,
            "seed": 42,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
            "colorGuidedGenerationParams": {
                "colors": ["#FFFFFF"],
                "referenceImage": "img",
                "negativeText": "blurry",
            },
        },
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert "colorGuidedGenerationParams" in req
    assert req["colorGuidedGenerationParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512


def test_transform_request_body_inpainting():
    params = {
        "taskType": "INPAINTING",
        "imageGenerationConfig": {
            "cfgScale": 7,
            "seed": 42,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
            "inpaintingParams": {
                "maskImage": "mask",
                "inputImage": "input",
                "negativeText": "blurry",
            },
        },
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert "inpaintingParams" in req
    assert req["inpaintingParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512


def test_transform_response_dict_to_openai_response():
    response_dict = {"images": ["b64img1", "b64img2"]}
    model_response = ImageResponse()
    result = AmazonNovaCanvasConfig.transform_response_dict_to_openai_response(
        model_response, response_dict
    )
    assert hasattr(result, "data")
    assert len(result.data) == 2
    assert result.data[0].b64_json == "b64img1"


@pytest.mark.parametrize(
    "model,optional_params,expected_cost_per_image",
    [
        # AWS defaults to standard quality at <=1024x1024 when unset.
        ("amazon.nova-canvas-v1:0", {}, 0.04),
        (
            "amazon.nova-canvas-v1:0",
            {"quality": "standard", "width": 1024, "height": 1024},
            0.04,
        ),
        (
            "amazon.nova-canvas-v1:0",
            {"quality": "premium", "width": 1024, "height": 1024},
            0.06,
        ),
        (
            "amazon.nova-canvas-v1:0",
            {"quality": "standard", "width": 2048, "height": 2048},
            0.06,
        ),
        (
            "amazon.nova-canvas-v1:0",
            {"quality": "premium", "width": 2048, "height": 2048},
            0.08,
        ),
        # Cross-region inference ids must price the same as the base model id.
        (
            "us.amazon.nova-canvas-v1:0",
            {"quality": "standard", "width": 512, "height": 512},
            0.04,
        ),
        (
            "us.amazon.nova-canvas-v1:0",
            {"quality": "premium", "width": 2048, "height": 2048},
            0.08,
        ),
        # Only one dimension needs to cross the 1024 threshold to hit high-res pricing.
        (
            "amazon.nova-canvas-v1:0",
            {"quality": "standard", "width": 2048, "height": 512},
            0.06,
        ),
    ],
)
def test_cost_calculator_quality_and_resolution_tiers(
    model, optional_params, expected_cost_per_image
):
    """
    Regression test: Nova Canvas bills per image on a quality (standard/premium) x
    resolution (<=1024x1024 vs <=2048x2048) grid, so a premium 2048x2048 image costs
    twice as much as a standard 1024x1024 image. Before the fix, cost_calculator
    ignored `optional_params` entirely and always billed the flat $0.06 rate,
    overcharging standard/low-res calls and undercharging premium/high-res calls.
    """
    image_response = ImageResponse(data=[ImageObject(b64_json="img")])
    cost = AmazonNovaCanvasConfig.cost_calculator(
        model=model,
        image_response=image_response,
        optional_params=optional_params,
    )
    assert cost == pytest.approx(expected_cost_per_image)


def test_cost_calculator_scales_with_number_of_images():
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")]
    )
    cost = AmazonNovaCanvasConfig.cost_calculator(
        model="amazon.nova-canvas-v1:0",
        image_response=image_response,
        optional_params={"quality": "premium", "width": 2048, "height": 2048},
    )
    assert cost == pytest.approx(0.16)


def test_cost_calculator_falls_back_to_flat_rate_for_unmapped_tier(monkeypatch):
    """
    If a Nova Canvas model id has no tier-specific pricing in the cost map (e.g. a
    newer inference id not yet added), cost_calculator must fall back to that
    model's flat `output_cost_per_image` instead of raising.
    """
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "some.future.nova-canvas-model",
        {
            "litellm_provider": "bedrock",
            "mode": "image_generation",
            "output_cost_per_image": 0.5,
        },
    )
    image_response = ImageResponse(data=[ImageObject(b64_json="img")])
    cost = AmazonNovaCanvasConfig.cost_calculator(
        model="some.future.nova-canvas-model",
        image_response=image_response,
        optional_params={"quality": "premium", "width": 2048, "height": 2048},
    )
    assert cost == pytest.approx(0.5)
