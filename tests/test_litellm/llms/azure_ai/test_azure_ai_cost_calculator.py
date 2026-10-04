"""
Test Azure AI cost calculator, especially Model Router flat cost.
"""

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.azure_ai.cost_calculator import (
    _is_azure_model_router,
    cost_per_token,
)
from litellm.types.utils import Usage
from litellm.utils import get_model_info

# Get the flat cost from model_prices_and_context_window.json
_model_info = get_model_info(model="model_router", custom_llm_provider="azure_ai")
AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS = (
    _model_info.get("input_cost_per_token", 0) * 1_000_000
)


class TestAzureModelRouterDetection:
    """Test that we correctly identify Azure Model Router models.

    Model Router deployments follow the pattern: model_router/<deployment-name>
    where deployment-name is the Azure deployment (e.g., 'azure-model-router', 'prod-router')
    """

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Deployment names containing 'model-router' or 'model_router'
            ("azure-model-router", True),
            ("AZURE-MODEL-ROUTER", True),
            ("model-router", True),
            ("MODEL-ROUTER", True),
            ("my-model-router-deployment", True),
            ("prod-model_router", True),
            # New pattern: model_router/<deployment-name>
            ("model_router/azure-model-router", True),
            ("model-router/prod-router", True),
            ("model_router/my-deployment", True),
            ("MODEL_ROUTER/AZURE-MODEL-ROUTER", True),
            # Non-router models
            ("gpt-4o", False),
            ("gpt-4o-mini", False),
            ("claude-sonnet-4-5", False),
            ("my-regular-deployment", False),
        ],
    )
    def test_is_azure_model_router(self, model: str, expected: bool):
        """Test Azure Model Router detection."""
        assert _is_azure_model_router(model) == expected


class TestAzureModelRouterPrefix:
    """Test Azure Model Router prefix stripping."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Model router deployments - the deployment name comes after model_router/
            ("model_router/azure-model-router", "azure-model-router"),
            ("model-router/my-router-deployment", "my-router-deployment"),
            ("model_router/prod-router", "prod-router"),
            # Non-router models - should pass through unchanged
            ("gpt-4o", "gpt-4o"),
            ("azure-model-router", "azure-model-router"),
            ("claude-sonnet-4", "claude-sonnet-4"),
        ],
    )
    def test_strip_model_router_prefix(self, model: str, expected: str):
        """Test that model_router prefix is stripped correctly.

        The pattern is: model_router/<deployment-name>
        where deployment-name is the Azure deployment (e.g., 'azure-model-router', 'prod-router')
        """
        from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

        result = AzureFoundryModelInfo.strip_model_router_prefix(model)
        assert result == expected


class TestAzureModelRouterFlatCost:
    """
    Test that `cost_per_token` (the per-model cost function) never bakes the Model
    Router flat routing fee into its result.

    The flat fee is calculated exactly once, by AzureModelRouterConfig.calculate_additional_costs
    via completion_cost's additional_costs step (see TestAzureModelRouterFlatCostNotDoubled). If
    cost_per_token also added it, every Model Router request would be billed the fee twice.
    """

    def test_model_router_flat_cost_basic(self):
        """A bare router placeholder model has no token pricing of its own, so cost_per_token returns 0."""
        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_model_router_flat_cost_large_request(self):
        """Same as above for a larger request: no flat fee leaks into cost_per_token."""
        model = "model-router"
        usage = Usage(
            prompt_tokens=100_000,
            completion_tokens=50_000,
            total_tokens=150_000,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_model_router_flat_cost_1m_tokens(self):
        """Same as above at 1M prompt tokens: no flat fee leaks into cost_per_token."""
        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=1_000_000,
            completion_tokens=100_000,
            total_tokens=1_100_000,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_non_model_router_no_flat_cost(self):
        """Test that non-Model Router models don't get the flat cost."""
        model = "gpt-4o"
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        # No flat cost should be added for non-Model Router models
        # The cost might be 0 or based on the model's pricing
        print(f"Non-Model Router prompt cost: ${prompt_cost:.6f}")
        # We just ensure it doesn't crash and returns valid values
        assert prompt_cost >= 0
        assert completion_cost >= 0

    def test_model_router_with_cached_tokens(self):
        """A bare router placeholder model is still not priced, with or without cached tokens."""
        model = "azure-model-router"
        usage = Usage(
            prompt_tokens=2000,
            completion_tokens=800,
            total_tokens=2800,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=200,
        )

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_router_flat_cost_when_response_has_actual_model(self):
        """
        When the request was made via Model Router but Azure's response names the actual
        model used (e.g. gpt-5-nano), cost_per_token must return ONLY that model's own cost.

        The flat routing fee is intentionally NOT included here; completion_cost adds it
        exactly once via _get_additional_costs. If cost_per_token also added it (the bug),
        the fee would be billed twice for every routed request.
        """
        usage = Usage(
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000,
        )

        # Response model is the actual model Azure used (not a router name)
        response_model = "gpt-5-nano-2025-08-07"
        # Request model is the router - user called azure_ai/model_router/model-router
        request_model = "azure_ai/model_router/model-router"

        prompt_cost, _completion_cost = cost_per_token(
            model=response_model,
            usage=usage,
            request_model=request_model,
        )

        # The flat routing fee must not have leaked into the per-model cost
        flat_cost = (
            usage.prompt_tokens
            * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS
            / 1_000_000
        )
        assert flat_cost == pytest.approx(0.0014, rel=1e-9)

        # prompt_cost must equal gpt-5-nano's own cost for 10k prompt tokens, not
        # gpt-5-nano's cost plus the flat routing fee
        expected_model_only_prompt_cost, _ = generic_cost_per_token(
            model=response_model, usage=usage, custom_llm_provider="azure_ai"
        )
        assert prompt_cost == pytest.approx(expected_model_only_prompt_cost, rel=1e-9)
        assert prompt_cost != pytest.approx(
            expected_model_only_prompt_cost + flat_cost, rel=1e-9
        )


class TestAzureModelRouterCostBreakdown:
    """Test that Azure Model Router flat cost is tracked in cost breakdown."""

    def test_flat_cost_calculation_helper(self):
        """Test that flat cost can be calculated using the helper function."""
        from litellm.llms.azure_ai.cost_calculator import (
            calculate_azure_model_router_flat_cost,
        )

        model = "azure-model-router"
        prompt_tokens = 10000

        # Calculate flat cost using helper function
        flat_cost = calculate_azure_model_router_flat_cost(
            model=model, prompt_tokens=prompt_tokens
        )

        # Expected flat cost
        expected_flat_cost = (
            prompt_tokens * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        assert flat_cost > 0
        assert flat_cost == pytest.approx(expected_flat_cost, rel=1e-9)
        print(f"Flat cost calculated: ${flat_cost:.6f}")

    def test_flat_cost_integration_with_completion_cost(self):
        """Test that flat cost is properly integrated into completion_cost calculation."""
        import litellm
        from litellm.cost_calculator import completion_cost
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        # Create a mock response for azure_ai model router
        response = ModelResponse(
            id="test-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Test response",
                    ),
                )
            ],
            created=1234567890,
            model="azure-model-router",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=5000,
                completion_tokens=2000,
                total_tokens=7000,
            ),
        )

        # Set hidden params for provider
        response._hidden_params = {"custom_llm_provider": "azure_ai"}

        # Calculate cost
        cost = completion_cost(
            completion_response=response,
            model="azure-model-router",
            custom_llm_provider="azure_ai",
        )

        # Expected flat cost
        expected_flat_cost = (
            5000 * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )

        # "azure-model-router" has no token pricing of its own, so the total cost is
        # exactly the flat fee, once. Exact equality here catches it being billed twice.
        assert cost == pytest.approx(expected_flat_cost, rel=1e-9)

    def test_flat_cost_not_doubled_for_literal_router_catalog_key(self):
        """
        Regression test: "model_router"/"model-router" (no "azure-" prefix) are themselves
        present in model_prices_and_context_window.json, priced at the same per-token rate as
        the flat routing fee, purely so calculate_azure_model_router_flat_cost can look up that
        rate. Before the fix, completion_cost billed a response that came back labeled with this
        literal key at double the real Azure invoice: once because generic_cost_per_token priced
        "model-router" as if it were a regular billable model (its catalog entry happens to carry
        the flat rate), and again because _get_additional_costs computed the same flat fee
        on top. This must come out to the flat fee exactly once.
        """
        from litellm.cost_calculator import completion_cost
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        prompt_tokens = 100_000
        response = ModelResponse(
            id="test-456",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content="Test response"),
                )
            ],
            created=1234567890,
            model="model-router",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=50_000,
                total_tokens=prompt_tokens + 50_000,
            ),
        )
        response._hidden_params = {"custom_llm_provider": "azure_ai"}

        cost = completion_cost(
            completion_response=response,
            model="model-router",
            custom_llm_provider="azure_ai",
        )

        expected_flat_cost = (
            prompt_tokens * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )
        assert cost == pytest.approx(expected_flat_cost, rel=1e-9)
        assert cost != pytest.approx(2 * expected_flat_cost, rel=1e-9)

    def test_additional_costs_in_cost_breakdown(self):
        """Test that Azure Model Router flat cost appears in additional_costs dict."""
        from datetime import datetime

        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        # Create logging object with required parameters
        logging_obj = Logging(
            model="azure-model-router",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-123",
            function_id="test-function",
        )

        # Create a mock response for azure_ai model router
        response = ModelResponse(
            id="test-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Test response",
                    ),
                )
            ],
            created=1234567890,
            model="azure-model-router",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=5000,
                completion_tokens=2000,
                total_tokens=7000,
            ),
        )

        # Set hidden params for provider
        response._hidden_params = {"custom_llm_provider": "azure_ai"}

        # Calculate cost with logging object
        cost = completion_cost(
            completion_response=response,
            model="azure-model-router",
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )

        # Check that cost breakdown contains additional_costs
        assert hasattr(logging_obj, "cost_breakdown")
        assert logging_obj.cost_breakdown is not None
        assert "additional_costs" in logging_obj.cost_breakdown
        assert isinstance(logging_obj.cost_breakdown["additional_costs"], dict)

        # Check that the Azure Model Router flat cost is in additional_costs
        additional_costs = logging_obj.cost_breakdown["additional_costs"]
        assert "Azure Model Router Flat Cost" in additional_costs

        # Verify the flat cost value
        expected_flat_cost = (
            5000 * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )
        actual_flat_cost = additional_costs["Azure Model Router Flat Cost"]
        assert actual_flat_cost == pytest.approx(expected_flat_cost, rel=1e-9)

        print(f"Additional costs in breakdown: {additional_costs}")
        print(f"Azure Model Router Flat Cost: ${actual_flat_cost:.6f}")

    def test_additional_costs_when_response_has_actual_model_via_hidden_params(self):
        """additional_costs populated when response has actual model but request was via model router (hidden_params)."""
        from datetime import datetime

        from litellm.cost_calculator import completion_cost
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        logging_obj = Logging(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-123",
            function_id="test-function",
        )
        response = ModelResponse(
            id="test-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content="Hello"),
                )
            ],
            created=1234567890,
            model="gpt-4.1-nano-2025-04-14",
            object="chat.completion",
            usage=Usage(prompt_tokens=5000, completion_tokens=2000, total_tokens=7000),
        )
        response._hidden_params = {
            "custom_llm_provider": "azure_ai",
            "litellm_model_name": "azure_ai/model-router",
        }
        cost = completion_cost(
            completion_response=response,
            model="gpt-4.1-nano-2025-04-14",
            custom_llm_provider="azure_ai",
            litellm_logging_obj=logging_obj,
        )
        expected_flat_cost = (
            5000 * AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        )
        assert cost >= expected_flat_cost
        assert logging_obj.cost_breakdown is not None
        assert "additional_costs" in logging_obj.cost_breakdown
        assert (
            "Azure Model Router Flat Cost"
            in logging_obj.cost_breakdown["additional_costs"]
        )
        assert logging_obj.cost_breakdown["additional_costs"][
            "Azure Model Router Flat Cost"
        ] == pytest.approx(expected_flat_cost, rel=1e-9)

        # Regression guard: the total must be the routed model's own cost plus the
        # flat routing fee exactly ONCE. Before the fix, azure_ai_cost_per_token()
        # baked the flat fee into input_cost AND _get_additional_costs added it again,
        # so `cost` came out as model_cost + 2x flat_cost here.
        model_only_prompt_cost, model_only_completion_cost = generic_cost_per_token(
            model="gpt-4.1-nano-2025-04-14",
            usage=Usage(prompt_tokens=5000, completion_tokens=2000, total_tokens=7000),
            custom_llm_provider="azure_ai",
        )
        expected_total_cost = (
            model_only_prompt_cost + model_only_completion_cost + expected_flat_cost
        )
        assert cost == pytest.approx(expected_total_cost, rel=1e-9)
        assert cost != pytest.approx(expected_total_cost + expected_flat_cost, rel=1e-9)


class TestAzureAIServiceTierCostCalculation:
    """Test that service_tier is passed through Azure AI cost calculation."""

    @pytest.fixture(autouse=True)
    def register_test_model(self):
        import litellm
        litellm.register_model(model_cost={
            "test-azure-ai-model": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "input_cost_per_token_priority": 0.01,
                "output_cost_per_token_priority": 0.02,
                "input_cost_per_token_flex": 0.0005,
                "output_cost_per_token_flex": 0.001,
                "litellm_provider": "azure_ai",
                "max_tokens": 8192,
            }
        })

    def test_service_tier_priority_higher_cost(self):
        """Priority tier should cost more than standard for azure_ai."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(
            model="test-azure-ai-model", usage=usage
        )
        priority_prompt, priority_completion = cost_per_token(
            model="test-azure-ai-model", usage=usage, service_tier="priority"
        )

        assert priority_prompt > standard_prompt
        assert priority_completion > standard_completion

    def test_service_tier_flex_lower_cost(self):
        """Flex tier should cost less than standard for azure_ai."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(
            model="test-azure-ai-model", usage=usage
        )
        flex_prompt, flex_completion = cost_per_token(
            model="test-azure-ai-model", usage=usage, service_tier="flex"
        )

        assert flex_prompt < standard_prompt
        assert flex_completion < standard_completion
