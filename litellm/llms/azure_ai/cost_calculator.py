"""
Azure AI cost calculation helper.
Handles Azure AI Foundry Model Router flat cost and other Azure AI specific pricing.
"""

from typing import Optional, Tuple

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def _is_azure_model_router(model: str) -> bool:
    """
    Check if the model is Azure AI Foundry Model Router.

    Detects patterns like:
    - "azure-model-router"
    - "model-router"
    - "model_router/<actual-model>"
    - "model-router/<actual-model>"

    Args:
        model: The model name

    Returns:
        bool: True if this is a model router model
    """
    model_lower = model.lower()
    return (
        "model-router" in model_lower
        or "model_router" in model_lower
        or model_lower == "azure-model-router"
    )


def calculate_azure_model_router_flat_cost(model: str, prompt_tokens: int) -> float:
    """
    Calculate the flat cost for Azure AI Foundry Model Router.

    Args:
        model: The model name (should be a model router model)
        prompt_tokens: Number of prompt tokens

    Returns:
        float: The flat cost in USD, or 0.0 if not applicable
    """
    if not _is_azure_model_router(model):
        return 0.0

    # Get the model router pricing from model_prices_and_context_window.json
    # Use "model_router" as the key (without actual model name suffix)
    model_info = get_model_info(model="model_router", custom_llm_provider="azure_ai")
    router_flat_cost_per_token = model_info.get("input_cost_per_token", 0)

    if router_flat_cost_per_token and router_flat_cost_per_token > 0:
        return prompt_tokens * router_flat_cost_per_token

    return 0.0


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: Optional[float] = 0.0,
    request_model: Optional[str] = None,
    service_tier: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Calculate the cost per token for Azure AI models.

    This returns only the cost of the underlying model, i.e. what generic_cost_per_token
    computes. The Azure AI Foundry Model Router flat infrastructure fee is NOT included here;
    it is calculated separately via AzureModelRouterConfig.calculate_additional_costs and added
    once by completion_cost's _get_additional_costs step, so it is never double counted.

    Args:
        model: str, the model name without provider prefix (from response)
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
        request_model: Optional[str], the original request model name (to detect router usage)

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        ValueError: If the model is not found in the cost map and cost cannot be calculated
            (except for Model Router models where we return 0 cost, since the flat routing
            cost is accounted for elsewhere)
    """
    # "model_router"/"azure-model-router" aliases are the router itself, not a billable model.
    # Their entry in model_prices_and_context_window.json exists only so
    # calculate_azure_model_router_flat_cost can look up the flat per-token rate; pricing it
    # here too would double count that same flat fee once _get_additional_costs adds it.
    if _is_azure_model_router(model):
        return 0.0, 0.0

    # Calculate base cost using generic cost calculator
    # This may raise an exception if the model is not in the cost map
    try:
        return generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="azure_ai",
            service_tier=service_tier,
        )
    except Exception as e:
        # The response can come back with the router alias instead of the actual routed-to
        # model (e.g. when Azure hasn't resolved it yet). In that case there's no per-model
        # cost to add - only the flat routing fee applies, handled elsewhere.
        if request_model is not None and _is_azure_model_router(request_model):
            verbose_logger.debug(
                f"Azure AI Model Router: model '{model}' not in cost map, returning 0 cost for the underlying model. Error: {e}"
            )
            return 0.0, 0.0
        # Re-raise for non-router models - they should have pricing defined
        raise
