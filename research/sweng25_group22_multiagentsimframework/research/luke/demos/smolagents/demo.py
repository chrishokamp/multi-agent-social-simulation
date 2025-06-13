from smolagents import CodeAgent, ManagedAgent, LiteLLMModel, tool

model = LiteLLMModel(
    model_id = "ollama/llama2",
    num_ctx = 8192
)

p_price: int = 40000

@tool
def get_price() -> int:
    """
    Returns the current price of the car.
    """
    return p_price

@tool
def set_price(price: int) -> None:
    """
    Sets the current price of the car.

    Args:
        price: Updated price.
    """
    p_price = price

seller = CodeAgent(
    tools = [get_price, set_price], 
    model = model, 
    add_base_tools = True
)

buyer = CodeAgent(
    tools = [get_price], 
    model = model, 
    add_base_tools = True
)

managed_buyer = ManagedAgent(
    agent = buyer,
    name = "buyer",
    description = """
        Represents a person who wants to purchase a new car.
    """
)

managed_seller = ManagedAgent(
    agent = seller,
    name = "seller",
    description = """
        Represents the seller for the new car.
    """
)

manager = CodeAgent(
    tools = [],
    model = model,
    managed_agents = [
        managed_buyer,
        managed_seller
    ]
)

manager.run(
    """
    Can you simulate a buyer negotiating the price of a car with a seller using your managed buyer 
    and seller agents?
    """
)

print(p_price)