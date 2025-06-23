"""
Utility-aware agents.

Every agent inherits the full behaviour of `AssistantAgent` *plus*
a domain-specific `compute_utility()` method.  Nothing else in the
framework has to change – you can still pass them anywhere a normal
`AssistantAgent` is expected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping
import json

from autogen.agentchat import AssistantAgent

from utils import client_for_endpoint


class UtilityAgent(AssistantAgent, ABC):
    """
    Base class that adds an overridable `compute_utility` hook.
    Child classes should fill in the details for their own task.
    """
    def __init__(
        self,
        system_prompt: str,
        *args,
        strategy: Mapping[str, Any] | None = None,
        llm_config=None,
        model: str | None = None,
        optimization_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, system_message=system_prompt, llm_config=llm_config, **kwargs)
        self.system_prompt: str = system_prompt
        self.strategy: dict[str, Any] = dict(strategy or {})
        # Let the agent remember the environment it saw last time
        self._last_environment: Mapping[str, Any] | None = None
        self._client, self.model_name = client_for_endpoint(model=model)
        self._utility_history: list[float] = []
        self.optimization_prompt: str | None = optimization_prompt

    def compute_utility(
        self,
        environment: Mapping[str, Any] | None = None,
    ) -> float:
        """
        Return a scalar utility measuring *this* agent’s satisfaction.

        Concrete subclasses may use anything they find in `environment`
        (for example `environment["outputs"]` or the full chat history).

        The default simply returns 0 – override me!
        """
        # Keep a reference so the agent can see its previous round later on
        self._last_environment = environment
        return 0.0

    def learn_from_feedback(
        self,
        utility: float,
        environment: Mapping[str, Any] | None = None
    ) -> None:
        
        if environment is None or "runs" not in environment:
            return  # no environment to learn from

        # Check if there are any runs to learn from
        if not environment["runs"]:
            return  # no previous runs
            
        history_lines = []
        for run_id, run in environment["runs"]:
            history_lines.append(f"#### RUN {run_id} ####")
            if "messages" in run and run["messages"]:
                for m in run["messages"]:
                    # Handle different message formats
                    if isinstance(m, dict):
                        agent_name = m.get('agent', m.get('name', 'Unknown'))
                        message_content = m.get('message', m.get('content', ''))
                        history_lines.append(f"{agent_name}: {message_content}")
                    else:
                        history_lines.append(str(m))
            history_lines.append("")
        history = "\n".join(history_lines)

        # Use custom optimization prompt if provided, otherwise use default
        optimization_content = self.optimization_prompt or (
            "You are an expert prompt engineer tasked with maximizing agent utility. "
            "Your goal is to rewrite the agent's prompt to achieve the HIGHEST POSSIBLE UTILITY SCORE. "
            "Focus on aggressive negotiation tactics, protecting private information, and walking away from bad deals. "
            "Use the agent's strategy, conversation history, and utility score to identify what worked and what didn't. "
            "The new prompt should be more assertive, business-like, and utility-maximizing than the current one. "
            "Remember: Higher utility = Better performance. Respond ONLY with the new prompt text."
        )
        
        messages = [
            {
                "role": "system",
                "content": optimization_content,
            },
            {
                "role": "user",
                "content": (
                    f"CURRENT PROMPT:\n{self.system_prompt}\n\n"
                    f"STRATEGY:\n{json.dumps(self.strategy)}\n\n"
                    f"FULL HISTORY:\n{history}\n\n"
                    f"UTILITY ACHIEVED: {utility:.3f}\n\n"
                    "Rewrite now:"
                ),
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        new_prompt = response.choices[0].message.content.strip()
        self.system_prompt = new_prompt
        return


# ----------- example agents -------------

class BuyerAgent(UtilityAgent):
    """
    Scores itself on the *money saved* in a negotiation.

    Assumes:
      environment["outputs"]["final_price"] – price actually paid
      self.strategy["max_price"]            – highest acceptable price
    """

    def compute_utility(
        self,
        environment: Mapping[str, Any] | None = None,
    ) -> float:
        if environment is None:
            environment = self._last_environment or {}
        self._last_environment = environment

        # Return 0 if environment doesn't have the required structure
        if "outputs" not in environment:
            return 0.0
        
        outputs = environment["outputs"]
        if "final_price" not in outputs or "deal_reached" not in outputs:
            return 0.0

        final_price = outputs["final_price"]
        max_price   = self.strategy["max_price"]

        if outputs["deal_reached"] is False:
            # No deal reached - return 0 for neutral outcome
            return 0.0

        # Convert final_price to float if it's a string
        if isinstance(final_price, str):
            final_price = float(final_price)

        # Normalise to [0, 1]: 1 ⇒ huge saving, 0 ⇒ paid max price.
        utility = 1.0 - (final_price / max_price)
        return utility


class SellerAgent(UtilityAgent):
    """
    Utility = revenue / target.
    """

    def compute_utility(self, environment=None):
        environment = environment or self._last_environment or {}
        self._last_environment = environment
        
        # Return 0 if environment doesn't have the required structure
        if "outputs" not in environment:
            return 0.0
        
        outputs = environment["outputs"]
        if "final_price" not in outputs or "deal_reached" not in outputs:
            return 0.0
            
        final_price = outputs["final_price"]
        target      = self.strategy["target_price"]

        if outputs["deal_reached"] is False:
            # No deal reached - return 0 for neutral outcome
            return 0.0

        # Convert final_price to float if it's a string
        if isinstance(final_price, str):
            final_price = float(final_price)

        utility = min(final_price / target, 1.0)
        return utility