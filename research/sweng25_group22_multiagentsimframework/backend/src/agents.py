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

from autogen_agentchat.agents import AssistantAgent

from utils import client_for_endpoint, create_logger
logger = create_logger(__name__)


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.system_prompt: str = system_prompt
        self.strategy: dict[str, Any] = dict(strategy or {})
        # Let the agent remember the environment it saw last time
        self._last_environment: Mapping[str, Any] | None = None
        self._client, self.model_name = client_for_endpoint()
        self._utility_history: list[float] = []

    @property
    def utility_history(self) -> list[float]:
        """
        A list of the utility values computed in previous rounds
        """
        return [run["outputs"]["utility"] for run in self._last_environment["runs"]]

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
        environment: Mapping[str, Any] | None = None
    ) -> None:
        
        if environment is None:
            return  # no environment to learn from

        history = []
        for run in environment["runs"]:
            run_id = run["run_id"]
            msg = f"###########\nRUN {run_id}:\n"
            for m in run["messages"]:
                msg += f"{m['agent']}: {m['message']}\n"
            msg += f"FINAL OUTPUTS: {run['outputs']}\n"
            msg += "###########\n"
            history.append(msg)

        # history = "\n".join(history[-10:])  # truncate to last 10 runs

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI prompt-optimizer. Rewrite the prompt for the agent "
                    f"to achieve a higher utility. Utility with respect to the agent's strategy {self.strategy}. "
                    "Respond with ONLY the new prompt. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CURRENT PROMPT:\n{self.system_prompt}\n\n"
                    f"HISTORY (truncated):\n{history}\n\n"
                    "Rewrite now:"
                ),
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        new_prompt = response.choices[0].message.content.strip()
        logger.info(f"Agent {self.name} learned new prompt: {new_prompt} (previous: {self.system_prompt})")
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
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}

        most_recent_run = environment["runs"][-1]

        final_price = most_recent_run["outputs"]["final_price"]
        max_price   = self.strategy["max_price"]

        if most_recent_run["outputs"]["deal_reached"] is False:
            # No deal reached, so no utility
            utility = 0.0

        # Normalise to [0, 1]: 1 ⇒ huge saving, 0 ⇒ paid max price.
        utility = 1.0 - (final_price / max_price)
        environment["runs"][-1]["outputs"]["utility"] = utility
        self._last_environment = environment
        return environment


class SellerAgent(UtilityAgent):
    """
    Utility = revenue / target.
    """

    def compute_utility(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        environment = environment or self._last_environment or {}
        most_recent_run = environment["runs"][-1]

        final_price = most_recent_run["outputs"]["final_price"]
        target      = self.strategy["target_price"]

        if most_recent_run["outputs"]["deal_reached"] is False:
            # No deal reached, so no utility
            utility = 0.0

        utility = min(final_price / target, 1.0)
        environment["runs"][-1]["outputs"]["utility"] = utility
        self._last_environment = environment
        return environment
