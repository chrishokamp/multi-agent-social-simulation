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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.system_prompt: str = system_prompt
        self.strategy: dict[str, Any] = dict(strategy or {})
        # Let the agent remember the environment it saw last time
        self._last_environment: Mapping[str, Any] | None = None
        self._client, self.model_name = client_for_endpoint()
        self._utility_history: list[float] = []

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
        
        if environment is None:
            return  # no environment to learn from

        history = []
        for run_id, run in environment["runs"]:
            history.append(f"###########\nRUN {run_id}:")
            for m in run["messages"]:
                msg = f"{m['agent']}: {m['message']}"
                history.append(msg)
            history.append("###########\n")

        # history = "\n".join(history[-10:])  # truncate to last 10 messages

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI prompt-optimizer. Rewrite the prompt "
                    "to achieve a lower final price next time. Respond with "
                    "ONLY the new prompt. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CURRENT PROMPT:\n{self.system_prompt}\n\n"
                    f"LAST DIALOGUE (truncated):\n{history}\n\n"
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

    async def _reflect_privately(self, last_public_msg: str) -> str:
        """
        Ask the LLM for a one-sentence silent reflection about the
        latest public message.  Completely private – never sent back
        into the chat.
        """
        prompt = (
            "You are thinking silently as " + self.name + ". "
            "In ONE short sentence, note what you believe or plan "
            "after reading:\n“" + last_public_msg + "”"
        )
        response = self._client.chat.completions.create(
            model=self.model_name or "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    async def think_and_print(self, running_history: list[dict[str, str]]) -> None:
        """
        Produce the reflection, print it, and attach it to the most
        recent entry of `running_history` (so the simulation can store
        it later).
        """
        thought = await self._reflect_privately(running_history[-1]["msg"])
        print(f"[{self.name} – private] {thought}", flush=True)
        running_history[-1]["thought"] = thought


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

        final_price = environment["outputs"]["final_price"]
        max_price   = self.strategy["max_price"]

        if environment["outputs"]["deal_reached"] is False:
            # No deal reached, so no utility
            return 0.0

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
        final_price = environment["outputs"]["final_price"]
        target      = self.strategy["target_price"]

        if environment["outputs"]["deal_reached"] is False:
            # No deal reached, so no utility
            return 0.0

        utility = min(final_price / target, 1.0)
        return utility