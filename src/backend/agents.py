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

    @property
    def utility_history(self) -> list[float]:
        """
        A list of the utility values computed in previous rounds
        """
        return [run["output_variables"]["utility"] for run in self._last_environment["runs"]]

    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> float:
        """
        Return a scalar utility measuring *this* agent’s satisfaction.

        Concrete subclasses may use anything they find in `environment`
        (for example `environment["output_variables"]` or the full chat history).

        """
        # Keep a reference so the agent can see its previous round later on
        self._last_environment = environment
        return environment

    def learn_from_feedback(
        self,
        environment: Mapping[str, Any] | None = None
    ) -> None:
        
        if environment is None or "runs" not in environment:
            return  # no environment to learn from

        # Check if there are any runs to learn from
        if not environment["runs"]:
            return  # no previous runs
        
        most_recent_run = environment["runs"][-1]
        utility = most_recent_run["output_variables"].get("utility", {})[self.name]
            
        history_lines = []
        for run in environment["runs"][-5:]:  # only look at the last 5 runs
            run_id = run["run_id"]
            msg = f"###########\nRUN {run_id}:\n"
            for m in run["messages"]:
                # Handle different message formats
                if isinstance(m, dict):
                    agent_name = m.get('agent', m.get('name', 'Unknown'))
                    message_content = m.get('message', m.get('content', ''))
                    history_lines.append(f"{agent_name}: {message_content}")
                else:
                    history_lines.append(str(m))
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
            model=self.model_name or "gpt-4o",
            messages=messages,
        )

        new_prompt = response.choices[0].message.content.strip()
        logger.info(f"Agent {self.name} learned new prompt: {new_prompt} (previous: {self.system_prompt})")
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
      environment["output_variables"]["final_price"] – price actually paid
      self.strategy["max_price"]            – highest acceptable price
    """

    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        if environment is None or "runs" not in environment:
            return environment  # no environment to learn from

        # Check if there are any runs to learn from
        if not environment["runs"]:
            return environment  # no previous runs

        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["deal_reached"] is False:
            # No deal reached, so no utility
            utility = 0.0
        else:
            final_price = float(most_recent_run["output_variables"]["final_price"])
            max_price   = float(self.strategy["max_price"])
            # Normalise to [0, 1]: 1 ⇒ huge saving, 0 ⇒ paid max price.
            utility = 1.0 - (final_price / max_price)

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


class SellerAgent(UtilityAgent):
    """
    Utility = revenue / target.
    """

    def compute_utility(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        environment = environment or self._last_environment or {}
        if environment is None or "runs" not in environment:
            return environment  # no environment to learn from

        # Check if there are any runs to learn from
        if not environment["runs"]:
            return environment  # no previous runs

        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["deal_reached"] is False:
            # No deal reached, so no utility
            utility = 0.0
        else:
            final_price = float(most_recent_run["output_variables"]["final_price"])
            target      = float(self.strategy["target_price"])
            utility = min(final_price / target, 1.0)
        
        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment
