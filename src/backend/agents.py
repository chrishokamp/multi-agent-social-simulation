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
        super().__init__(
            *args, system_message=system_prompt, llm_config=llm_config, **kwargs
        )
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
        return [
            run["output_variables"]["utility"] for run in self._last_environment["runs"]
        ]

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

    def get_history(self, environment: Mapping[str, Any], n_runs: int) -> str:
        history_lines = []
        for run in environment["runs"][-n_runs:]:  # only look at the last n runs
            run_id = run["run_id"]
            msg = f"###########\nRUN {run_id}:\n"
            for m in run["messages"]:
                # Handle different message formats
                if isinstance(m, dict):
                    agent_name = m.get("agent", m.get("name", "Unknown"))
                    message_content = m.get("message", m.get("content", ""))
                    history_lines.append(f"{agent_name}: {message_content}")
                else:
                    history_lines.append(str(m))
        history = "\n".join(history_lines)
        return history

    def learn_from_feedback(self, environment: Mapping[str, Any] | None = None) -> None:

        if environment is None or "runs" not in environment:
            return  # no environment to learn from

        # Check if there are any runs to learn from
        if not environment["runs"]:
            return  # no previous runs

        history = self.get_history(environment, n_runs=5)
        most_recent_run = environment["runs"][-1]
        utility = most_recent_run["output_variables"].get("utility", {})[self.name]

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
        logger.info(
            f"Agent {self.name} learned new prompt: {new_prompt} (previous: {self.system_prompt})"
        )
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
            max_price = float(self.strategy["max_price"])
            # Normalise to [0, 1]: 1 ⇒ huge saving, 0 ⇒ paid max price.
            utility = 1.0 - (final_price / max_price)

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][
            -1
        ]["output_variables"].get("utility", {})
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
            target = float(self.strategy["target_price"])
            utility = min(final_price / target, 1.0)

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][
            -1
        ]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


class NegotiationCoachAgent(UtilityAgent, ABC):
    """
    A base class for agents that receive strategies from a coach that
    they can use to improve their performance in future rounds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negotiation_strategies: list = []
        self.original_system_prompt = self.system_prompt

    def learn_from_feedback(self, environment: Mapping[str, Any] | None = None) -> None:
        
        agent_strategies_key = f"strategies_{self.name.lower()}"

        if environment is None or "runs" not in environment:
            return  # no environment to learn from

        # Check if there are any runs to learn from
        if not environment["runs"]:
            return  # no previous runs
        environment.setdefault(agent_strategies_key, [])
        
        history = self.get_history(environment, n_runs=1)
        most_recent_run = environment["runs"][-1]
        utility = most_recent_run["output_variables"].get("utility", {})[self.name]

        tag = (
            "great"
            if utility > 0.8
            else "okay" if utility > 0.4 else "poor" if utility > 0 else "loss"
        )

        # Use custom optimization prompt if provided, otherwise use default
        optimization_content = (
            "You are a seasoned negotiation coach.\n"
            f"Previous strategies:\n- "
            + "\n- ".join(environment.get(agent_strategies_key, []))
            + "\n"
            "Analyse the transcript and devise exactly ONE new negotiation strategy "
            f"sentence the {self.name} could apply in a *future* negotiation to get a better price.\n"
            "If neither party uttered 'Yes, deal!', that means no deal was reached. "
            "In that case, focus on how to reach a good deal faster next time.\n"
            "Start with an action verb and do NOT duplicate prior strategies. "
            "Do NOT mention specific prices, names or budgets from the dialogue.\n"
            f"{self._get_private_constraints()}\n."
            f"The {self.name}'s normalised utility for this deal was {utility:.2f} ({tag}).\n"
            "• If utility was 'loss' or 'poor', focus on improvement. "
            "• If 'great', suggest how to replicate or slightly enhance success. \n"
            "Include one recognised negotiation tactic (e.g., anchoring, mirroring, time-pressure) that fits what you observed in the transcript."
            "Think step-by-step and return ONLY that single negotiation strategy sentence."
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
                    f"UTILITY ACHIEVED: {utility:.3f}({tag})\n\n"
                    "New negotiation strategy:"
                ),
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model_name or "gpt-4o",
            messages=messages,
        )
        new_strategy = response.choices[0].message.content.strip()
        logger.info(f"Agent {self.name} learned new strategy: {new_strategy}")
        self.negotiation_strategies.append(new_strategy)
        environment[agent_strategies_key].append(new_strategy)
        self.system_prompt = self.original_system_prompt + (
            "\nNegotiation strategies:\n" + "\n".join(self.negotiation_strategies)
            if self.negotiation_strategies
            else ""
        )
        return


class NegotiationCoachBuyerAgent(NegotiationCoachAgent):
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
            budget = float(self.strategy.get("budget", 0))
            max_price = float(most_recent_run["output_variables"].get("final_price", 0))
            if budget == 0:
                utility = 0.0
            else:
                utility = (budget - max_price) / budget

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][
            -1
        ]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment

    def _get_private_constraints(self) -> str:
        return f"Buyer budget was ${self.strategy.get('budget', 0)}"


class NegotiationCoachSellerAgent(NegotiationCoachAgent):
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
            max_price = float(self.strategy["asking_price"])
            seller_floor = float(self.strategy.get("asking_price", 0))
            # Normalise to [0, 1]: 1 ⇒ paid seller_floor, 0 ⇒ paid max_price.
            denominator = max_price - seller_floor
            if denominator == 0:
                utility = 0.0
            else:
                utility = (final_price - seller_floor) / denominator

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][
            -1
        ]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment

    def _get_private_constraints(self) -> str:
        return f"Seller floor was ${self.strategy.get('seller_floor', 0)}"
