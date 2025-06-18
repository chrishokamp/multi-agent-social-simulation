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


class UtilityAgent(AssistantAgent, ABC):
    """
    Base class that adds an overridable `compute_utility` hook.
    Child classes should fill in the details for their own task.
    """
    def __init__(
        self,
        *args,
        strategy: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strategy: dict[str, Any] = dict(strategy or {})
        # Let the agent remember the environment it saw last time
        self._last_environment: Mapping[str, Any] | None = None

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

    @property
    def last_utility(self) -> float | None:
        if self._last_environment is None:
            return None
        return self.compute_utility(self._last_environment)


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

        try:
            final_price = environment["outputs"]["final_price"]
            max_price   = self.strategy["max_price"]
        except (KeyError, TypeError):              # graceful fallback
            return 0.0

        # Normalise to [0, 1]: 1 ⇒ huge saving, 0 ⇒ paid max price.
        return 1.0 - (final_price / max_price)


class SellerAgent(UtilityAgent):
    """
    Utility = revenue / target.
    """

    def compute_utility(self, environment=None):
        env = environment or self._last_environment or {}
        try:
            final_price = env["outputs"]["final_price"]
            target      = self.strategy["target_price"]
        except (KeyError, TypeError):
            return 0.0
        # Cap at 1 so overshooting does not dominate later averaging
        return min(final_price / target, 1.0)