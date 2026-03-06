"""LLM escalation — called ONLY when no feasible path exists.

The whole point of self-healing-router is to avoid LLM calls.
This module is the last resort: when Dijkstra can't find any path,
we escalate to an LLM to decide what to do (retry, skip, abort, 
or suggest an alternative approach).
"""

from __future__ import annotations

from typing import Any, Callable

from .types import EscalationResult, EscalationCallback


class EscalationHandler:
    """Handles LLM escalation when no routing path exists.
    
    By design, this should be called rarely. The paper shows
    93% fewer LLM calls vs ReAct — this handler accounts for
    the remaining 7%.
    """

    def __init__(
        self,
        callback: EscalationCallback | None = None,
        max_escalations: int = 3,
    ) -> None:
        self._callback = callback
        self._max_escalations = max_escalations
        self._escalation_count = 0
        self._history: list[dict[str, Any]] = []

    def escalate(
        self,
        failed_tool: str,
        attempted_paths: list[list[str]],
        context: dict[str, Any] | None = None,
    ) -> EscalationResult:
        """Escalate to LLM for decision.
        
        Args:
            failed_tool: The tool that caused the routing failure
            attempted_paths: Paths that were tried and failed
            context: Additional context (tool outputs so far, errors, etc)
            
        Returns:
            EscalationResult with the LLM's decision
        """
        self._escalation_count += 1
        
        escalation_context = {
            "failed_tool": failed_tool,
            "attempted_paths": attempted_paths,
            "escalation_number": self._escalation_count,
            "max_escalations": self._max_escalations,
            **(context or {}),
        }
        
        self._history.append(escalation_context)

        # If no callback, default to abort
        if self._callback is None:
            return EscalationResult(
                action="abort",
                detail=f"No escalation handler configured. Failed at '{failed_tool}' "
                       f"after trying {len(attempted_paths)} path(s).",
            )

        # If we've exceeded max escalations, force abort
        if self._escalation_count > self._max_escalations:
            return EscalationResult(
                action="abort",
                detail=f"Max escalations ({self._max_escalations}) exceeded.",
            )

        # Call the LLM
        return self._callback(failed_tool, attempted_paths, escalation_context)

    @property
    def escalation_count(self) -> int:
        return self._escalation_count

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        """Reset escalation counter (e.g., between tasks)."""
        self._escalation_count = 0
        self._history.clear()


def default_escalation(
    failed_tool: str,
    attempted_paths: list[list[str]],
    context: dict[str, Any],
) -> EscalationResult:
    """Default escalation handler that just aborts. 
    
    Replace this with your LLM call:
    
        from openai import OpenAI
        
        def my_escalation(failed_tool, paths, context):
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a tool routing assistant. A tool has failed "
                               "and no alternative path exists. Decide: retry, skip, or abort."
                }, {
                    "role": "user", 
                    "content": f"Tool '{failed_tool}' failed. Paths tried: {paths}. Context: {context}"
                }]
            )
            # Parse response into EscalationResult
            return EscalationResult(action="abort", detail=response.choices[0].message.content)
    """
    return EscalationResult(
        action="abort",
        detail=f"No path to goal. Tool '{failed_tool}' failed. "
               f"Tried {len(attempted_paths)} path(s). Override with custom escalation handler.",
    )
