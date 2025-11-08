"""Utility for loading and serving agent prompts."""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class PromptManager:
    """Centralized loader for agent system and user prompts."""

    def __init__(
        self,
        prompt_path: Optional[str] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        self.base_path = base_path or Path(__file__).resolve().parents[3]
        default_path = self.base_path / "prompts" / "default_prompt.json"

        resolved_path = Path(prompt_path) if prompt_path else default_path
        if not resolved_path.is_absolute():
            resolved_path = (self.base_path / resolved_path).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Prompt file not found at {resolved_path}")

        self.prompt_path = resolved_path
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompts: Dict[str, Dict[str, str]] = json.load(f)

    def get(self, agent: str, prompt_key: str) -> str:
        """Return the raw prompt string for a given agent and prompt key."""
        try:
            return self.prompts[agent][prompt_key]
        except KeyError as exc:
            available_agents = ", ".join(sorted(self.prompts))
            raise KeyError(
                f"Prompt '{prompt_key}' for agent '{agent}' not found. "
                f"Available agents: {available_agents}"
            ) from exc

    def render(self, agent: str, prompt_key: str, **kwargs: Any) -> str:
        """Render a prompt template with the provided keyword arguments."""
        template = self.get(agent, prompt_key)
        if kwargs:
            try:
                return template.format(**kwargs)
            except KeyError as exc:
                missing = exc.args[0]
                raise KeyError(
                    f"Missing placeholder '{missing}' for prompt '{prompt_key}' "
                    f"of agent '{agent}'."
                ) from exc
        return template

    def get_agent_prompts(self, agent: str) -> Dict[str, str]:
        """Return all prompts associated with an agent."""
        if agent not in self.prompts:
            available_agents = ", ".join(sorted(self.prompts))
            raise KeyError(
                f"Agent '{agent}' not found in prompt file. "
                f"Available agents: {available_agents}"
            )
        return self.prompts[agent]
