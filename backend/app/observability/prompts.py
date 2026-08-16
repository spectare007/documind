"""Versioned agent prompts, backed by YAML with optional Phoenix PromptOps sync.

Prompt text is never embedded in application code. Each named prompt lives in
`prompts/<name>.yaml` (repo root) and is the fallback/source of truth. On
startup the app best-effort syncs those YAML prompts into Arize Phoenix's
prompt hub so they can be reviewed and edited there; `refresh_from_phoenix`
pulls any such UI edits back so they take precedence at read time.

Phoenix is always best-effort here: any failure (unreachable service,
incompatible client/server version, network error) is logged and swallowed,
never raised, so the app can start and prompts can be read with no services
running at all.
"""

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.config import get_settings

logger = logging.getLogger(__name__)

PROMPT_NAMES = ["router", "rewriter", "grader", "synthesizer", "hallucination_checker"]

# Model provider tag Phoenix associates with each synced prompt version. The
# backend calls local Ollama models (see Settings.ollama_base_url), so this
# is the accurate provider label for Phoenix's prompt-hub UI/metadata; it
# does not affect how PromptManager.get() resolves or formats templates.
_PHOENIX_MODEL_PROVIDER = "OLLAMA"


class PromptManager:
    """YAML-backed prompts, optionally overridden by Phoenix's prompt hub.

    Phoenix is best-effort: sync/refresh failures are logged, never raised,
    and the git-versioned YAML files remain the source-of-truth fallback.
    """

    def __init__(self, prompts_dir: Path | None = None, client=None) -> None:
        self.prompts_dir = prompts_dir or get_settings().prompts_dir
        self._yaml: dict[str, dict] = {}
        self._phoenix_templates: dict[str, str] = {}
        self._client = client
        self._load_yaml()

    def _load_yaml(self) -> None:
        for name in PROMPT_NAMES:
            path = self.prompts_dir / f"{name}.yaml"
            with path.open(encoding="utf-8") as f:
                self._yaml[name] = yaml.safe_load(f)

    def template(self, name: str) -> str:
        if name in self._phoenix_templates:
            return self._phoenix_templates[name]
        return self._yaml[name]["template"]

    def version(self, name: str) -> str:
        source = "phoenix" if name in self._phoenix_templates else "yaml"
        return f"{source}:v{self._yaml[name].get('version', 1)}"

    def get(self, name: str, **variables: str) -> str:
        text = self.template(name)
        for key, value in variables.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    # --- Phoenix integration (best-effort) ---

    def _phoenix_client(self):
        if self._client is None:
            from phoenix.client import Client

            self._client = Client(base_url=get_settings().phoenix_base_url)
        return self._client

    def sync_to_phoenix(self) -> None:
        """Register YAML prompts in Phoenix so they can be reviewed/edited there.

        The installed arize-phoenix-client (3.x) builds prompt versions from a
        sequence of chat messages rather than a raw template string:
        `PromptVersion(messages, model_name=..., model_provider=...,
        template_format=...)`, and `client.prompts.create(name=..., version=...,
        prompt_description=...)` upserts a new version under that name
        (creating the prompt on first call). Each YAML prompt is synced as a
        single system message. `template_format="NONE"` tells Phoenix not to
        attempt its own mustache/f-string substitution -- this PromptManager
        owns `{var}` substitution via `get()`, so Phoenix must treat the
        template as opaque text.
        """
        try:
            from phoenix.client.types import PromptVersion

            client = self._phoenix_client()
            settings = get_settings()
            for name, data in self._yaml.items():
                client.prompts.create(
                    name=name,
                    version=PromptVersion(
                        [{"role": "system", "content": data["template"]}],
                        model_name=settings.llm_model,
                        model_provider=_PHOENIX_MODEL_PROVIDER,
                        template_format="NONE",
                    ),
                    prompt_description=data.get("description"),
                )
            logger.info("synced %d prompts to phoenix", len(self._yaml))
        except Exception as exc:
            logger.warning("phoenix prompt sync skipped: %s", exc)

    def refresh_from_phoenix(self) -> None:
        """Pull latest prompt versions from Phoenix (UI edits win over YAML).

        `client.prompts.get(prompt_identifier=name)` returns a `PromptVersion`;
        `.format()` (with no variables, against a `template_format="NONE"`
        version) renders it back to `{"messages": [...]}` without attempting
        substitution, so `messages[0]["content"]` is the raw template text.
        """
        try:
            client = self._phoenix_client()
            for name in PROMPT_NAMES:
                prompt = client.prompts.get(prompt_identifier=name)
                messages = prompt.format().get("messages", [])
                if messages:
                    self._phoenix_templates[name] = messages[0]["content"]
        except Exception as exc:
            logger.warning("phoenix prompt refresh skipped: %s", exc)


@lru_cache
def get_prompt_manager() -> PromptManager:
    return PromptManager()
