"""Versioned agent prompts, backed by YAML with optional Phoenix PromptOps sync.

Prompt text is never embedded in application code. Each named prompt lives in
`prompts/<name>.yaml` (repo root) and is the fallback/source of truth. On
startup the app best-effort syncs those YAML prompts into Arize Phoenix's
prompt hub so they can be reviewed and edited there, then immediately pulls
the (possibly UI-edited) versions back via `refresh_from_phoenix` so they
take precedence over YAML at read time -- see `app.main.lifespan`.

Phoenix is always best-effort here: any failure (unreachable service,
incompatible client/server version, network error) is logged and swallowed,
never raised, so the app can start and prompts can be read with no services
running at all.
"""

import logging
import re
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

# Matches `{name}` placeholders in a template, e.g. "{question}", "{history}".
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


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
        """Render prompt `name`, substituting every `{var}` placeholder.

        Substitution happens in a single pass over the template: every
        placeholder is matched and replaced against the *original* template
        text, so a substituted value that itself contains a
        `{placeholder}`-looking substring is never re-scanned or clobbered by
        a later replacement.

        Raises:
            ValueError: if the template declares a placeholder for which no
                keyword value was supplied (e.g. a typo'd variable name).
                Braces that only appear inside substituted *values* are not
                placeholders and never trigger this error.
        """
        text = self.template(name)
        declared = set(_PLACEHOLDER_RE.findall(text))
        unresolved: set[str] = set()

        def _substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            unresolved.add(key)
            return match.group(0)

        result = _PLACEHOLDER_RE.sub(_substitute, text)
        if unresolved:
            raise ValueError(
                f"prompt {name!r}: missing value(s) for placeholder(s) "
                f"{sorted(unresolved)} (declared: {sorted(declared)}, "
                f"provided: {sorted(variables)})"
            )
        return result

    # --- Phoenix integration (best-effort) ---

    def _phoenix_client(self):
        if self._client is None:
            from phoenix.client import Client

            self._client = Client(base_url=get_settings().phoenix_base_url)
        return self._client

    def _current_phoenix_template(self, client, name: str) -> str | None:
        """Return the template text currently stored in Phoenix for `name`,
        or None if that prompt does not exist there yet.
        """
        try:
            prompt = client.prompts.get(prompt_identifier=name)
        except ValueError:
            return None  # prompt not found -- nothing to compare against
        messages = prompt.format().get("messages", [])
        return messages[0]["content"] if messages else None

    def sync_to_phoenix(self) -> None:
        """Register YAML prompts in Phoenix so they can be reviewed/edited there.

        Idempotent on content: for each prompt, the current live template
        (if any) is fetched and compared against the YAML template first;
        a new Phoenix prompt version is only created when the text actually
        differs (or the prompt does not exist yet), so re-running this on
        every app startup does not create an unbounded pile of byte-identical
        versions.

        The installed arize-phoenix-client (3.x) builds prompt versions from a
        sequence of chat messages rather than a raw template string:
        `PromptVersion(messages, model_name=..., model_provider=...,
        template_format=...)`, and `client.prompts.create(name=..., version=...,
        prompt_description=...)` creates a new version under that name
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
            created, unchanged = 0, 0
            for name, data in self._yaml.items():
                template = data["template"]
                current = self._current_phoenix_template(client, name)
                if current == template:
                    unchanged += 1
                    logger.info("phoenix prompt %r unchanged, sync skipped", name)
                    continue
                client.prompts.create(
                    name=name,
                    version=PromptVersion(
                        [{"role": "system", "content": template}],
                        model_name=settings.llm_model,
                        model_provider=_PHOENIX_MODEL_PROVIDER,
                        template_format="NONE",
                    ),
                    prompt_description=data.get("description"),
                )
                created += 1
            logger.info(
                "phoenix prompt sync: %d version(s) created, %d unchanged",
                created,
                unchanged,
            )
        except Exception as exc:
            logger.warning("phoenix prompt sync skipped: %s", exc)

    def refresh_from_phoenix(self) -> None:
        """Pull latest prompt versions from Phoenix (UI edits win over YAML).

        `client.prompts.get(prompt_identifier=name)` returns a `PromptVersion`;
        `.format()` (with no variables, against a `template_format="NONE"`
        version) renders it back to `{"messages": [...]}` without attempting
        substitution, so `messages[0]["content"]` is the raw template text.

        Mutates this instance's `_phoenix_templates` in place, so calling it
        again on the same (`lru_cache`d, process-wide) `PromptManager` picks
        up new Phoenix edits immediately -- no process restart required.
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


def refresh_prompts() -> None:
    """Force the process-wide PromptManager to re-pull from Phoenix now.

    Convenience entry point for callers that need a fresh pull without a
    process restart (e.g. an admin endpoint added later): the underlying
    `PromptManager` is `lru_cache`d and mutated in place, so this updates
    the exact instance every other `get_prompt_manager()` caller already
    holds a reference to.
    """
    get_prompt_manager().refresh_from_phoenix()
