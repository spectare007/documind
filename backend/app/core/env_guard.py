"""Containment for third-party imports that mutate `os.environ`.

--- The one deliberate, documented exception to "no os.environ in app code" ---

Importing CrewAI runs `crewai.llm`'s module-level `dotenv.load_dotenv()`,
which walks up from the current working directory and -- in this repo layout
-- finds and loads the repo-root `.env` (which exists for docker-compose
variable substitution) into the real process environment, e.g.
`DOCUMIND_API_KEY=change-me`. Real env vars outrank pydantic-settings' own
`env_file=".env"` lookup, so that silently overrides every `get_settings()`
call for the rest of the process: API-key auth starts rejecting the
configured key, and the pipeline starts reading a different model name than
the one the app was configured with.

This module is the only place in the codebase that reads or mutates
`os.environ`, and it does so narrowly. It never *adds* configuration from the
environment; it only *removes* keys that (a) were absent immediately before a
guarded import and (b) appeared during it -- i.e. it can only ever undo
pollution caused by that exact import. A real deployment's variables (Docker,
docker-compose, a shell export) are present before the interpreter starts, so
they are captured in `env_before` and never touched. `load_dotenv()` does not
override already-set variables, so newly-appearing keys are the complete set
of damage.

Used by `app.observability.tracing` (which imports the CrewAI OpenInference
instrumentor) and by `app.agents` (which imports CrewAI itself).
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def scrub_env_pollution(env_before: set[str]) -> list[str]:
    """Delete env vars that appeared since `env_before` was snapshotted.

    Returns the names removed, sorted, for logging and assertions.
    """
    leaked = sorted(set(os.environ) - env_before)
    for key in leaked:
        del os.environ[key]
    if leaked:
        logger.warning(
            "scrubbed %d env var(s) leaked by third-party import(s): %s",
            len(leaked), leaked,
        )
    return leaked


@contextmanager
def no_env_pollution() -> Iterator[None]:
    """Run a block (typically a third-party import) and undo any env vars it
    added. The scrub runs even if the block raises, so a partially-completed
    import cannot leave configuration poisoned behind it.
    """
    env_before = set(os.environ)
    try:
        yield
    finally:
        scrub_env_pollution(env_before)
