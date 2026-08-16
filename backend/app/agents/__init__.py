"""CrewAI agent layer: the LLM handle, the search tool and the crew stages
that the corrective-RAG pipeline (`app.pipelines.agentic`) orchestrates.

Importing CrewAI is done here, once, inside `no_env_pollution()`: `crewai.llm`
calls `dotenv.load_dotenv()` at module scope, which loads the repo-root `.env`
into the real process environment and would then outrank pydantic-settings for
every subsequent `get_settings()` call (see `app.core.env_guard` for the full
account). Every submodule below does a plain `from crewai import ...`, which
hits the already-imported, already-cleaned module -- so this guard covers the
whole package as long as it stays the first CrewAI import in the process.
"""

from app.core.env_guard import no_env_pollution

with no_env_pollution():
    import crewai as _crewai  # noqa: F401  (imported for its side effects only)
