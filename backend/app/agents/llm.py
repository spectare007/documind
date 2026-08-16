"""Shared CrewAI LLM handle pointing at the local Ollama server.

The model string carries the provider prefix (`ollama/<model>`) that CrewAI
routes on; `base_url` points it at our own Ollama rather than a default
localhost guess, and `timeout` bounds a single completion -- CPU inference of
a 3B model takes seconds to tens of seconds per stage and the pipeline runs
up to eight stages, so this must be generous.

`temperature=0.0` is deliberate: every stage either classifies (router,
grader, groundedness checker) or must quote context verbatim (synthesizer).
Sampling buys nothing there and makes the parsers' job harder.

NOTE ON THE RETURN TYPE: on crewai 1.x `LLM(...)` is a *factory*, not a
constructor -- `LLM.__new__` inspects the model string and returns an
instance of a native provider class instead of `crewai.llm.LLM` itself.
With a `base_url` set, `ollama/...` resolves to
`crewai.llms.providers.openai_compatible.completion.OpenAICompatibleCompletion`,
which talks to Ollama's OpenAI-compatible `/v1` endpoint (verified live).
So the honest annotation is the shared base class, `BaseLLM`; annotating
`LLM` would be a type error that happens to typecheck.

That routing also has an observability consequence worth knowing about: the
native provider calls the `openai` SDK directly and never touches LiteLLM,
so `LiteLLMInstrumentor` (wired up in `app.observability.tracing`) produces
nothing and Phoenix traces contain crew/agent/tool spans but no LLM spans.
"""

from functools import lru_cache

from crewai import LLM, BaseLLM

from app.core.config import get_settings


@lru_cache
def get_crew_llm() -> BaseLLM:
    """Process-wide CrewAI LLM. Cached: construction reads settings and the
    object holds no per-request state, so every stage can share one.
    """
    s = get_settings()
    return LLM(
        model=f"ollama/{s.llm_model}",
        base_url=s.ollama_base_url,
        temperature=0.0,
        timeout=s.llm_timeout_seconds,
    )
