"""Error classes the app treats as *expected operating conditions*.

`LLM_UNAVAILABLE_ERRORS` lives here rather than in `app.api.query` because
two layers now need the same tuple and neither should import the other:

* `app.api.query` catches it to return a structured 503 instead of FastAPI's
  generic unhandled 500.
* `app.pipelines.agentic` deliberately *re-raises* it out of its own
  catch-all containment, so that the endpoint above still sees it. Upstream
  unavailability is not an "unexpected stage failure" -- it has a designed
  response, and swallowing it into a friendly 200 would silently undo that.

The tuple was built empirically against the two LLM clients this app
actually constructs, not guessed:

- `SimplePipeline`'s `llama_index.llms.ollama.Ollama` calls Ollama over
  `httpx` directly and catches nothing, so connect/read/write/pool timeouts
  and connection-refused all surface as `httpx.TransportError` subclasses
  (confirmed via `Ollama(...).complete(...)` against a closed port).
- `AgenticPipeline`'s `crewai.LLM` resolves to CrewAI's
  `OpenAICompatibleCompletion`, which talks to Ollama's OpenAI-compatible
  endpoint via the `openai` SDK. Both a refused connection and a client-side
  timeout were confirmed to surface as a plain builtin `ConnectionError` (an
  `OSError` subclass) -- CrewAI catches `openai.APIConnectionError`/
  `APITimeoutError` internally and re-raises this instead.

Keeping this tuple narrow is the point. A bare `except Exception` would also
swallow real programming errors (a `KeyError` in prompt formatting, a
Pydantic validation bug) and misreport them as "upstream unavailable", which
is worse than a 500 because it hides the actual defect behind a
plausible-sounding wrong diagnosis.
"""

import httpx

LLM_UNAVAILABLE_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    ConnectionError,
)
