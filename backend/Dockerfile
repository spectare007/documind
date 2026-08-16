FROM python:3.12-slim

# libgl/libglib needed by docling's layout models (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
ENV PATH="/app/.venv/bin:$PATH"

# Docling's model path calls torch.compile / TorchInductor, which needs a C++
# compiler (g++) to JIT-compile kernels. This slim base image has no C++
# toolchain, so compilation fails with InvalidCxxCompiler. We run CPU-only
# here anyway, so torch.compile buys no speedup -- disable TorchDynamo so
# Docling/torch fall back to eager execution instead of failing ingestion.
ENV TORCHDYNAMO_DISABLE=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
