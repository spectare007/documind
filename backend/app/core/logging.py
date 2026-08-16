import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:  # idempotent under uvicorn reload
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] [cid=%(correlation_id)s] %(message)s"
    ))
    handler.addFilter(_CorrelationFilter())
    root.addHandler(handler)
    root.setLevel(level)


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            from app.core.correlation import get_correlation_id
            record.correlation_id = get_correlation_id() or "-"
        return True
