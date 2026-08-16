import threading

import app.ingestion.preprocessor as preprocessor


def test_get_converter_initializes_once_under_concurrent_access(monkeypatch):
    """`_get_converter()` must construct the shared DocumentConverter exactly once even
    when many threads race to initialise it (each ingest job runs on its own thread)."""

    init_calls: list[int] = []
    init_calls_lock = threading.Lock()

    class _CountingConverter:
        def __init__(self):
            with init_calls_lock:
                init_calls.append(1)

    monkeypatch.setattr(preprocessor, "DocumentConverter", _CountingConverter)
    monkeypatch.setattr(preprocessor, "_converter", None)

    n_threads = 16
    results: list[object] = [None] * n_threads
    barrier = threading.Barrier(n_threads)

    def worker(index: int) -> None:
        barrier.wait()  # maximize the chance all threads hit the check-then-act race together
        results[index] = preprocessor._get_converter()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(init_calls) == 1, "DocumentConverter() must be constructed exactly once"
    assert all(r is results[0] for r in results), "all threads must receive the same instance"
