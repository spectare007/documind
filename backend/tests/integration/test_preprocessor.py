import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DOCS = Path(__file__).resolve().parents[3] / "data" / "documents"


@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="integration disabled")
def test_parse_real_pdf():
    pdfs = sorted(DOCS.glob("*.pdf"))
    if not pdfs:
        pytest.skip("no local PDFs")
    from app.ingestion.preprocessor import parse_pdf
    parsed = parse_pdf(pdfs[0])
    assert parsed.page_count >= 1
    assert len(parsed.chunks) >= 1
    assert all(c.text.strip() for c in parsed.chunks)
