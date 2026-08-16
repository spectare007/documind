from app.ingestion.types import RawChunk


def _chunk(**kw):
    base = dict(text="Total amount due: EUR 1,200", section_path=[], pages=[1], is_table=False)
    base.update(kw)
    return RawChunk(**base)


def test_header_with_sections():
    from app.ingestion.contextualizer import contextualize
    out = contextualize(_chunk(section_path=["Invoice Details", "Line Items"]), "Invoice June 2026")
    assert out.startswith("[Invoice June 2026 > Invoice Details > Line Items]\n\n")
    assert out.endswith("Total amount due: EUR 1,200")


def test_header_without_sections_uses_title_only():
    from app.ingestion.contextualizer import contextualize
    out = contextualize(_chunk(), "Payslip")
    assert out.startswith("[Payslip]\n\n")


def test_table_chunk_flagged():
    from app.ingestion.contextualizer import contextualize
    out = contextualize(_chunk(is_table=True, section_path=["Earnings"]), "Payslip")
    assert out.startswith("[Payslip > Earnings | table]\n\n")
