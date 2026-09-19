"""Offline notebook wiring checks, not an evaluation of model extraction quality."""

import ast
import json
from pathlib import Path
import pytest

import fabric_rlm


NOTEBOOK_DIR = Path(__file__).parents[1] / "examples" / "notebooks"
FIXTURE_ROOT = (
    "https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/"
    "2d93675d6d69b45f9eda9b8fc01e178323f8e6cb/examples/"
)
PDF_TEXT = {
    "contract_comparison/sample/input/microFIT-Contract-Version-2-0.pdf":
        "Version: 2.0\nPayment days: 30",
    "contract_comparison/sample/input/microFIT-Contract-Version-3-1-1.pdf":
        "Version: 3.1.1\nPayment days: 45",
    "document_analysis/sample/input/YYJ-2025-Parking-Management-RFP.pdf":
        "Issuer: Example Airport\nDeadline: 2025-06-30",
    "document_redaction/sample/input/PNFS-Employment-Agreement-2025.pdf":
        "Employee: Example Person\nEmail: example@example.invalid",
    "invoice_processing/sample/input/acme-invoice-2025-0042.pdf":
        "Vendor: Acme\nInvoice: 2025-0042\nAmount: 100.00",
    "invoice_processing/sample/input/globaltech-invoice-GT-10587.pdf":
        "Vendor: GlobalTech\nInvoice: GT-10587\nAmount: 250.00",
}
CONTRACT_SCRIPT = """
before, after = [int(record['Payment days']) for record in records]
summary = f'Payment term changed from {before} to {after} days.'
payload = dict(summary=summary, key_differences=[summary])
"""
CONTRACT_PAYLOAD = {
    "summary": "Payment term changed from 30 to 45 days.",
    "key_differences": ["Payment term changed from 30 to 45 days."],
}
CASES = {
    "rlm_pdf_contract_comparison": {
        "task": "Compare the two contract PDFs. Return a concise summary, section_diffs, and key_differences.",
        "inputs": "contracts",
        "files": list(PDF_TEXT)[:2],
        "script": CONTRACT_SCRIPT + "payload['section_diffs'] = [dict(section='Payment', before=before, after=after)]",
        "payload": {
            **CONTRACT_PAYLOAD,
            "section_diffs": [{"section": "Payment", "before": 30, "after": 45}],
        },
    },
    "rlm_minimal_contract_comparison": {
        "task": "Compare the two contract PDFs. Return a concise summary and key_differences as a list of short strings.",
        "inputs": "contracts",
        "files": list(PDF_TEXT)[:2],
        "script": CONTRACT_SCRIPT,
        "payload": CONTRACT_PAYLOAD,
    },
    "rlm_pdf_document_analysis": {
        "task": "Analyze the RFP PDF. Return a concise summary, key_dates, key_entities, and page_counts.",
        "inputs": "document",
        "files": [list(PDF_TEXT)[2]],
        "script": """
record = records[0]
payload = dict(summary=f"RFP issued by {record['Issuer']}.",
               key_dates=[dict(event='Deadline', date=record['Deadline'])],
               key_entities=[record['Issuer']], page_counts=page_counts)
""",
        "payload": {
            "summary": "RFP issued by Example Airport.",
            "key_dates": [{"event": "Deadline", "date": "2025-06-30"}],
            "key_entities": ["Example Airport"],
            "page_counts": [1],
        },
    },
    "rlm_pdf_document_redaction": {
        "task": "Find PII redaction targets in the PDF. Return total_redactions, targets, and a concise summary.",
        "inputs": "document",
        "files": [list(PDF_TEXT)[3]],
        "script": """
targets = [dict(category=key, text=value, page=1) for key, value in records[0].items()]
payload = dict(total_redactions=len(targets), targets=targets,
               summary=f'{len(targets)} PII targets on page 1.')
""",
        "payload": {
            "total_redactions": 2,
            "targets": [
                {"category": "Employee", "text": "Example Person", "page": 1},
                {"category": "Email", "text": "example@example.invalid", "page": 1},
            ],
            "summary": "2 PII targets on page 1.",
        },
    },
    "rlm_pdf_invoice_processing": {
        "task": "Extract invoice data from the PDFs. Return invoices, total_amount, and a concise summary.",
        "inputs": "invoices",
        "files": list(PDF_TEXT)[4:],
        "script": """
rows = [dict(vendor=r['Vendor'], invoice_number=r['Invoice'], amount=float(r['Amount']))
        for r in records]
payload = dict(invoices=rows, total_amount=sum(row['amount'] for row in rows),
               summary=f'{len(rows)} invoices extracted.')
""",
        "payload": {
            "invoices": [
                {"vendor": "Acme", "invoice_number": "2025-0042", "amount": 100.0},
                {"vendor": "GlobalTech", "invoice_number": "GT-10587", "amount": 250.0},
            ],
            "total_amount": 350.0,
            "summary": "2 invoices extracted.",
        },
    },
}


@pytest.fixture(params=CASES)
def notebook(request, tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    name = request.param
    case = CASES[name]
    notebook = json.loads((NOTEBOOK_DIR / f"{name}.ipynb").read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            assert source == f"%pip install -q fabric-rlm[pdf]=={fabric_rlm.__version__}"
            continue
        ast.parse(source, filename=f"{name}:cell{index}")
        source = source.replace("/tmp/", tmp_path.as_posix() + "/")
        cells.append(compile(source, f"{name}:cell{index}", "exec"))

    downloads = []

    def urlretrieve(url, filename):
        assert url.startswith(FIXTURE_ROOT)
        relative = url.removeprefix(FIXTURE_ROOT)
        assert Path(filename).parent == tmp_path
        with fitz.open() as pdf:
            pdf.new_page().insert_text((72, 72), PDF_TEXT[relative])
            pdf.save(filename)
        downloads.append(relative)
        return filename, None

    monkeypatch.setattr("urllib.request.urlretrieve", urlretrieve)
    namespace = {}
    for cell in cells[:-1]:
        exec(cell, namespace)
    assert downloads == case["files"]
    return case, cells[-1], namespace


@pytest.mark.parametrize("submit", [True, False], ids=["submitted", "not-submitted"])
def test_pdf_notebook_execution(notebook, monkeypatch, submit):
    case, task_cell, namespace = notebook
    messages_seen = []
    inputs = case["inputs"]
    files = "[document]" if inputs == "document" else inputs
    script = f"""
import fitz
records = []
page_counts = []
for file in {files}:
    assert file.exists()
    with fitz.open(file.path) as pdf:
        page_counts.append(pdf.page_count)
        text = ''.join(page.get_text() for page in pdf)
        records.append(dict(line.split(': ', 1) for line in text.splitlines()))
{case['script']}
print(payload)
"""
    script += "SUBMIT(**payload)" if submit else "print('Deliberately not submitting')"

    def scripted_lm(*, messages):
        messages_seen.append([dict(message) for message in messages])
        return f"```python\n{script}\n```"

    def fabric_lm(model, **kwargs):
        assert model == "gpt-5.1"
        assert kwargs == {"reasoning_effort": "low"}
        return scripted_lm

    monkeypatch.setattr(fabric_rlm, "FabricLM", fabric_lm)
    if submit:
        exec(task_cell, namespace)
        assert namespace["result"].payload == case["payload"]
        assert set(namespace["result"].payload) == set(case["payload"])
    else:
        with pytest.raises(AssertionError) as error:
            exec(task_cell, namespace)
        result = namespace["result"]
        assert not result.submitted
        assert result.failure_reason
        assert str(error.value) == result.failure_reason

    assert isinstance(namespace["rlm"], fabric_rlm.RLM)
    prompt = messages_seen[0][0]["content"]
    task_section = prompt.split("## Task", 1)[1].split("## Inputs available", 1)[0]
    assert case["task"] in task_section
    assert "pdf_document_analysis" in prompt
    assert "Open every PDF with PyMuPDF" in prompt
    assert set(namespace["rlm"]._inline_outputs) == set(case["payload"])
    assert all(not turn.error for turn in namespace["result"].trajectory)
