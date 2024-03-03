import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    doc = fitz.open("testing.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
