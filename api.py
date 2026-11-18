from fastapi import FastAPI, File, UploadFile

from receipt_processor import process_pdf

app = FastAPI(title="Receipt Processing API")


@app.post("/api/receipts")
def process_receipt(file: UploadFile = File(...)):
    pdf_bytes = file.file.read()
    return process_pdf(file.filename or "uploaded.pdf", pdf_bytes)