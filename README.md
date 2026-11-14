# Receipt Intake and Classification

A Streamlit app that accepts receipt and expense PDFs, extracts their details, classifies them, and provides a review table and CSV download. It uses Groq-hosted `qwen/qwen3.8-27b` for vision transcription, classification, and field extraction. Set `GROQ_MODEL` in `.env` to change the model.

## What the app does

- Accepts multiple PDF files in one upload.
- Shows a progress bar and current step for each file. Up to four processing tasks can run at once.
- Reads selectable text from digital PDFs with `pypdf`.
- For scanned PDFs, renders pages with `pypdfium2`, asks the Groq vision model to transcribe them, and uses Tesseract OCR if vision transcription fails. For a mixed PDF, it OCRs only pages without selectable text and joins page text in order.
- Marks a PDF as `Digital` when every page has selectable text; if any page requires OCR, marks the PDF as `OCR`.
- Treats all pages in a PDF as one document. For the amount, the prompt asks for the explicitly stated overall document total and does not ask the model to calculate it from subtotals or line items.
- Extracts document type, category, vendor, date, amount, currency, receipt number, confidence, and a short reason.
- Shows the main fields in a results table. Amount and currency share a column. Confidence, reason, receipt number, and extracted text are available in the selected row's detail panel.
- Leaves the category filter empty by default, so all files show until a category is selected.
- Shows a summary grouped by category, listing the files and totals split by currency. Different currencies are never added together.
- Downloads a CSV with the same visible rows and columns as the results table, including any active category filter.

### Status colors

- **Done:** green
- **Needs review:** amber
- **Unsupported:** grey
- **Failed:** no special color

## How processing works

1. `app.py` receives uploaded PDF names and bytes from Streamlit.
2. `app.py` calls `process_pdf()` in `receipt_processor.py` for each file and displays progress updates.
3. `receipt_processor.py` uses `pypdf` to extract selectable text from every page.
4. If any page has no selectable text, `pypdfium2` renders those pages as images. The Groq vision model transcribes the images first; Tesseract is the OCR fallback. Digital and OCR text is then combined in page order.
5. The extracted text is sent to Groq for classification and field extraction. The processor validates and normalizes the returned fields, assigns a status, and returns a result dictionary to `app.py`.
6. `app.py` stores the results for the current Streamlit session and uses them to render the table, summary, detail panel, and CSV download.

PDFium rendering is protected by a lock because PDFium must not be called simultaneously from multiple threads. Groq requests are serialized and spaced to reduce rate-limit errors. Each file is handled separately, so one file error does not stop the remaining files.

## Requirements

- Python 3.11 or newer
- A Groq API key with access to the configured model
- Tesseract OCR is optional, but needed when the vision transcription fallback is used

The app looks for Tesseract on `PATH` and at the standard Windows installation paths. For a different installation path, set `TESSERACT_CMD` in `.env`.

## Data handling

Use synthetic documents only. Digital PDF text and scanned page images are sent to the configured Groq service for inference. If Tesseract is used, its OCR text is also sent to Groq for classification and field extraction.

## Setup and run

From the project folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set `GROQ_API_KEY` to your key. The `.env` file is ignored by Git; do not commit it or paste the key into chat.

Start the app:

```powershell
python -m streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Project files

- `app.py` — Streamlit upload interface, per-file progress, result table, category filter, summary, detail panel, and CSV download.
- `receipt_processor.py` — PDF text extraction, page rendering, vision/OCR transcription, Groq requests, field validation, confidence handling, and statuses.
- `evaluate.py` — Compares processor output with `expected_results.csv` and reports accuracy and mismatches.
- `generate_synthetic_samples.py` — Regenerates the locally created PDF samples.
- `expected_results.csv` — Expected values for those locally created samples.
- `sample_docs/` — Synthetic PDFs used for local evaluation.
- `.env.example` — Example Groq configuration without an API key.
- `requirements.txt` — Python package dependencies.

## Evaluation

The assessment PDF was provided, but its original `sample_docs/`, `expected_results.csv`, and `evaluate.py` were not. The local samples and evaluator in this repository were created as substitutes based on the brief. Their results do not represent performance on the original assessment PDFs.

Regenerate the local samples and run the evaluator with:

```powershell
python generate_synthetic_samples.py
python evaluate.py
```

The evaluator also compares confidence with correctness for each field. It reports mean confidence, observed accuracy, Expected Calibration Error (ECE), and Brier score. ECE uses five equal-width confidence bins; Brier is the mean squared difference between confidence and the correct/incorrect outcome. Lower ECE and Brier scores indicate a closer fit, but a small sample cannot establish reliable calibration. Overall document calibration only includes fixtures with all six expected fields populated.

Latest run on the replacement samples: document type **12/12**, category **11/12**, vendor **12/12**, date **12/12**, and amount **12/12**. The one category mismatch was R10: expected Travel, predicted Other.

| Confidence measure | Samples | Mean confidence | Accuracy | ECE | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Document type | 12 | 0.989 | 1.000 | 0.011 | 0.000 |
| Category | 11 | 0.949 | 0.909 | 0.040 | 0.076 |
| Vendor | 11 | 0.968 | 1.000 | 0.032 | 0.001 |
| Date | 11 | 0.990 | 1.000 | 0.010 | 0.000 |
| Amount | 10 | 0.989 | 1.000 | 0.011 | 0.000 |
| Currency | 11 | 0.990 | 1.000 | 0.010 | 0.000 |
| Overall document | 10 | 0.945 | 1.000 | 0.055 | 0.003 |

The dataset is too small and synthetic to establish reliable calibration. Overall confidence is evaluated only for fixtures with all six expected fields populated.

## Known weaknesses

1. **No complete offline fallback.** Tesseract can transcribe a scan if vision transcription fails, but classification and field extraction still require Groq. If Groq is unavailable or the account remains rate-limited, processing can fail.
2. **Limited concurrency.** Up to four worker tasks are started, but PDFium rendering and Groq requests are serialized for thread safety and rate limiting. Larger batches can take longer and are not fully processed in parallel.
3. **Confidence is not calibrated.** Confidence values are estimates returned by the model, not statistically calibrated probabilities.
