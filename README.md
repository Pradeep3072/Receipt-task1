# Receipt Intake and Classification

## Run

1. Install Python 3.11 or newer.
2. Install the Python dependencies with `python -m pip install -r requirements.txt`.
3. Install Tesseract OCR and make its executable available on `PATH`, or set `TESSERACT_CMD` to its full path. pypdfium2 renders scanned PDF pages for the vision service and OCR fallback.
4. Copy `.env.example` to `.env` and put the Groq API key in the `GROQ_API_KEY` entry. `.env` is ignored by Git. Keep the key out of source control and chat.
5. Run `streamlit run app.py`.

The app sends receipt text or scanned page images to the configured Groq API. Use synthetic documents only. A request can fail if the service key is missing, the account reaches a rate limit, or the service is unavailable; each file is handled independently and shows an error.

## Design

Streamlit accepts multiple files and the processor handles each PDF independently. It extracts a text layer when present and asks the hosted Qwen vision model to classify and extract the required fields. Image-only PDFs are rendered as page images and sent to the vision model for transcription first; if that fails, Tesseract OCR is used as a fallback. The extracted text is then classified and normalized. The app shows uncertainty as Needs review, marks non-expense documents Unsupported, and keeps currency totals separate. The processor does not use filenames to determine document content.

## AI tools

- Groq-hosted Qwen 3.8 27B: vision transcription, document-type and expense-category classification, field extraction, confidence estimates, and short category reasons.
- No external AI coding service is required by the runtime.

## Evaluation and fixtures

The assessment PDF describes 12 synthetic examples and includes their expected metadata. The original sample PDFs and evaluator are not present in this workspace. Run `python generate_synthetic_samples.py` to recreate the locally authored PDF fixtures from `expected_results.csv`, then run `python evaluate.py` to compare document type, category, vendor, date, and amount and list every mismatch. Results on these replacement fixtures are not results on the original assessment documents. The locally authored replacement fixtures were evaluated with the configured Groq service: document type 12/12, category 11/12, vendor 12/12, date 12/12, and amount 12/12. R10 category was predicted as Other instead of Travel because its amount is obscured. These results only measure the replacement fixtures.

## Known weaknesses

- The model's confidence values are estimates returned by the model; they are not calibrated probabilities.
- The vision-to-OCR fallback requires the Tesseract executable to be installed and configured.
- Hosted inference depends on the provider's availability, API key, request limits, and current model access.
