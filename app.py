from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from queue import Empty, Queue

import pandas as pd
import streamlit as st

from receipt_processor import CATEGORIES, process_pdf


st.set_page_config(page_title="Receipt Intake and Classification", layout="wide")
st.title("Receipt Intake and Classification")
st.caption("Upload synthetic staff claim documents for reading, classification, and review.")

st.subheader("Upload")
uploads = st.file_uploader(
    "Select or drop PDF files. Non-PDF files are rejected with a clear message.",
    accept_multiple_files=True,
    key="receipt_uploads",
)

if uploads:
    signature = hashlib.sha256(
        b"".join(f.name.encode("utf-8") + f.getvalue() for f in uploads)
    ).hexdigest()
    if st.session_state.get("upload_signature") != signature:
        st.session_state["upload_signature"] = signature
        st.session_state.pop("receipt_results", None)
        st.session_state.pop("receipt_details", None)

    if st.button("Process uploaded files", type="primary"):
        progress_area = st.container()
        progress_area.subheader("Progress")
        overall_bar = progress_area.progress(0, text="Overall progress")
        file_rows = []
        for uploaded in uploads:
            with progress_area.container():
                st.markdown(f"**{uploaded.name}**")
                file_rows.append((st.progress(0), st.empty()))

        results = [None] * len(uploads)
        details = {}
        total = len(uploads)
        progress_events: Queue[tuple[int, str]] = Queue()
        for _, status in file_rows:
            status.markdown("Starting")

        def process_one(index: int, filename: str, data: bytes) -> dict:
            def update_step(step: str) -> None:
                progress_events.put((index, step))

            return process_pdf(filename, data, update_step)

        # Keep a few requests in flight together while leaving room under hosted API rate limits.
        worker_count = min(4, total)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_one, index, uploaded.name, uploaded.getvalue()): index
                for index, uploaded in enumerate(uploads)
            }
            pending = set(futures)
            completed = 0
            while pending:
                done, pending = wait(pending, timeout=0.15, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        index, step = progress_events.get_nowait()
                    except Empty:
                        break
                    fraction = {
                        "Extracting digital PDF text": 0.3,
                        "Reading page images with the vision model": 0.3,
                        "Running OCR fallback": 0.55,
                        "Classifying and extracting OCR text": 0.75,
                    }.get(step, 0.1)
                    bar, status = file_rows[index]
                    bar.progress(fraction)
                    status.markdown(step)

                for future in done:
                    index = futures[future]
                    result = future.result()
                    results[index] = result
                    details[str(index)] = result
                    completed += 1
                    bar, status = file_rows[index]
                    bar.progress(1.0)
                    status.markdown(
                        f"**{result['status']}**"
                        + (f" — {result['error']}" if result["error"] else "")
                    )
                    overall_bar.progress(completed / total, text=f"Overall progress: {completed}/{total} files")

        st.session_state["receipt_results"] = results
        st.session_state["receipt_details"] = details

if "receipt_results" in st.session_state:
    results = st.session_state["receipt_results"]
    details = st.session_state.get("receipt_details", {})
    st.subheader("Progress")
    for result in results:
        label = result["status"]
        if result["error"]:
            label += f" — {result['error']}"
        st.write(f"{result['file']}   {label}")

    columns = {
        "file": "File",
        "document_type": "Document type",
        "category": "Category",
        "vendor": "Vendor",
        "date": "Date",
        "amount": "Amount",
        "currency": "Currency",
        "receipt_number": "Receipt number",
        "confidence": "Confidence",
        "source": "Source",
        "status": "Status",
        "reason": "Reason",
    }
    table = pd.DataFrame(results).rename(columns=columns)[list(columns.values())]

    st.subheader("Results table")
    categories = sorted(CATEGORIES)
    selected_categories = st.multiselect("Filter by category", categories, default=categories)
    filtered = table[table["Category"].isin(selected_categories) | table["Category"].eq("")]

    def status_style(row: pd.Series) -> list[str]:
        if row.get("Status") == "Needs review":
            return ["background-color: #fff2cc"] * len(row)
        if row.get("Status") == "Unsupported":
            return ["background-color: #e7e6e6"] * len(row)
        return [""] * len(row)

    styled = filtered.style.apply(status_style, axis=1)
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    st.subheader("Summary by category and currency")
    totals: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for result in results:
        if not result["category"] or not result["amount"] or not result["currency"]:
            continue
        try:
            totals[(result["category"], result["currency"])] += Decimal(result["amount"])
        except InvalidOperation:
            continue
    if totals:
        summary_rows = [
            {"Category": category, "Currency": currency, "Total": f"{amount:.2f}"}
            for (category, currency), amount in sorted(totals.items())
        ]
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    else:
        st.write("No categorized amounts to summarize.")

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        visible_index = filtered.index[selected_rows[0]]
        selected_file = str(table.loc[visible_index, "File"])
        result = next((item for item in results if item["file"] == selected_file), None)
    else:
        result = None

    if result:
        st.subheader(f"Detail: {result['file']}")
        st.write(
            {
                "Document type": result["document_type"],
                "Category": result["category"],
                "Vendor": result["vendor"],
                "Date": result["date"],
                "Amount": result["amount"],
                "Currency": result["currency"],
                "Receipt number": result["receipt_number"],
                "Confidence": result["confidence"],
                "Source": result["source"],
                "Status": result["status"],
                "Reason": result["reason"],
            }
        )
        st.text_area("Extracted or OCR text", result["text"], height=260, disabled=True)
        if result["error"]:
            st.error(result["error"])

    csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="receipt_results.csv",
        mime="text/csv",
    )
