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
    signature_hash = hashlib.sha256()
    for uploaded in uploads:
        filename_bytes = uploaded.name.encode("utf-8")
        file_view = uploaded.getbuffer()
        signature_hash.update(len(filename_bytes).to_bytes(8, "big"))
        signature_hash.update(filename_bytes)
        signature_hash.update(len(file_view).to_bytes(8, "big"))
        signature_hash.update(file_view)
    signature = signature_hash.hexdigest()
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
            futures = {}
            pending = set()
            next_index = 0
            completed = 0
            while next_index < total and len(pending) < worker_count:
                uploaded = uploads[next_index]
                future = executor.submit(
                    process_one, next_index, uploaded.name, uploaded.getvalue()
                )
                futures[future] = next_index
                pending.add(future)
                next_index += 1

            while pending:
                done, pending = wait(pending, timeout=0.15, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        index, step = progress_events.get_nowait()
                    except Empty:
                        break
                    fraction = {
                        "Extracting digital PDF text": 0.3,
                        "Running OCR": 0.3,
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
                    completed += 1
                    bar, status = file_rows[index]
                    bar.progress(1.0)
                    status.markdown(
                        f"**{result['status']}**"
                        + (f" — {result['error']}" if result["error"] else "")
                    )
                    overall_bar.progress(completed / total, text=f"Overall progress: {completed}/{total} files")

                while next_index < total and len(pending) < worker_count:
                    uploaded = uploads[next_index]
                    future = executor.submit(
                        process_one, next_index, uploaded.name, uploaded.getvalue()
                    )
                    futures[future] = next_index
                    pending.add(future)
                    next_index += 1

        st.session_state["receipt_results"] = results
else:
    st.session_state.pop("upload_signature", None)
    st.session_state.pop("receipt_results", None)
    st.session_state.pop("receipt_details", None)

if "receipt_results" in st.session_state:
    results = st.session_state["receipt_results"]
    st.subheader("Progress")
    for result in results:
        label = result["status"]
        if result["error"]:
            label += f" — {result['error']}"
        st.write(f"{result['file']}   {label}")

    table_data = pd.DataFrame(results)
    table_data["amount_currency"] = [
        f"{result['currency']} {result['amount']}".strip() for result in results
    ]

    columns = {
        "file": "File",
        "document_type": "Document type",
        "category": "Category",
        "vendor": "Vendor",
        "date": "Date",
        "amount_currency": "Currency / Amount",
        "source": "Source",
        "status": "Status",
    }
    table = table_data.rename(columns=columns)[list(columns.values())]

    st.subheader("Results table")
    categories = sorted(CATEGORIES)
    selected_categories = st.multiselect(
        "Filter by category",
        categories,
        default=[],
        help="Select one or more categories to filter. Leave empty to show all files.",
    )
    filtered = table[table["Category"].isin(selected_categories)] if selected_categories else table

    status_colors = {
        "Done": "background-color: #d9ead3; color: #000000; font-weight: bold",
        "Needs review": "background-color: #fff2cc; color: #000000; font-weight: bold",
        "Unsupported": "background-color: #e7e6e6; color: #000000; font-weight: bold",
    }
    styled = filtered.style.map(lambda value: status_colors.get(value, ""), subset=["Status"])
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    st.subheader("Summary by category")
    summary_by_category: dict[str, dict] = {}
    for result in results:
        category = result["category"]
        if not category:
            continue
        summary = summary_by_category.setdefault(category, {"files": [], "totals": defaultdict(Decimal)})
        summary["files"].append(result["file"])
        if not result["amount"] or not result["currency"]:
            continue
        try:
            summary["totals"][result["currency"]] += Decimal(result["amount"])
        except InvalidOperation:
            continue
    if summary_by_category:
        summary_rows = []
        for category, summary in sorted(summary_by_category.items()):
            amounts = "; ".join(
                f"{currency} {amount:.2f}"
                for currency, amount in sorted(summary["totals"].items())
            )
            summary_rows.append(
                {
                    "Category": category,
                    "Files": ", ".join(summary["files"]),
                    "Currency / total": amounts or "No complete amounts",
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    else:
        st.write("No categorized amounts to summarize.")

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        visible_index = filtered.index[selected_rows[0]]
        result = results[visible_index]
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

    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="receipt_results.csv",
        mime="text/csv",
    )
