import asyncio
import json
import os
from pathlib import Path
from urllib.parse import quote

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Where the FastAPI backend lives (the compose file points this at the `api` service).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
# Inngest dev server origin. The SDK uses it for both the event and REST APIs.
INNGEST_URL = os.getenv("INNGEST_URL", "http://127.0.0.1:8288").rstrip("/")
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
        api_base_url=INNGEST_URL,
        event_api_base_url=INNGEST_URL,
    )


def save_uploaded_pdf(file) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Path(...).name strips any directory parts, so "../../x.pdf" can't escape uploads/
    file_path = UPLOADS_DIR / Path(file.name).name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


def fetch_documents() -> list[dict]:
    resp = requests.get(f"{API_URL}/documents", timeout=10)
    resp.raise_for_status()
    return resp.json()["documents"]


def delete_document(source: str) -> None:
    resp = requests.delete(f"{API_URL}/documents/{quote(source, safe='')}", timeout=30)
    resp.raise_for_status()


# ------------------------------------------------------------------ Ingest UI
st.title("Upload PDFs to Ingest")

st.session_state.setdefault("uploader_n", 0)
flash_messages = st.session_state.pop("flash", [])
for kind, message in flash_messages:
    (st.success if kind == "ok" else st.error)(message)

# Changing the widget key after each batch clears the uploader. Without that,
# Streamlit would keep returning the same files on every rerun (e.g. after a
# delete) and they would be ingested again.
uploaded_files = st.file_uploader(
    "Choose one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_n']}",
)

if uploaded_files:
    flash: list[tuple[str, str]] = []
    with st.spinner(f"Uploading {len(uploaded_files)} file(s) and triggering ingestion..."):
        for uploaded in uploaded_files:
            try:
                path = save_uploaded_pdf(uploaded)
                asyncio.run(send_rag_ingest_event(path))
                flash.append(("ok", f"Triggered ingestion for: {path.name}"))
            except Exception as exc:
                flash.append(("err", f"Could not ingest {uploaded.name}: {exc}"))
    st.session_state["flash"] = flash
    st.session_state["uploader_n"] += 1
    st.rerun()

# ------------------------------------------------------------- Documents list
st.divider()
st.title("Your documents")
st.caption("Indexing runs in the background. Hit Refresh to see new files appear.")
st.button("🔄 Refresh")

documents: list[dict] = []
try:
    documents = fetch_documents()
except Exception as exc:
    st.warning(f"Could not load the document list: {exc}")

if not documents and not flash_messages:
    st.info("No documents yet. Upload a PDF above to get started.")

for doc in documents:
    name_col, button_col = st.columns([5, 1], vertical_alignment="center")
    name_col.write(f"📄 **{doc['source']}**  ·  {doc['chunks']} chunks")
    if button_col.button("🗑️", key=f"delete_{doc['source']}", help="Delete this document"):
        try:
            delete_document(doc["source"])
        except Exception as exc:
            st.error(f"Could not delete {doc['source']}: {exc}")
        else:
            st.session_state["flash"] = [("ok", f"Deleted {doc['source']}")]
            st.rerun()

# -------------------------------------------------------------------- Chat UI
st.divider()
st.title("Ask a question about your PDFs")

# The conversation lives in the browser session. Each request sends the earlier
# turns along, so the backend can resolve follow-ups like "what about the second one?".
st.session_state.setdefault("messages", [])

scope_col, k_col, clear_col = st.columns([3, 2, 1], vertical_alignment="bottom")
scope = scope_col.selectbox("Search in", ["All documents"] + [d["source"] for d in documents])
top_k = k_col.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
if clear_col.button("Clear chat", disabled=not st.session_state["messages"]):
    st.session_state["messages"] = []
    st.rerun()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))

if question := st.chat_input("Ask a question about your PDFs"):
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"]
    ]
    payload = {"question": question.strip(), "top_k": int(top_k), "history": history}
    if scope != "All documents":
        payload["source"] = scope

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with requests.post(
                f"{API_URL}/query/stream", json=payload, stream=True, timeout=(10, 120)
            ) as resp:
                if resp.status_code == 429:
                    st.warning("Too many requests. Please wait a moment and try again.")
                else:
                    resp.raise_for_status()
                    sources = json.loads(resp.headers.get("X-Sources", "[]"))
                    # Tokens appear as soon as the backend receives them from the LLM;
                    # write_stream returns the complete text once it finishes.
                    answer = st.write_stream(
                        resp.iter_content(chunk_size=None, decode_unicode=True)
                    )
                    if sources:
                        st.caption("Sources: " + "; ".join(sources))
                    # Only successful turns become part of the memory.
                    st.session_state["messages"] += [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer, "sources": sources},
                    ]
        except Exception as exc:
            st.error(f"Something went wrong while answering: {exc}")