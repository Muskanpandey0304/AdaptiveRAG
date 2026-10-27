"""Streamlit UI for AdaptiveRAG.

A query box, the answer with citations, and a side panel showing exactly
how the pipeline arrived at it (route, reason, escalation, confidence,
latency, tokens) -- the same fields logged to logs/queries.jsonl, just
rendered for a human instead of appended to a file.

Streamlit reruns this whole script on every widget change. The inputs live in
a form (nothing runs until "Ask" is pressed) and the finished Result is kept in
st.session_state, so toggling a checkbox or reopening an expander never calls
the LLM again (20-40 s on a CPU).
"""

from pathlib import Path

import httpx
import ollama
import streamlit as st

from src.ingest import MD_DIR
from src.pipeline import ask

MAX_IMAGES_PER_CHUNK = 2

st.set_page_config(page_title="AdaptiveRAG", layout="wide")
st.title("AdaptiveRAG")

with st.form("ask_form"):
    query = st.text_input("Ask a question")
    use_rerank = st.checkbox("Use reranking (hybrid route only)", value=True)
    submitted = st.form_submit_button("Ask")

if submitted and query.strip():
    key = (query.strip(), use_rerank)
    if st.session_state.get("result_key") != key:
        try:
            with st.spinner("Retrieving and answering (a local model can take 20-40 s)..."):
                st.session_state["result"] = ask(query.strip(), use_rerank=use_rerank)
            st.session_state["result_key"] = key
        except FileNotFoundError:
            st.session_state.pop("result", None)
            st.error("The search index is missing. Run `python -m src.ingest` then `python -m src.indexer`.")
        except (ConnectionError, httpx.HTTPError):
            st.session_state.pop("result", None)
            st.error("Could not reach Ollama. Start it (`ollama serve`) and try again.")
        except ollama.ResponseError as error:
            st.session_state.pop("result", None)
            st.error(f"Ollama rejected the request: {error}")

result = st.session_state.get("result")
if result is not None:
    main_col, side_col = st.columns([3, 1])

    with main_col:
        st.subheader("Answer")
        st.write(result.answer.text)

        if result.answer.citations:
            st.subheader("Citations")
            titles = {hit.chunk["source"]: hit.chunk.get("title") for hit in result.hits}
            for citation in result.answer.citations:
                title = titles.get(citation["source"])
                label = f"{title} ({citation['source']})" if title else citation["source"]
                st.write(f"[{citation['n']}] {label}, page {citation['page']}")

        if result.hits:
            with st.expander(f"Retrieved chunks ({len(result.hits)})"):
                for hit in result.hits:
                    st.markdown(
                        f"**chunk {hit.chunk_id}** "
                        f"(source={hit.chunk['source']}, page={hit.chunk['page']}, score={hit.score:.3f})"
                    )
                    st.text(hit.chunk["text"])
                    # Figures on the cited page, saved as PNGs by src/convert.py.
                    for image in (hit.chunk.get("images") or [])[:MAX_IMAGES_PER_CHUNK]:
                        if (Path(MD_DIR) / image).exists():
                            st.image(str(Path(MD_DIR) / image))

    with side_col:
        st.subheader("Call details")
        st.write(f"**Route:** {result.route}")
        st.write(f"**Reason:** {result.reason}")
        st.write(f"**Escalated:** {result.escalated}")
        confidence_text = f"{result.confidence:.3f}" if result.confidence is not None else "n/a"
        st.write(f"**Confidence:** {confidence_text}")
        st.write(f"**Latency:** {result.latency_s:.3f}s")
        st.write(f"**Tokens in/out:** {result.tokens_in} / {result.tokens_out}")
