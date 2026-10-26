"""Streamlit UI for AdaptiveRAG.

A query box, the answer with citations, and a side panel showing exactly
how the pipeline arrived at it (route, reason, escalation, confidence,
latency, tokens) -- the same fields logged to logs/queries.jsonl, just
rendered for a human instead of appended to a file.
"""

import streamlit as st

from src.pipeline import ask

st.set_page_config(page_title="AdaptiveRAG", layout="wide")
st.title("AdaptiveRAG")

use_rerank = st.checkbox("Use reranking (hybrid route only)", value=True)
query = st.text_input("Ask a question")

if query:
    result = ask(query, use_rerank=use_rerank)

    main_col, side_col = st.columns([3, 1])

    with main_col:
        st.subheader("Answer")
        st.write(result.answer.text)

        if result.answer.citations:
            st.subheader("Citations")
            for citation in result.answer.citations:
                st.write(f"[{citation['n']}] {citation['source']}, page {citation['page']}")

        if result.hits:
            with st.expander(f"Retrieved chunks ({len(result.hits)})"):
                for hit in result.hits:
                    st.markdown(
                        f"**chunk {hit.chunk_id}** "
                        f"(source={hit.chunk['source']}, page={hit.chunk['page']}, score={hit.score:.3f})"
                    )
                    st.text(hit.chunk["text"])

    with side_col:
        st.subheader("Call details")
        st.write(f"**Route:** {result.route}")
        st.write(f"**Reason:** {result.reason}")
        st.write(f"**Escalated:** {result.escalated}")
        confidence_text = f"{result.confidence:.3f}" if result.confidence is not None else "n/a"
        st.write(f"**Confidence:** {confidence_text}")
        st.write(f"**Latency:** {result.latency_s:.3f}s")
        st.write(f"**Tokens in/out:** {result.tokens_in} / {result.tokens_out}")
