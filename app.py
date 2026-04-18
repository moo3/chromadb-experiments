from pathlib import Path

import chromadb
import streamlit as st

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "hr_policies"
TOP_K = 5

EXAMPLE_QUERIES = [
    "Can I take my laptop on vacation?",
    "Can I wear jeans on a Friday?",
    "How many sick days do I get?",
    "Do I need VPN when working from home?",
    "What is the policy on parental leave?",
]


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def run_query(query: str) -> None:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=TOP_K)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    st.subheader(f"Top {len(docs)} results")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {meta['policy']}** — `{meta['source']}`")
            st.write(doc)
            st.caption(f"distance: {dist:.4f}")


st.title("HR Policy Search")
st.caption("ChromaDB + all-MiniLM-L6-v2 embeddings")

if "query" not in st.session_state:
    st.session_state.query = ""

st.markdown("**Example queries:**")
cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    if col.button(example, use_container_width=True):
        st.session_state.query = example

query = st.text_input("Ask a question about HR policies:", value=st.session_state.query)

if query:
    run_query(query)
