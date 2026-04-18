"""
Streamlit UI: the "read" side of the pipeline.

Flow:
    user types question  ->  embed the question  ->  ask ChromaDB for nearest N
                             (done by ChromaDB)      vectors  ->  display them

Again, no LLM is involved. The model embedding the query is the same
one that embedded the stored documents — that's critical. If you query
with a different embedding model, the vectors live in a different space
and nothing useful comes back.

Streamlit re-runs this entire script top-to-bottom on every user
interaction. That's its core mental model: your script is your UI.
Anything expensive (loading models, opening DB connections) must be
cached with @st.cache_resource or @st.cache_data, otherwise it would
re-run on every keystroke.
"""

from pathlib import Path

import chromadb
import streamlit as st

# Must match the path ingest.py writes to, otherwise we'd open an empty DB.
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "hr_policies"

# How many results to return per query. 5 is a common default — enough
# variety to be useful, few enough to fit on screen.
TOP_K = 5

# Pre-written questions shown as clickable buttons so the user has
# something to try without thinking. Each one is crafted to match a
# different policy file, demonstrating that semantic search works
# across all of them.
EXAMPLE_QUERIES = [
    "Can I take my laptop on vacation?",
    "Can I wear jeans on a Friday?",
    "How many sick days do I get?",
    "Do I need VPN when working from home?",
    "What is the policy on parental leave?",
]


@st.cache_resource
def get_collection():
    """
    Open the ChromaDB collection once and reuse it for the lifetime of
    the app.

    @st.cache_resource is Streamlit's way of saying "this return value
    is shared across all users and all reruns of this script" — perfect
    for heavy objects like DB clients and ML models. Without it, we'd
    re-open the DB (and re-load the embedding model) on every keystroke.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def run_query(query: str) -> None:
    """Run a semantic search and render the top results."""
    collection = get_collection()

    # collection.query() does four things under the hood:
    #   1. Embeds the query text using the same model used at ingest.
    #   2. Looks up the TOP_K nearest vectors in the index.
    #   3. Fetches the original documents and metadata for those vectors.
    #   4. Returns distances so we can show how confident each match is.
    #
    # query_texts is a LIST because you can batch multiple queries in
    # one call. We only send one, so everything we care about is at [0].
    results = collection.query(query_texts=[query], n_results=TOP_K)

    docs = results["documents"][0]       # the matched bullet text
    metas = results["metadatas"][0]      # {"source": ..., "policy": ...}
    distances = results["distances"][0]  # lower = more similar

    st.subheader(f"Top {len(docs)} results")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        # st.container(border=True) draws a subtle card around each result.
        with st.container(border=True):
            st.markdown(f"**{i}. {meta['policy']}** — `{meta['source']}`")
            st.write(doc)
            # Distance is ChromaDB's default L2 (Euclidean) distance by
            # default. 0 would be a perfect match; values above ~1.5 mean
            # the model isn't finding much semantic overlap. Showing it
            # teaches the user to read retrieval quality at a glance.
            st.caption(f"distance: {dist:.4f}")


# --------------------------- PAGE LAYOUT ---------------------------
# Everything below runs top-to-bottom on every interaction. Streamlit
# diffs the output against the previous render and only updates what
# changed, so this is cheaper than it looks.

st.title("HR Policy Search")
st.caption("ChromaDB + all-MiniLM-L6-v2 embeddings")

# st.session_state persists across reruns within a single browser
# session. We use it to let a button click populate the text input.
if "query" not in st.session_state:
    st.session_state.query = ""

# Render the example queries as a row of buttons. Clicking one writes
# that query into session_state, and the next rerun (triggered by the
# click) picks it up as the default value of the text input.
st.markdown("**Example queries:**")
cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    if col.button(example, use_container_width=True):
        st.session_state.query = example

# The text box. `value=` seeds it with whatever is in session_state,
# which is how example-button clicks end up inside it.
query = st.text_input("Ask a question about HR policies:", value=st.session_state.query)

# Only render results when the user has typed (or clicked) something.
if query:
    run_query(query)
