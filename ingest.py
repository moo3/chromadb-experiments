"""
Ingestion script: reads the markdown policies and loads them into ChromaDB.

This is the "write" side of our vector search pipeline. It runs once
(and again whenever the policies change). The "read" side is `app.py`.

Pipeline overview:
    policies/*.md  ->  split into bullets  ->  embed each bullet  ->  store in ChromaDB
                                               (done by ChromaDB)

We never call an embedding model ourselves here. ChromaDB's default
embedding function (all-MiniLM-L6-v2 via sentence-transformers) runs
automatically when we call `collection.add(...)` with raw text.
"""

from pathlib import Path

import chromadb

# Paths are resolved relative to this file so the script works no matter
# which directory you run it from.
POLICIES_DIR = Path(__file__).parent / "policies"

# ChromaDB stores its data as plain files on disk. We use a local folder
# so there's no server to run. `.gitignore` excludes this directory.
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# A ChromaDB "collection" is like a table in a SQL database — a named
# container for documents + their embeddings. One collection is enough
# for this experiment; real projects often have several (one per
# document type, per tenant, etc.).
COLLECTION_NAME = "hr_policies"


def extract_line_items(markdown_text: str) -> list[str]:
    """
    Pull each bullet line out of a markdown file.

    We treat every bullet ("- ...") as an independent document. This is
    the simplest possible *chunking* strategy: each chunk is one policy
    statement. Smaller chunks = more precise matches; larger chunks =
    more context per match. For a real project you would use a proper
    chunker (by token count, by section, etc.).
    """
    items = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        # Only lines starting with "- " are treated as bullets.
        # Headings (#) and blank lines are skipped.
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def policy_name_from_filename(filename: str) -> str:
    """Turn 'holidays_and_leaves.md' into 'Holidays And Leaves' for display."""
    stem = filename.removesuffix(".md")
    return stem.replace("_", " ").title()


def main() -> None:
    # PersistentClient writes to disk, so our data survives between runs.
    # chromadb.Client() (without "Persistent") keeps everything in memory
    # and loses it when the script exits — handy for tests, not for this.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Make the script idempotent: dropping and recreating the collection
    # means you can re-run ingest.py any time the policies change without
    # ending up with duplicate or stale vectors.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Deleting existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)

    # Creating the collection doesn't pick an embedding model explicitly —
    # ChromaDB falls back to its default (all-MiniLM-L6-v2). The first
    # time you call .add() it will download the model (~80 MB) into a
    # local cache, then reuse it forever.
    collection = client.create_collection(name=COLLECTION_NAME)

    # ChromaDB's .add() takes three parallel lists:
    #   documents  - the raw text (ChromaDB embeds these for us)
    #   metadatas  - a dict per document, used for filtering at query time
    #                and for showing the source in the UI
    #   ids        - a unique string id per document
    # We build them up in lock-step as we walk the files.
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # sorted() so the order is deterministic across runs — makes debugging
    # and diffs easier.
    policy_files = sorted(POLICIES_DIR.glob("*.md"))
    if not policy_files:
        raise SystemExit(f"No policy files found in {POLICIES_DIR}")

    for policy_file in policy_files:
        text = policy_file.read_text()
        items = extract_line_items(text)
        policy_name = policy_name_from_filename(policy_file.name)
        print(f"  {policy_file.name}: {len(items)} line items")

        for i, item in enumerate(items):
            documents.append(item)
            # Metadata is what you use to filter later — e.g. "only search
            # dress_code.md" via collection.query(where={"source": "dress_code.md"}).
            # It's also what we display next to each result in the UI.
            metadatas.append({"source": policy_file.name, "policy": policy_name})
            # IDs must be unique across the collection. Combining filename
            # stem with the line index guarantees uniqueness and gives you
            # something human-readable like "dress_code-3".
            ids.append(f"{policy_file.stem}-{i}")

    # One batched .add() is much faster than calling it per-document —
    # the embedding model processes the whole batch on the GPU/CPU at once.
    # For large datasets you'd batch in chunks of ~100 to balance memory
    # and throughput.
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(documents)} line items into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
