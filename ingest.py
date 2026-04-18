from pathlib import Path

import chromadb

POLICIES_DIR = Path(__file__).parent / "policies"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "hr_policies"


def extract_line_items(markdown_text: str) -> list[str]:
    items = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def policy_name_from_filename(filename: str) -> str:
    stem = filename.removesuffix(".md")
    return stem.replace("_", " ").title()


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Deleting existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

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
            metadatas.append({"source": policy_file.name, "policy": policy_name})
            ids.append(f"{policy_file.stem}-{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(documents)} line items into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
