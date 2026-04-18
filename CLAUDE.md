# Chroma Embeddings Experiment

## What to Build

A learning project to experiment with ChromaDB vector embeddings. Three parts:

1. **Sample HR policy documents** — three policies with ~10 line items each
2. **ChromaDB ingestion script** — loads the policies into a persistent local ChromaDB
3. **Streamlit query UI** — lets you ask natural language questions and see matching policy results

## Requirements

### 1. Sample HR Policies (`policies/`)

Create three markdown files in a `policies/` directory:

**`holidays_and_leaves.md`** — ~10 line items covering:
- Annual PTO allowance, sick leave, parental leave, bereavement, jury duty
- Holiday calendar, carryover rules, approval process, minimum notice, blackout periods

**`dress_code.md`** — ~10 line items covering:
- General professional attire expectations, casual Fridays, client meeting dress code
- Prohibited items, footwear, accessories, remote work attire, seasonal guidelines
- Department-specific exceptions, personal grooming

**`technology_use.md`** — ~10 line items covering:
- Company device policies, personal device usage (BYOD), software installation rules
- Data security on devices, traveling with company equipment, VPN requirements
- Acceptable use of internet, social media during work, password policies, incident reporting

Keep policies realistic but concise. Each line item should be 1-2 sentences — enough to give the embeddings meaningful content to match against.

### 2. ChromaDB Ingestion (`ingest.py`)

- Use `chromadb` with persistent local storage (store in `./chroma_db/` directory)
- Use ChromaDB's default embedding function (all-MiniLM-L6-v2 via sentence-transformers) — do NOT use OpenAI or any external API
- Read each policy markdown file from `policies/`
- Split each policy into individual line items (each bullet/line becomes its own document)
- Store each line item as a document with metadata: `{ "source": "filename", "policy": "policy_name" }`
- Create a single collection called `hr_policies`
- The script should be idempotent — delete and recreate the collection if it already exists

### 3. Streamlit App (`app.py`)

- Simple single-page Streamlit app
- Text input for the user's question
- On submit, query the ChromaDB collection using the question text
- Return top 5 most relevant results
- Display each result showing:
  - The matched policy text
  - The source policy name
  - The similarity/distance score
- Include a few example queries as clickable buttons:
  - "Can I take my laptop on vacation?"
  - "Can I wear jeans on a Friday?"
  - "How many sick days do I get?"
  - "Do I need VPN when working from home?"
  - "What is the policy on parental leave?"

### 4. Project Setup

- Use `uv` for Python package management (NOT pip, NOT conda)
- Initialize with `uv init` then `uv add` for dependencies
- Required packages: `chromadb`, `streamlit`, `sentence-transformers`
- Python 3.11+ (use whatever uv defaults to)
- Add a `README.md` with:
  - What this project is (learning experiment)
  - How to run: `uv run python ingest.py` then `uv run streamlit run app.py`
  - Brief explanation of what ChromaDB and embeddings are (2-3 sentences)

## Architecture

```
chroma-experiments/
├── CLAUDE.md           (this file — already exists)
├── README.md
├── pyproject.toml      (created by uv init)
├── policies/
│   ├── holidays_and_leaves.md
│   ├── dress_code.md
│   └── technology_use.md
├── ingest.py           (loads policies into ChromaDB)
├── app.py              (Streamlit UI)
└── chroma_db/          (persistent ChromaDB storage — gitignored)
```

## Constraints

- This is a learning project, NOT production code. Keep it simple.
- No external API keys required — everything runs locally.
- No Docker, no deployment config, no CI/CD.
- No over-engineering — no classes, no abstractions, just straightforward scripts.
- Add `chroma_db/` and `.venv/` to `.gitignore`.

## Run Order

1. `uv run python ingest.py` — creates the ChromaDB and loads policies
2. `uv run streamlit run app.py` — launches the query UI at localhost:8501
