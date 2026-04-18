# Chroma Embeddings Experiment

A learning project to experiment with ChromaDB vector embeddings. It ingests a handful of sample HR policies into a local vector database and provides a Streamlit UI for semantic search over them.

This README is written for juniors who are just starting to learn about AI and agentic development. The code itself is intentionally tiny — the interesting part is understanding *why* each piece is there.

## How to run

Prerequisites: [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv run python ingest.py         # creates ./chroma_db and loads policies
uv run streamlit run app.py     # launches the query UI at localhost:8501
```

Re-running `ingest.py` is safe — it drops and recreates the collection every time.

## Layout

```
chromadb-experiment/
├── CLAUDE.md           spec for this project
├── README.md
├── pyproject.toml
├── policies/
│   ├── holidays_and_leaves.md
│   ├── dress_code.md
│   └── technology_use.md
├── ingest.py
├── app.py
└── chroma_db/          persistent ChromaDB storage (gitignored)
```

---

## 1. The role of `all-MiniLM-L6-v2` embeddings

An **embedding** is a list of numbers (a *vector*) that represents the *meaning* of a piece of text. Text about "vacation days" ends up with a vector that is numerically close to text about "PTO allowance", even though they share no keywords. Text about "laptops" ends up far away.

`all-MiniLM-L6-v2` is a small open-source model from the `sentence-transformers` family. In this project it plays exactly one role:

> **Turn text into a 384-dimensional vector.**

It runs locally on your machine, takes a few hundred milliseconds to load, and produces the same vector every time for the same input. We use it in two places:

1. **At ingest time** — we pass each bullet from the policy files to the model and store the resulting vector in ChromaDB alongside the original text.
2. **At query time** — we pass the user's question through the *same* model to get a query vector, then ask ChromaDB for the stored vectors closest to it.

There are bigger, more accurate embedding models (OpenAI's `text-embedding-3-large`, Cohere's embed models, Google's `gecko`). MiniLM is deliberately chosen here because it is small (~80 MB), free, and runs offline — perfect for learning. ChromaDB uses it as its default when you don't configure anything else.

## 2. The nature of vector databases, and why ChromaDB here

A **vector database** is a database that stores vectors and makes one specific operation fast: *given a query vector, find the N stored vectors that are closest to it.* "Closest" is measured by cosine similarity or Euclidean distance in that high-dimensional space.

That is very different from what a normal database (Postgres, MySQL) does. A normal database is great at *exact* matching — "find the row where `employee_id = 42`". It is terrible at *semantic* matching — "find the policy that answers this question in natural language", because there is no SQL `WHERE` clause for "means roughly the same thing as".

Vector databases solve that with specialized indexes (HNSW, IVF, etc.) that can search millions of vectors in milliseconds without comparing against every single one.

**Why ChromaDB for this experiment?**

- **Zero setup.** `pip install chromadb` and you're done. No server to run, no credentials to manage. It just writes files to a local directory.
- **Batteries included.** It ships with a default embedding function, so beginners don't have to pick and wire up an embedding model on day one.
- **Small, readable API.** `client.create_collection`, `collection.add`, `collection.query`. That's most of what you need.
- **Open source and free.** No vendor lock-in while you're learning.

Alternatives you'll hear about: **pgvector** (Postgres extension — good when you already have Postgres), **Qdrant** and **Weaviate** (more feature-rich, run as servers), **Pinecone** (managed cloud service). For production at scale you would usually move to one of those; ChromaDB is ideal for prototypes and single-machine workloads.

## 3. Why we don't need an active connection to an LLM

This app performs **semantic search**, not **generation**. That distinction matters a lot.

- An **LLM** (GPT-4, Claude, Llama) *generates* new text — it writes answers, summaries, code, etc. It is expensive, network-bound, and non-deterministic.
- An **embedding model** *measures meaning*. It doesn't write anything. It just converts text to a vector. It is small, cheap, and deterministic.

All this app does is:

1. Convert the user's question into a vector.
2. Find the nearest stored bullets.
3. Display those bullets verbatim.

Nowhere in that pipeline do we ask anyone to *write a sentence*. We just retrieve and display text that was already written. That's why no LLM is needed, no API keys are required, and the whole thing works offline.

The step that **would** require an LLM is turning the retrieved bullets into a conversational answer — "Based on your policies, yes, you can take your laptop on vacation but must keep it with you in carry-on luggage and use a privacy screen." That pattern is called **RAG** (Retrieval-Augmented Generation): use a vector DB to retrieve the relevant facts, then pass them to an LLM to write the answer. This project is the "R" half of RAG without the "G". Once you understand this half, adding the LLM step is straightforward.

## 4. How can you scale this app?

The current setup handles ~30 documents on a laptop. Scaling it out happens in stages:

**More documents (thousands to millions).** ChromaDB on a single machine will comfortably handle hundreds of thousands of vectors. Beyond that you want a server-backed vector DB (Qdrant, Weaviate, pgvector, Pinecone). You also start caring about batching — embedding one bullet at a time is fine for 30 items but wasteful for 30,000; pass them to the embedding model in batches of 32–128 instead.

**Bigger documents.** Real-world policies, PDFs, and web pages are long. You can't embed a 50-page document as one vector — the meaning gets blurred. You need a **chunking strategy**: split documents into overlapping windows of ~200–500 tokens, embed each chunk, and store the chunks with metadata pointing back to the source. Libraries like LangChain and LlamaIndex have ready-made chunkers.

**Better retrieval quality.** Pure vector search can miss things that keyword search would catch (exact names, product SKUs, error codes). **Hybrid search** combines vector similarity with BM25 keyword scoring and re-ranks the results. A **re-ranker model** (like `cross-encoder/ms-marco-MiniLM-L-6-v2`) can re-score the top 50 hits to get a better top 5.

**More concurrent users.** Streamlit is single-process and not designed for high concurrency. A production UI would be a proper web app (Next.js, SvelteKit, FastAPI + React) with the vector DB behind an API server.

**Lower latency.** Cache embeddings for hot queries. Keep the embedding model loaded in memory between requests (we already use `@st.cache_resource` for ChromaDB — same principle applies to the model). Use a GPU-backed or API-hosted embedding service if CPU embedding becomes a bottleneck.

## 5. What would it take to take this to production?

This code is explicitly a learning experiment. To ship something real on top of the same idea, at minimum you would need to think about:

**Data pipeline**
- Real documents ingested from a source of truth (Confluence, Notion, a CMS, a file store) — not hand-written markdown.
- Scheduled re-ingestion when documents change, with deletions and updates (not just rebuilding from scratch every time).
- A chunking strategy that preserves enough context per chunk for meaningful retrieval.

**Quality and evaluation**
- A **test set** of real questions with known-good answers, and a script that measures retrieval accuracy (recall@k, MRR) whenever you change the embedding model, chunk size, or DB.
- Monitoring for queries that returned nothing useful, so you can find gaps in your content.

**Safety and correctness**
- Access control — employees should only see policies they are allowed to see. Metadata filtering in the vector DB (ChromaDB supports `where` clauses) handles per-team/per-role visibility.
- PII handling — don't embed or log sensitive data you don't need.
- Source attribution in the UI so users can verify the answer against the original document.
- If you add an LLM on top (to generate answers), you need guardrails against hallucination — grounding the answer in the retrieved text, refusing to answer when no relevant results come back, and making citations clickable.

**Operations**
- A proper vector DB: Qdrant, Weaviate, pgvector, or Pinecone, behind an API. ChromaDB on a laptop isn't a production deployment story.
- Observability: logs, metrics, traces on ingestion runs and query latency.
- CI/CD, containerization, a real web frontend, authentication, rate limiting.
- Cost monitoring — embedding calls and LLM calls both cost money at scale.

**Agentic extensions**
- The next step beyond RAG is **agentic retrieval**: an LLM agent decides *which* tools to call (search, fetch a specific doc, query a DB, ask a clarifying question) in a multi-turn loop. ChromaDB would be one tool among several. Frameworks like the Claude Agent SDK, LangGraph, and CrewAI are designed for this pattern.

---

The gap between "30 bullets on a laptop" and "production RAG over all of your company's knowledge" is significant, but the core idea — embed text, store vectors, retrieve by similarity — is the same. Understanding this small version well is the foundation for all of it.
