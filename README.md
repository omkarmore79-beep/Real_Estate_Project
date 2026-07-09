# Real Estate Hybrid Multimodal RAG System

A production-quality document chatbot for real estate brochures, powered by **Hybrid Multimodal RAG** using Qdrant, `BAAI/bge-m3`, `jinaai/jina-clip-v2`, and `BAAI/bge-reranker-large`.

---

## Architecture

```
PDF Upload
    │
    ├─► PyMuPDF text extraction (per page, NO OCR)
    ├─► PDF page → PNG render (per page)
    ├─► Text chunking (500–800 tokens, 100 overlap, with metadata)
    ├─► Image classification (text-context, no OCR)
    │
    ├─► BAAI/bge-m3 → text chunk dense vectors  ──► Qdrant (text collection)
    └─► jinaai/jina-clip-v2 → image + caption vectors ─► Qdrant (image collection)
         + MongoDB GridFS (PDF + images) + MongoDB collection (metadata)

Chat Query
    │
    ├─► Dense text search (Qdrant cosine, bge-m3 query vector)
    ├─► BM25 keyword search (over Qdrant payloads)
    ├─► Image search (when visual intent detected, jina-clip-v2 query vector)
    ├─► RRF Fusion (Reciprocal Rank Fusion, k=60)
    ├─► Reranking (BAAI/bge-reranker-large cross-encoder)
    └─► Grounded Answer (Groq llama-3.3-70b, cites only retrieved chunks)
```

---

## Features

- **No OCR** — text extracted directly from PDF text layer
- **No hallucination** — LLM answers only from retrieved evidence
- **Hybrid search** — dense + BM25 keyword + image-aware retrieval
- **RRF + reranking** — for best result quality
- **Citations** — every answer includes source file, page number, snippet
- **Image results** — floor plans, location maps, amenities with captions
- **Confidence scoring** — high / medium / low based on evidence strength
- **Legacy fallback** — if Qdrant is unavailable, falls back to MongoDB keyword search

---

## Quick Start

### 1. Start Qdrant
```bash
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Start MongoDB
```bash
# Local
mongod --dbpath /data/db

# Or use MongoDB Atlas URI in .env
```

### 3. Set up environment
```bash
cp .env.example .env
# Edit .env with your GROQ_API_KEY and MONGODB_URI
```

### 4. Install backend dependencies
```bash
cd backend
pip install -r ../requirements.txt
```

> **Note:** First run will download ~5 GB of models from HuggingFace:
> - `BAAI/bge-m3` (~2.3 GB)
> - `jinaai/jina-clip-v2` (~1.5 GB)
> - `BAAI/bge-reranker-large` (~1.4 GB)

### 5. Start the backend
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 6. Start the frontend
```bash
cd frontend-next
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `MONGODB_URI` | — | **Required.** MongoDB connection URI |
| `MONGODB_DB` | `real_estate_chatbot` | Database name |
| `MONGODB_COLLECTION` | `projects` | Collection name |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant API key (for cloud) |
| `QDRANT_COLLECTION_TEXT` | `real_estate_text_chunks` | Text vector collection |
| `QDRANT_COLLECTION_IMAGES` | `real_estate_image_chunks` | Image vector collection |
| `TEXT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Text embedding model |
| `IMAGE_EMBEDDING_MODEL` | `jinaai/jina-clip-v2` | Image embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Reranker model |
| `TEXT_VECTOR_DIM` | `1024` | bge-m3 dense dimension |
| `IMAGE_VECTOR_DIM` | `512` | jina-clip-v2 dimension |

---

## API Reference

### Upload
```http
POST /upload
Content-Type: multipart/form-data

file: <PDF>
builder: "Builder Name"
project: "Project Name"
document_type: "Brochure"
title: "Document Title"
```

**Response:**
```json
{
  "document_id": "abc123",
  "total_pages": 12,
  "text_chunks_indexed": 48,
  "images_indexed": 12,
  "qdrant_status": "success",
  "ocr_used": false,
  "message": "Document uploaded and indexed successfully."
}
```

---

### Chat
```http
POST /chat
Content-Type: application/json

{
  "message": "Show the floor plan and explain room types",
  "document_id": "abc123",
  "top_k": 8,
  "include_images": true
}
```

**Response:**
```json
{
  "question": "Show the floor plan…",
  "answer": "The project offers 2 BHK and 3 BHK configurations…",
  "citations": [
    {
      "document_id": "abc123",
      "source_file": "brochure.pdf",
      "page_number": 5,
      "source_type": "text",
      "snippet": "2 BHK carpet area: 650 sq.ft…"
    }
  ],
  "images": [
    {
      "document_id": "abc123",
      "image_id": "page_5",
      "image_url": "/documents/abc123/images/page_5",
      "page_number": 5,
      "image_type": "floor_plan",
      "caption": "Floor plan on page 5."
    }
  ],
  "confidence": "high",
  "intent": { "requires_image": true, "requires_text": true }
}
```

---

### RAG Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/rag/health` | Check Qdrant, collections, model status |
| `POST` | `/rag/search` | Raw hybrid retrieval (debug) |
| `POST` | `/rag/reindex/{document_id}` | Rebuild Qdrant index for one document |
| `DELETE` | `/rag/index/{document_id}` | Delete Qdrant vectors for one document |

---

## Running Tests

```bash
# Upload a test PDF and run all RAG tests
python tests/test_rag_pipeline.py --pdf /path/to/real_estate_brochure.pdf

# Use a different backend
python tests/test_rag_pipeline.py --pdf brochure.pdf --backend http://localhost:8000
```

---

## Grounding Behaviour

The chatbot will **never** invent or guess:
- Project prices, cost per sq.ft., payment schedules
- RERA / MahaRERA registration numbers
- Possession / handover dates
- Carpet area, super built-up area, saleable area
- Legal approvals, NOC details
- Amenities lists
- Builder claims or developer promises

If retrieved context does not contain the answer:
> "Data not available in the uploaded documents."

---

## Troubleshooting

### Qdrant not running
```
Error: Connection refused to http://localhost:6333
Fix:  docker run -d -p 6333:6333 qdrant/qdrant
```

### Models loading slowly
First startup downloads ~5 GB from HuggingFace. Subsequent starts use cached models (`~/.cache/huggingface/`). On CPU-only machines, embedding is slow — GPU strongly recommended for production.

### MongoDB Atlas SSL handshake failed fixes
If you receive a connection error or a `503 Service Unavailable` on `/upload` with a TLS/SSL handshake error like `[SSL: TLSV1_ALERT_INTERNAL_ERROR]`:

1. **Install/upgrade dependencies**:
   ```bash
   pip install --upgrade pymongo dnspython certifi python-dotenv
   ```
2. **Use certifi in MongoClient**:
   Make sure you are initializing MongoClient using `tls=True` and `tlsCAFile=certifi.where()` (already integrated in the code).
3. **Add current IP in MongoDB Atlas Network Access**:
   - Log into your MongoDB Atlas Console.
   - Go to **Network Access** under the Security tab.
   - Click **Add IP Address**.
   - Click **Add Current IP Address** or configure `0.0.0.0/0` (allow access from anywhere) for development.
4. **Give database user readWrite permission**:
   Verify in Database Access console that your user has `readWrite` permission on the cluster/database.
5. **URL encode special characters in password**:
   If your password contains special characters (e.g. `@`, `/`, `+`), URL encode them (e.g., `@` becomes `%40`).
6. **Try mobile hotspot if college/office WiFi blocks MongoDB Atlas**:
   Firewalls sometimes block port `27017` which Atlas uses. Switching to a mobile hotspot or personal network resolves this.
7. **Rotate password if URI was shared**.

### Setup and Verification Steps
Run the following commands in order to upgrade dependencies, verify the MongoDB connection, and start the backend:

```bash
cd backend
.venv\Scripts\activate
pip install --upgrade pymongo dnspython certifi python-dotenv
python scripts/test_mongo_connection.py
uvicorn app:app --host 127.0.0.1 --port 8000
```

> [!WARNING]
> **Server Reloads Interrupt Background Tasks**:
> When running FastAPI with `uvicorn --reload`, editing your python files triggers a server restart. If a document is currently processing in the background, it will be interrupted and marked as failed. For stable long-running processing, start without reload using `uvicorn app:app --host 127.0.0.1 --port 8000`.


Then test the MongoDB health endpoint:
```http
GET http://127.0.0.1:8000/health/mongo
```

---

### Local Development Fallback Mode (No MongoDB)
If MongoDB Atlas is completely blocked due to strict college/corporate firewalls, network TLS restrictions, or whitelisting issues, you can enable the **Local Development Fallback Mode**:

1. Open `backend/.env`.
2. Add:
   ```env
   ALLOW_UPLOAD_WITHOUT_MONGODB=true
   ```
3. Restart the backend server.
This allows the backend to:
* Skip MongoDB Atlas completely during `/upload`.
* Save metadata locally to `backend/storage/local_documents.json`.
* Save uploaded files/images locally to `backend/storage/local_files/`.
* Run the multimodal RAG embedding and Qdrant Cloud vector indexing smoothly.

---

### Qdrant Cloud Setup (No Docker required)
The RAG pipeline is fully optimized for **Qdrant Cloud** and does not require local Docker or local Qdrant processes:
1. Create a free cluster on [Qdrant Cloud Console](https://cloud.qdrant.io).
2. Set `QDRANT_URL` to your cluster URL (e.g. `https://xxx.cloud.qdrant.io`).
3. Set `QDRANT_API_KEY` to your API Key token.
4. Verify connectivity using the endpoint:
   ```http
   GET http://127.0.0.1:8000/rag/health
   ```

---


### Next.js Dynamic Route Slug Conflict
If you receive the error:
`Error: You cannot use different slug names for the same dynamic path ('id' !== 'document_id').`
This means the old `src/app/api/documents/[id]` directory conflicts with the standardized `[document_id]` routes.
1. Run `rmdir /s /q src\app\api\documents\[id]` (or in PowerShell: `Remove-Item -Recurse -Force -LiteralPath src/app/api/documents/[id]`).
2. Run `rmdir /s /q .next` to clear Next.js compile cache.
3. Restart development server with `npm run dev`.

### PaddleOCR Installation / Execution Issues
If PaddleOCR fails to initialize, ensure the system dependencies for OpenCV and PaddlePaddle are satisfied:
- Install headlessly with `pip install opencv-python-headless` (included in `requirements.txt`).
- Run the OCR verification script:
  ```bash
  python scripts/test_ocr.py <path_to_image_or_pdf>
  ```

---

## Project Structure

```
Real_Estate_Project/
├── backend/
│   ├── app.py                        # FastAPI app (all routes)
│   ├── config.py                     # Environment config
│   ├── chatbot/
│   │   ├── chat_handler.py           # Legacy keyword+LLM handler
│   │   └── grounded_answer.py        # ★ NEW: Strict RAG answer generator
│   ├── ingestion/
│   │   ├── pdf_processor.py          # ★ NEW: PyMuPDF + OCR extraction
│   │   ├── ocr_service.py            # ★ NEW: Lazy-loaded PaddleOCR engine
│   │   ├── chunker.py                # ★ NEW: Token-aware text chunker
│   │   ├── image_processor.py        # ★ NEW: Text-context image metadata
│   │   ├── image_extractor.py        # Existing: page renderer
│   │   └── image_analyzer.py         # Existing: Groq Vision + text fallback
│   ├── retrieval/
│   │   ├── qdrant_client.py          # ★ NEW: Centralized Qdrant Cloud client
│   │   ├── qdrant_service.py         # ★ NEW: Qdrant vector operations
│   │   ├── embeddings.py             # ★ NEW: bge-m3 + jina-clip-v2
│   │   ├── hybrid_retriever.py       # ★ NEW: Dense+BM25+Image+RRF+Rerank
│   │   ├── image_retriever.py        # Existing: MongoDB image scoring
│   │   └── intent_classifier.py      # Existing: rule-based intent
│   ├── scripts/
│   │   ├── test_mongo_connection.py  # Verify MongoDB connectivity
│   │   ├── test_qdrant_connection.py # Verify Qdrant Cloud connectivity
│   │   └── test_ocr.py               # Verify PaddleOCR execution
│   └── storage/
│       └── mongo_store.py            # Existing: MongoDB/GridFS storage
├── frontend-next/
│   └── src/
│       ├── app/
│       │   ├── chat/page.tsx          # ★ Updated: citations, confidence, images, OCR badge
│       │   └── documents/upload/page.tsx  # ★ Updated: RAG stats & progress
│       └── lib/
│           └── backend-data.ts        # ★ Updated: RAG types
├── requirements.txt                   # ★ Updated: full production deps
└── .env.example                       # ★ NEW: all env vars documented
```

