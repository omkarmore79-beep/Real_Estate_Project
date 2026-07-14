# RAG-as-a-Service Platform: 10-Level Architecture

This document outlines the scalable 10-level architecture for the Hyundai Heavy Machinery MVP (Excavator R215L) and future generic RAG deployments.

## Phase 1 & 2: MVP Capabilities (Already Built/Present)

### 1. Ingest + Normalise
*   **Current State:** PyMuPDF extracts text directly from PDFs (with PaddleOCR fallback). Images and diagrams are rendered separately. 
*   **Excavator Use Case:** Technical manuals, maintenance schedules, and machine diagrams (hydraulic systems, engine cross-sections) are extracted. Metadata is normalized across all documents.

### 2. Hybrid Retrieval
*   **Current State:** Uses dense vectors (`BAAI/bge-m3`) for semantic understanding and BM25 for exact keyword matching.
*   **Excavator Use Case:** Crucial for matching exact part numbers (e.g., `PX-215-L`) while also understanding semantic queries like "Why is the engine overheating?".

### 3. ANN + Reranking
*   **Current State:** Qdrant handles Approximate Nearest Neighbor (ANN) search. Results are re-ranked using a cross-encoder (`BAAI/bge-reranker-large`).
*   **Excavator Use Case:** Ensures the most relevant manual page containing the specific breakdown solution is ranked #1 before being sent to the LLM.

### 4. Source Confidence Scoring
*   **Current State:** The system calculates a confidence score (High/Medium/Low) based on the relevance of the retrieved evidence.
*   **Excavator Use Case:** If a user asks an obscure question not covered in the manual, the chatbot knows to flag it with low confidence rather than guessing.

### 5. Constrained Generation
*   **Current State:** The Groq LLM (`llama-3.3-70b-versatile`) uses a strict system prompt forbidding it from answering questions using outside knowledge.
*   **Excavator Use Case:** The LLM acts as a strict diagnostic assistant, answering "What causes hydraulic failure?" using *only* the official Hyundai manual.

### 6. Citation-backed Responses
*   **Current State:** Every answer includes exact citations (Document, Page Number, and Snippet).
*   **Excavator Use Case:** Mechanics on the field can verify the LLM's advice by clicking the citation to see the exact page and diagram from the service manual.

### 7. Hallucination Fallback
*   **Current State:** If the required info isn't in the context, the system safely falls back to "Data not available in the uploaded documents."
*   **Excavator Use Case:** Prevents the chatbot from suggesting incorrect or dangerous repair steps that aren't approved by the manufacturer.

---

## Phase 3 & 4: Future Scale (Post-Pitch Implementation)

### 8. Continuous Evals
*   **Plan:** Integrate evaluation frameworks like **RAGAS** or **TruLens**.
*   **Excavator Use Case:** Automatically measure "Context Precision" and "Answer Faithfulness" across hundreds of test queries to guarantee the bot isn't drifting.

### 9. Caching + Memory
*   **Plan:** Implement semantic caching (e.g., Redis with vector search or GPTCache).
*   **Excavator Use Case:** When multiple operators ask "How to reset the oil filter light?", the system returns the cached answer instantly, saving LLM token costs and reducing latency.

### 10. Observability
*   **Plan:** Add tracing via **LangSmith** or **Phoenix**.
*   **Excavator Use Case:** Track exactly how many tokens are being used per diagnostic session, identify which manuals have poor retrieval rates, and monitor the overall health of the Qdrant cluster.
