# Copilot Instructions for phlaw-chatbot

## Overview
This project is a full-stack legal chatbot platform for Philippine jurisprudence, consisting of:
- **Backend**: Django REST API (in `backend/`), with a custom chatbot pipeline combining rule-based, retrieval-augmented, and LLM-based answering. Integrates with Qdrant for vector search and uses local LLMs via `llama-cpp-python`.
- **Frontend**: Next.js (in `frontend/`), providing a chat UI that communicates with the backend at `/api/chat/`.
- **Vector DB**: Qdrant (see `docker-compose.yml`), used for semantic search over legal documents.

## Key Architectural Patterns
- **Chat Flow**: User queries are POSTed to `/api/chat/` (see `chatbot/views.py`). The backend first tries a rule-based responder (`rule_based.py`), then falls back to retrieval + LLM generation (`retriever.py`, `generator.py`).
- **Retrieval**: Legal documents are embedded (see `embed.py`) and indexed in Qdrant. Retrieval is section-aware and uses intent/section weighting.
- **LLM**: Uses a local GGUF model via `llama-cpp-python` (see `generator.py`).
- **Rule-based**: Handles small-talk, math, and general queries before invoking retrieval/LLM.

## Developer Workflows
- **Backend**
  - Run locally: `python backend/manage.py runserver`
  - Test: `python backend/manage.py test chatbot`
  - Embedding pipeline: Run `embed.py` to (re)index documents in Qdrant.
  - Environment: Set variables in `.env` (see `embed.py` for required keys).
- **Frontend**
  - Run locally: `cd frontend && npm run dev`
  - Main entry: `src/app/page.tsx` (calls backend at `http://127.0.0.1:8000/api/chat/`)
- **Docker**
  - See `Dockerfile` and `docker-compose.yml` for multi-service setup (backend, frontend, Qdrant).

## Project-Specific Conventions
- **API**: All chat requests use a single endpoint (`/api/chat/`), expecting `{ "query": ... }` and returning `{ "response": ... }`.
- **Section Extraction**: Uses regexes for extracting facts/issues/rulings (see `chat_engine.py`).
- **Embeddings**: Uses `Stern5497/sbert-legal-xlm-roberta-base` by default (configurable via env).
- **Testing**: See `chatbot/tests.py` for API test examples.
- **No auth/CSRF**: Disabled for local testing (see `ChatView`).

## Integration Points
- **Qdrant**: Vector DB for semantic search (see `retriever.py`, `embed.py`).
- **llama-cpp-python**: Local LLM inference (see `generator.py`).
- **REST API**: Consumed by Next.js frontend.

## References
- Backend: `backend/chatbot/` (core logic), `backend/backend/settings.py`, `backend/backend/urls.py`
- Frontend: `frontend/src/app/page.tsx`
- Embedding: `backend/chatbot/embed.py`
- Retrieval: `backend/chatbot/retriever.py`
- LLM: `backend/chatbot/generator.py`

---

**If you are an AI agent, follow these conventions and reference the above files for implementation details.**
