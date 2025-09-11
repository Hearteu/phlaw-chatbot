## PHLaw Chatbot

Backend + frontend chatbot focused on Philippine Supreme Court jurisprudence. The bot can ground answers on retrieved cases or reply conversationally from its pretrained knowledge using a local llama.cpp model.

### Features
- Conversational memory via `history` in requests
- Intent detection to decide when to retrieve jurisprudence vs. answer directly
- Retrieval-augmented generation with section-aware context (facts, issues, ruling)
- Local `llama_cpp` inference

### Requirements
- Python 3.10+
- Node 18+ (for Next.js frontend)
- (Optional) CUDA-capable GPU if you use local llama acceleration

### Setup (Backend)
1) Create a virtual environment and install deps:
```bash
pip install -r requirements.txt
```

2) Configure environment: (no external API required)
```bash
# No additional environment variables needed for local inference
```

3) Run Django:
```bash
cd backend
python manage.py migrate
python manage.py runserver
```

### API Usage
POST `http://localhost:8000/api/chat/`

Request body (history optional):
```json
{
  "query": "Give me the ruling for ...",
  "history": [
    {"role": "user", "content": "Summarize the facts."},
    {"role": "assistant", "content": "Facts: ... [1]"}
  ]
}
```

Response:
```json
{ "response": "Ruling: \"...\" [1]" }
```

Behavior:
- The engine detects if the query needs jurisprudence (e.g., mentions of G.R. numbers, articles/sections, “ruling/facts/issues”, case name patterns). If yes, it retrieves case snippets and grounds the answer with citations.
- Otherwise, it answers conversationally using the local model. You can adjust triggers in `backend/chatbot/chat_engine.py` (`_should_query_jurisprudence`).

### Inference Details
- Configured in `backend/chatbot/generator.py`.
- Uses local `llama_cpp` model via `backend/chatbot/model_cache.py`.

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
The app should be available at `http://localhost:3000` and call the backend at `http://localhost:8000`.

### Docker
Build and run with docker-compose:
```bash
docker compose up --build
```
No external API configuration is required.

### Development Notes
- Main flow: `ChatView` → `chat_engine.chat_with_law_bot` → `generator.generate_response(_from_messages)`.
- Retrieval logic and prompt construction live in `backend/chatbot/chat_engine.py`.
-- All responses are generated locally by default.

### Security
- Do not hardcode API keys in code or commit history. Use environment variables.
- Consider adding rate limiting and authentication before deploying publicly.

### Testing
Run Django system checks:
```bash
cd backend
python manage.py check
```
You can add tests under `backend/chatbot/tests.py` for intent detection and API contract.


