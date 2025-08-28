## PHLaw Chatbot

Backend + frontend chatbot focused on Philippine Supreme Court jurisprudence. The bot can ground answers on retrieved cases or reply conversationally from its pretrained knowledge, integrating the Rev21 Labs AI API with a local llama.cpp fallback.

### Features
- Conversational memory via `history` in requests
- Intent detection to decide when to retrieve jurisprudence vs. answer directly
- Retrieval-augmented generation with section-aware context (facts, issues, ruling)
- Rev21 Labs API integration (primary) with local `llama_cpp` fallback

### Requirements
- Python 3.10+
- Node 18+ (for Next.js frontend)
- (Optional) CUDA-capable GPU if you use local llama acceleration

### Setup (Backend)
1) Create a virtual environment and install deps:
```bash
pip install -r requirements.txt
```

2) Configure environment (Rev21 optional; defaults provided):
```bash
# Windows (cmd)
set REV21_BASE_URL=https://ai-tools.rev21labs.com/api/v1
set REV21_ENDPOINT_PATH=/chat/completions
set REV21_API_KEY=YOUR_KEY
set REV21_ENABLED=true

# Linux/macOS (bash)
export REV21_BASE_URL=https://ai-tools.rev21labs.com/api/v1
export REV21_ENDPOINT_PATH=/chat/completions
export REV21_API_KEY=YOUR_KEY
export REV21_ENABLED=true
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
- Otherwise, it answers conversationally (still using Rev21 when available). You can adjust triggers in `backend/chatbot/chat_engine.py` (`_should_query_jurisprudence`).

### Rev21 Integration Details
- Configured in `backend/chatbot/generator.py`.
- Uses `REV21_BASE_URL`, `REV21_ENDPOINT_PATH`, `REV21_API_KEY`, `REV21_ENABLED`.
- Attempts OpenAI-like chat completion first; falls back to local llama.cpp model if API is disabled, missing, or errors.

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
Set environment variables via `.env` or compose overrides to configure Rev21.

### Development Notes
- Main flow: `ChatView` → `chat_engine.chat_with_law_bot` → `generator.generate_response(_from_messages)`.
- Retrieval logic and prompt construction live in `backend/chatbot/chat_engine.py`.
- If you see latency or rate limits on Rev21, set `REV21_ENABLED=false` to use local llama.

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


