## PHLaw Chatbot

A simplified Philippine legal chatbot that provides case-specific information from Supreme Court jurisprudence. Features two distinct query paths: exact GR number lookup for detailed case digests, and keyword search for relevant case recommendations.

### Features
- **Two-Path Query System**: GR number exact match vs. keyword search
- **Docker Model Runner**: Primary inference with local LLM fallback
- **Case Type Detection**: Automatic categorization (annulment, criminal, civil, etc.)
- **Simplified Architecture**: Streamlined codebase with essential components only
- **Retrieval-Augmented Generation**: Grounded responses with case citations

### Requirements
- Python 3.10+
- Node 18+ (for Next.js frontend)
- Docker (for model runner)
- (Optional) CUDA-capable GPU for local LLM acceleration

### Quick Start

#### Option 1: Docker Model Runner (Recommended)
1) **Start Docker Model Runner:**
```bash
docker run -p 8001:8001 ai/llama3.2
```

2) **Setup Backend:**
```bash
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

#### Option 2: Local LLM Only
1) **Setup Backend:**
```bash
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

The system will automatically use local LLM if Docker model runner is unavailable.

### Query Types

#### 1. GR Number Queries
**Input:** `"G.R. No. 158088"` or `"What is the ruling in G.R. No. 158088?"`

**Output:** Detailed case digest with:
- Facts
- Issues  
- Rulings
- Full case citation

#### 2. Keyword Queries
**Input:** `"jehan vs nfa"`, `"annulment cases"`, `"find jehan cases"`

**Output:** Top 3 relevant cases:
```
Here are the possible cases:

1. [Case Title] (G.R. No. [number]) — [Case type]
2. [Case Title] (G.R. No. [number]) — [Case type]  
3. [Case Title] (G.R. No. [number]) — [Case type]
```

### API Usage
POST `http://localhost:8000/api/chat/`

Request body:
```json
{
  "query": "G.R. No. 158088",
  "history": []
}
```

Response:
```json
{
  "response": "**Facts:** [case facts]\n\n**Issues:** [legal issues]\n\n**Rulings:** [court decisions]"
}
```

### Architecture

#### Core Components
- **`chat_engine.py`**: Main conversation logic with two-path routing
- **`retriever.py`**: Document retrieval (exact GR match vs. keyword search)
- **`generator.py`**: LLM response generation with Docker/local fallback
- **`docker_model_client.py`**: Docker model runner integration
- **`model_cache.py`**: Centralized model management
- **`embed.py`**: Document embedding and metadata extraction

#### Data Flow
1. **Query Analysis**: Determines GR number vs. keyword path
2. **Retrieval**: Exact metadata search or ensemble retrieval (BM25 + vector)
3. **Generation**: Docker model runner (primary) or local LLM (fallback)
4. **Response**: Formatted case digest or case recommendations

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
The app should be available at `http://localhost:3000` and call the backend at `http://localhost:8000`.

### Testing
Test Docker integration:
```bash
cd backend
python test_docker_integration.py
```

Run Django system checks:
```bash
cd backend
python manage.py check
```

### File Structure
```
backend/chatbot/
├── chat_engine.py          # Main conversation logic
├── retriever.py            # Document retrieval
├── generator.py            # LLM response generation
├── docker_model_client.py  # Docker model integration
├── model_cache.py          # Model management
├── embed.py               # Document embedding
├── serializers.py         # API serializers
├── views.py               # Django views
├── urls.py                # URL routing
└── models.py              # Django models
```

### Simplified Architecture
The codebase has been streamlined to focus on the core two-path functionality:

**Removed Components:**
- Complex intent analysis and query processing
- Advanced legal chunking and metadata extraction
- Citation network analysis
- Ambiguity resolution
- Context-aware translation
- Conversation management
- Debug logging utilities
- Sentence splitting utilities
- Case similarity engines

**Kept Essential Components:**
- Two-path query routing (GR number vs. keywords)
- Basic document retrieval and embedding
- LLM generation with Docker/local fallback
- Case type detection
- Simple metadata extraction

### Development Notes
- **Simplified Logic**: Two-path system (GR number vs. keywords)
- **Docker First**: Uses Docker model runner when available
- **Local Fallback**: Automatic fallback to local LLM
- **Case Types**: Automatic detection and categorization
- **Clean Codebase**: Removed unnecessary components for maintainability

### Troubleshooting

#### Docker Model Runner Issues
- **Connection Failed**: Ensure Docker is running and port 8001 is available
- **Model Not Found**: Verify the model is pulled: `docker images | grep ai/llama3.2`
- **Port Conflicts**: Change port mapping: `docker run -p 8002:8001 ai/llama3.2`

#### Local LLM Issues
- **CUDA Errors**: Check GPU compatibility and drivers
- **Memory Issues**: Reduce `n_batch` and `n_ctx` in `model_cache.py`
- **Model Loading**: Ensure GGUF files are in the correct directory

#### General Issues
- **Import Errors**: Run `pip install -r requirements.txt`
- **Database Issues**: Run `python manage.py migrate`
- **Vector DB**: Ensure Qdrant is running (included in docker-compose)

### Security
- No external API keys required
- All inference happens locally or in Docker containers
- Consider adding rate limiting and authentication before deploying publicly

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python test_docker_integration.py`
5. Submit a pull request


