# AI-Powered Query Routing Architecture

## Overview

The chatbot now uses **TogetherAI's Llama-3.3-70B-Instruct-Turbo** model for intelligent query classification and routing, replacing the previous regex-based pattern matching system.

## Architecture Flow

```
User Query
    ↓
┌─────────────────────────────────────┐
│ TogetherAI Classification           │
│ (Llama-3.3-70B-Instruct-Turbo)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────┬───────────────────┐
│                 │                   │
│ Direct Response │   [SEARCH_CASES]  │
│                 │                   │
└─────────────────┘                   ↓
    ↓                    ┌──────────────────────┐
    │                    │ RAG Retrieval        │
    │                    │ - G.R. Number Search │
    │                    │ - Keyword Search     │
    │                    │ - Case Title Match   │
    │                    └──────────────────────┘
    │                                ↓
    │                    ┌──────────────────────┐
    │                    │ TogetherAI Generation│
    │                    │ (with case context)  │
    │                    └──────────────────────┘
    │                                ↓
    └────────────────────────────────┘
                    ↓
            Final Response to User
```

## Query Classification Types

### 1. **Greetings & Casual Conversation**
**Examples:**
- "hi", "hello", "hey"
- "thanks", "thank you"
- "bye", "goodbye"
- "how are you"

**Response:** Friendly welcome message with capabilities overview

### 2. **General/Vague Questions**
**Examples:**
- "what are doctrines"
- "what is law"
- "how does court work"
- "explain legal concepts"

**Response:** Polite request for more specific query with examples

### 3. **Specific Legal Queries** → Triggers RAG
**Examples:**
- "G.R. No. 123456"
- "People v. Sanchez ruling"
- "cases about illegal dismissal"
- "what is certiorari" (specific legal term)
- "negligence in tort law"

**Response:** Full RAG retrieval + AI-generated answer with case citations

## Implementation Details

### Key Function: `_classify_and_handle_query()`

**Location:** `backend/chatbot/chat_engine.py` (lines 24-93)

**Returns:**
- `None` → Proceed with RAG retrieval (case search)
- `String` → Direct response (greeting/general question)

**Configuration:**
```python
temperature=0.2  # Low temperature for consistent classification
top_p=0.9
max_tokens=512
```

### Integration Point: `chat_with_law_bot()`

**Location:** `backend/chatbot/chat_engine.py` (lines 1328-1340)

```python
def chat_with_law_bot(query: str, history: List[Dict] = None):
    # Step 1: AI Classification
    ai_response = _classify_and_handle_query(query, history)
    if ai_response is not None:
        return ai_response  # Direct response
    
    # Step 2: RAG Retrieval
    # ... (existing retrieval logic)
```

## Benefits of AI-Powered Routing

### 1. **Natural Language Understanding**
- No need for rigid regex patterns
- Handles variations and typos naturally
- Understands context and intent

### 2. **Simplified Codebase**
- Removed 100+ lines of regex patterns
- No hardcoded response templates
- Self-documenting via system prompt

### 3. **Easy Maintenance**
- Update classification behavior by editing system prompt
- No code changes needed for new query types
- AI learns from examples in prompt

### 4. **Better User Experience**
- More natural, conversational responses
- Contextual awareness
- Consistent tone and style

### 5. **Scalability**
- Easy to add new query types
- Can incorporate conversation history
- Supports multilingual queries (future)

## Performance Considerations

### Latency
- **Classification call:** ~200-500ms
- **Total overhead:** Minimal compared to RAG retrieval
- **Benefit:** Avoids expensive RAG for simple queries

### Cost
- **Cost per classification:** ~$0.0001 (512 tokens)
- **Savings:** Prevents full RAG retrieval for greetings/general questions
- **Net effect:** Cost-neutral or savings

### Fallback Behavior
- If TogetherAI fails → defaults to RAG retrieval
- Ensures system remains functional even during API issues

## Testing Examples

### Before vs After

| Query | Before | After |
|-------|--------|-------|
| "hi" | ❌ Searches for cases with "hi" | ✅ Welcome message |
| "what are doctrines" | ❌ Case search with no results | ✅ Asks for specificity |
| "what is certiorari" | ❌ Would use regex pattern | ✅ AI routes to RAG (needs case context) |
| "G.R. No. 123456" | ✅ Case lookup | ✅ Case lookup (unchanged) |
| "illegal dismissal cases" | ✅ Case search | ✅ Case search (unchanged) |

## Future Enhancements

### Potential Improvements
1. **Conversation Memory:** Use history for context-aware routing
2. **Multi-turn Queries:** Handle follow-up questions intelligently
3. **Intent Confidence:** Return confidence scores for routing decisions
4. **A/B Testing:** Compare AI routing vs rule-based for optimization
5. **Multilingual Support:** Handle Tagalog/Filipino queries

### Monitoring
- Track classification accuracy
- Monitor API response times
- Measure user satisfaction by query type
- Analyze cases where routing fails

## Configuration

### Environment Variables
```bash
TOGETHERAI_API_KEY=your_api_key_here
```

### System Prompt Tuning
Edit the `classification_system` prompt in `_classify_and_handle_query()` to adjust:
- Classification criteria
- Response tone
- Supported query types
- Examples and guidance

## Troubleshooting

### Issue: All queries trigger RAG
**Solution:** Check TogetherAI API key and connectivity

### Issue: Inconsistent classification
**Solution:** Lower temperature (currently 0.2)

### Issue: Wrong query type detected
**Solution:** Add specific examples to system prompt

## Source References

All case digest responses now include a reference link to the [Supreme Court E-Library](https://elibrary.judiciary.gov.ph/) for users to access the complete case text and additional legal resources.

**Implementation:**
- Backend adds reference section to all case responses
- Frontend RichText component supports clickable external links
- Links open in new tab with proper security attributes

## Conclusion

The AI-powered query routing system provides a more intelligent, maintainable, and user-friendly approach to handling diverse queries. By leveraging Llama-3.3-70B's natural language understanding, we've simplified the codebase while improving the user experience. The addition of source references ensures users can verify information and access complete legal documents when needed.

