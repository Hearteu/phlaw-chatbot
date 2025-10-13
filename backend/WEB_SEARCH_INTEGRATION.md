# Web Search Integration with Google Custom Search API

## Overview

The chatbot now supports web search functionality using Google Custom Search JSON API to provide current legal information, recent developments, and access to sources not available in the local database.

## Features

### ✅ **Current Information Access**
- Recent Supreme Court decisions
- Latest legal updates and amendments
- Current legal news and developments
- Recent court rulings and interpretations

### ✅ **Constitutional & Statutory References**
- Current constitutional provisions
- Recent statutory law changes
- Legal codes and regulations
- Government legal documents

### ✅ **Legal Procedures & Requirements**
- How-to information for legal processes
- Current filing requirements
- Legal procedure guidelines
- Administrative requirements

### ✅ **Source Attribution**
- Direct links to official sources
- Source domain identification
- Relevance scoring and ranking
- Multiple source synthesis

## Architecture

### **Query Flow**
```
User Query → AI Classification → [WEB_SEARCH] → Google Custom Search → TogetherAI Response → User
```

### **Components**

#### 1. **LegalWebSearcher** (`chatbot/web_search.py`)
- Google Custom Search API integration
- Philippine legal source filtering
- Relevance scoring and ranking
- Rate limiting and error handling

#### 2. **Enhanced Classification** (`chatbot/chat_engine.py`)
- Web search query detection
- Integration with existing AI routing
- Follow-up question handling

#### 3. **Response Generation**
- TogetherAI-powered synthesis
- Source attribution
- Fallback responses

## Configuration

### **Required API Keys**

1. **Google Custom Search API Key**
   - Get from: https://developers.google.com/custom-search/v1/introduction
   - Enable Custom Search JSON API

2. **Custom Search Engine ID**
   - Create at: https://cse.google.com/cse/
   - Configure to search Philippine legal sources

### **Environment Variables**
```bash
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id_here
```

### **Django Settings**
```python
# backend/backend/settings.py
GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY', '')
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
WEB_SEARCH_ENABLED = bool(GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID)
```

## Usage Examples

### **Recent Legal Developments**
```
User: "recent Supreme Court decisions on labor law"
Bot: [Searches web] → Provides current decisions with sources
```

### **Current Legal Information**
```
User: "current tax law provisions for businesses"
Bot: [Searches web] → Provides latest tax law information
```

### **Legal Procedures**
```
User: "how to file a petition for certiorari"
Bot: [Searches web] → Provides current procedure guidelines
```

### **Constitutional References**
```
User: "current interpretation of due process clause"
Bot: [Searches web] → Provides recent constitutional interpretations
```

## Query Classification

### **Web Search Triggers**
- "recent", "latest", "current" + legal topics
- Constitutional provisions
- Statutory laws and codes
- Legal procedures and requirements
- Legal news and developments

### **Local Database Triggers**
- Specific G.R. numbers
- Case names and parties
- Follow-up questions about discussed cases
- Specific legal doctrines in context

## Legal Sources

### **Prioritized Sources**
1. **elibrary.judiciary.gov.ph** - Supreme Court E-Library
2. **sc.judiciary.gov.ph** - Supreme Court Official
3. **lawphil.net** - LawPhil Legal Database
4. **chanrobles.com** - ChanRobles Legal Database
5. **officialgazette.gov.ph** - Official Gazette
6. **supremecourt.gov.ph** - Supreme Court

### **Source Filtering**
- Philippine legal sources prioritized
- Government domains boosted
- Relevance scoring based on content
- Official source preference

## Response Format

### **Structured Response**
```
**Current Information Summary**

[AI-generated synthesis of web search results]

**Key Points:**
- Point 1 with source attribution
- Point 2 with source attribution
- Point 3 with source attribution

---
📚 **Sources:**
- [Title 1](URL1) - Source Domain
- [Title 2](URL2) - Source Domain
- [Title 3](URL3) - Source Domain
```

## Error Handling

### **Graceful Degradation**
- API key missing → Inform user, suggest local search
- API rate limit → Retry with backoff
- Network error → Fallback to local database
- No results → Suggest alternative queries

### **User Experience**
- Clear error messages
- Alternative suggestions
- Fallback to local case search
- Transparent source attribution

## Rate Limiting

### **Configuration**
- 100 requests per hour per user (configurable)
- 1 second minimum between searches
- Automatic retry with exponential backoff

### **Implementation**
```python
# Rate limiting in LegalWebSearcher
self.min_search_interval = 1  # seconds
self.last_search_time = 0
```

## Testing

### **Test Function**
```python
from chatbot.web_search import test_web_search
test_web_search()
```

### **Manual Testing**
1. Set environment variables
2. Test with sample queries
3. Verify source attribution
4. Check response quality

## Benefits

### **Enhanced Capabilities**
✅ **Current Information** - Access to latest legal developments  
✅ **Constitutional Context** - Current constitutional provisions  
✅ **Statutory Updates** - Recent changes to laws and codes  
✅ **Legal News** - Current legal events and court decisions  
✅ **Procedural Information** - How to file cases, requirements  
✅ **Source Attribution** - Links to official sources  

### **Educational Value**
- Teaches users about current legal landscape
- Provides context for legal developments
- Links to authoritative sources
- Encourages further research

### **Professional Accuracy**
- Official source prioritization
- Multiple source verification
- Transparent attribution
- Current information emphasis

## Future Enhancements

### **Potential Improvements**
1. **Caching** - Cache frequent searches
2. **Personalization** - User preference learning
3. **Advanced Filtering** - Date range, jurisdiction filters
4. **Integration** - Combine with local case analysis
5. **Analytics** - Search pattern tracking

### **Advanced Features**
- Legal document parsing
- Citation extraction
- Precedent linking
- Legal timeline creation

## Troubleshooting

### **Common Issues**

#### **API Credentials**
```bash
# Check environment variables
echo $GOOGLE_SEARCH_API_KEY
echo $GOOGLE_SEARCH_ENGINE_ID
```

#### **Rate Limiting**
- Wait 1 second between searches
- Check API quota limits
- Monitor usage in Google Cloud Console

#### **No Results**
- Verify search engine configuration
- Check legal source inclusion
- Adjust query specificity

#### **Network Issues**
- Check internet connectivity
- Verify API endpoint accessibility
- Check firewall settings

## Security Considerations

### **API Key Protection**
- Store in environment variables
- Never commit to version control
- Use different keys for dev/prod
- Monitor usage and costs

### **Rate Limiting**
- Implement user-based limits
- Monitor for abuse
- Set reasonable quotas
- Alert on unusual usage

## Cost Management

### **Google Custom Search API**
- 100 free searches per day
- $5 per 1000 queries after free tier
- Monitor usage in Google Cloud Console
- Set up billing alerts

### **Optimization**
- Cache frequent searches
- Implement smart query routing
- Use local database when possible
- Optimize search parameters

---

## Quick Start

1. **Get API Keys**
   - Google Custom Search API key
   - Custom Search Engine ID

2. **Set Environment Variables**
   ```bash
   export GOOGLE_SEARCH_API_KEY="your_key_here"
   export GOOGLE_SEARCH_ENGINE_ID="your_engine_id_here"
   ```

3. **Test Integration**
   ```python
   from chatbot.web_search import test_web_search
   test_web_search()
   ```

4. **Use in Chat**
   - Ask about recent legal developments
   - Request current legal information
   - Inquire about legal procedures

The web search integration significantly enhances the chatbot's capabilities while maintaining the existing local case database functionality! 🌐⚖️
