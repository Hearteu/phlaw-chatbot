# web_search.py - Web search integration for legal information
import json
import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse

import requests

logger = logging.getLogger(__name__)

class LegalWebSearcher:
    """Web search service for legal information using Google Custom Search API"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google Custom Search API credentials not configured")
            self.available = False
        else:
            self.available = True
            logger.info("Legal Web Searcher initialized with Google Custom Search API")
        
        # Rate limiting
        self.last_search_time = 0
        self.min_search_interval = 1  # seconds between searches
        
        # Legal-focused search sites for Philippine law
        self.legal_sources = [
            "site:elibrary.judiciary.gov.ph",
            "site:sc.judiciary.gov.ph", 
            "site:lawphil.net",
            "site:chanrobles.com",
            "site:supremecourt.gov.ph",
            "site:officialgazette.gov.ph",
            "site:lawphil.net"
        ]
    
    def search_legal_content(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search for legal content on the web with focus on Philippine legal sources"""
        
        if not self.available:
            logger.warning("Web search not available - API credentials missing")
            return []
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_search_time < self.min_search_interval:
            time.sleep(self.min_search_interval - (current_time - self.last_search_time))
        
        try:
            # Enhance query with legal context and Philippine law focus
            enhanced_query = self._enhance_legal_query(query)
            
            # Perform search
            results = self._perform_search(enhanced_query, num_results)
            
            # Process and filter results
            processed_results = self._process_search_results(results)
            
            self.last_search_time = time.time()
            logger.info(f"Web search completed for query: '{query}' - Found {len(processed_results)} results")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Web search error for query '{query}': {e}")
            return []
    
    def _enhance_legal_query(self, query: str) -> str:
        """Enhance query with legal context and Philippine law focus"""
        
        # Add Philippine legal context
        philippine_context = "Philippines"
        
        # For general queries, use broader search
        # For specific queries, add legal context
        if any(word in query.lower() for word in ['constitution', 'law', 'court', 'legal', 'statute']):
            enhanced_query = f"{query} {philippine_context} law"
        else:
            enhanced_query = f"{query} {philippine_context}"
        
        return enhanced_query
    
    def _perform_search(self, query: str, num_results: int) -> Dict:
        """Perform the actual Google Custom Search API call"""
        
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10),  # API limit is 10
            'safe': 'medium',
            'lr': 'lang_en',  # English language
            'sort': 'date'  # Sort by date for current information
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Custom Search API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Google Custom Search API response: {e}")
            raise
    
    def _process_search_results(self, raw_results: Dict) -> List[Dict]:
        """Process and clean web search results"""
        processed = []
        
        search_info = raw_results.get('searchInformation', {})
        total_results = search_info.get('totalResults', '0')
        logger.info(f"Google search found {total_results} total results")
        
        for item in raw_results.get('items', []):
            try:
                processed_item = {
                    'title': self._clean_text(item.get('title', '')),
                    'snippet': self._clean_text(item.get('snippet', '')),
                    'url': item.get('link', ''),
                    'source': self._extract_source_domain(item.get('link', '')),
                    'date': item.get('formattedUrl', ''),
                    'relevance_score': self._calculate_relevance_score(item, raw_results.get('queries', {}))
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        # Sort by relevance score
        processed.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove HTML entities if any
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        
        return text.strip()
    
    def _extract_source_domain(self, url: str) -> str:
        """Extract domain name from URL for source attribution"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Clean up domain name
            domain = domain.replace('www.', '')
            
            # Map to more readable names
            domain_mapping = {
                'elibrary.judiciary.gov.ph': 'Supreme Court E-Library',
                'sc.judiciary.gov.ph': 'Supreme Court',
                'lawphil.net': 'LawPhil',
                'chanrobles.com': 'ChanRobles',
                'supremecourt.gov.ph': 'Supreme Court Official',
                'officialgazette.gov.ph': 'Official Gazette'
            }
            
            return domain_mapping.get(domain, domain)
            
        except Exception:
            return "Unknown Source"
    
    def _calculate_relevance_score(self, item: Dict, queries: Dict) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        
        # Base score
        score += 10.0
        
        # Legal source bonus
        url = item.get('link', '').lower()
        for legal_source in ['elibrary.judiciary.gov.ph', 'sc.judiciary.gov.ph', 'lawphil.net']:
            if legal_source in url:
                score += 20.0
                break
        
        # Government source bonus
        if any(gov_domain in url for gov_domain in ['.gov.ph', '.judiciary.gov.ph']):
            score += 15.0
        
        # Title relevance
        title = item.get('title', '').lower()
        if any(keyword in title for keyword in ['supreme court', 'decision', 'ruling', 'case']):
            score += 10.0
        
        return score
    
    def search_recent_decisions(self, topic: str = "", limit: int = 5) -> List[Dict]:
        """Search specifically for recent Supreme Court decisions"""
        
        query = f"recent Supreme Court decisions {topic} Philippines 2024"
        return self.search_legal_content(query, num_results=limit)
    
    def search_constitutional_provisions(self, provision: str) -> List[Dict]:
        """Search for constitutional provisions and interpretations"""
        
        query = f"Philippine Constitution {provision} Article interpretation"
        return self.search_legal_content(query, num_results=5)
    
    def search_statutory_law(self, law_name: str) -> List[Dict]:
        """Search for statutory laws and recent amendments"""
        
        query = f"{law_name} Philippines law statute recent amendments"
        return self.search_legal_content(query, num_results=5)
    
    def is_available(self) -> bool:
        """Check if web search is available"""
        return self.available

# Global instance
_web_searcher = None

def get_web_searcher() -> LegalWebSearcher:
    """Get or create global web searcher instance"""
    global _web_searcher
    if _web_searcher is None:
        _web_searcher = LegalWebSearcher()
    return _web_searcher

def test_web_search():
    """Test function for web search functionality"""
    searcher = get_web_searcher()
    
    if not searcher.is_available():
        print("❌ Web search not available - check API credentials")
        return
    
    print("🔍 Testing web search...")
    
    # Test basic search
    results = searcher.search_legal_content("Supreme Court labor law decisions", num_results=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()

if __name__ == "__main__":
    test_web_search()
