# optimized_reranker.py — Philippine Legal Optimized Reranker
import re
from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder


class PhilippineLegalReranker:
    """Optimized reranker for Philippine legal documents"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.legal_keywords = {
            # Constitutional Law
            "constitutional": 0.9, "constitution": 0.9, "bill of rights": 0.95,
            "due process": 0.9, "equal protection": 0.9, "freedom of speech": 0.9,
            
            # Civil Law
            "contract": 0.8, "obligation": 0.8, "damages": 0.8, "tort": 0.8,
            "property": 0.8, "ownership": 0.8, "possession": 0.8,
            "succession": 0.8, "inheritance": 0.8, "will": 0.8,
            
            # Criminal Law
            "criminal": 0.8, "crime": 0.8, "penalty": 0.8, "imprisonment": 0.8,
            "murder": 0.9, "homicide": 0.9, "theft": 0.8, "robbery": 0.8,
            "fraud": 0.8, "estafa": 0.8, "malversation": 0.8,
            
            # Administrative Law
            "administrative": 0.8, "government": 0.7, "public officer": 0.8,
            "civil service": 0.8, "discipline": 0.8, "dismissal": 0.8,
            
            # Labor Law
            "labor": 0.8, "employment": 0.8, "wage": 0.8, "benefits": 0.8,
            "termination": 0.8, "unfair labor practice": 0.9,
            
            # Commercial Law
            "corporation": 0.8, "partnership": 0.8, "bankruptcy": 0.8,
            "negotiable instrument": 0.8, "insurance": 0.8,
            
            # Family Law
            "marriage": 0.8, "divorce": 0.8, "annulment": 0.8,
            "custody": 0.8, "support": 0.8, "adoption": 0.8,
            
            # Special Laws
            "land reform": 0.9, "agrarian": 0.9, "eminent domain": 0.9,
            "expropriation": 0.9, "just compensation": 0.9,
            "environmental": 0.8, "pollution": 0.8, "natural resources": 0.8,
            
            # Legal Procedures
            "jurisdiction": 0.8, "venue": 0.7, "service of process": 0.8,
            "pleading": 0.7, "evidence": 0.8, "burden of proof": 0.8,
            "appeal": 0.7, "motion": 0.7, "judgment": 0.8,
            
            # Philippine Legal Terms
            "supreme court": 0.9, "court of appeals": 0.8, "regional trial court": 0.8,
            "municipal trial court": 0.7, "sandiganbayan": 0.8,
            "ombudsman": 0.8, "commission on audit": 0.8,
            "civil code": 0.9, "revised penal code": 0.9, "rules of court": 0.8,
        }
        
        self.case_patterns = [
            r"G\.R\.\s*No\.\s*\d+",  # G.R. No. format
            r"G\.R\.\s*Nos\.\s*\d+",  # G.R. Nos. format
            r"vs\.",  # vs. pattern
            r"petitioner",  # petitioner
            r"respondent",  # respondent
            r"plaintiff",  # plaintiff
            r"defendant",  # defendant
            r"appellant",  # appellant
            r"appellee",  # appellee
        ]
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               base_scores: List[float]) -> List[Tuple[Dict[str, Any], float]]:
        """Enhanced reranking with Philippine legal knowledge"""
        
        # Calculate enhanced scores
        enhanced_scores = []
        
        for doc, base_score in zip(documents, base_scores):
            # Get CrossEncoder score
            doc_text = self._extract_document_text(doc)
            cross_score_result = self.model.predict([query, doc_text])
            # Handle both scalar and array results
            cross_score = cross_score_result[0] if hasattr(cross_score_result, '__len__') and len(cross_score_result) > 0 else cross_score_result
            
            # Calculate legal relevance bonus
            legal_bonus = self._calculate_legal_relevance_bonus(query, doc_text)
            
            # Calculate case pattern bonus
            pattern_bonus = self._calculate_pattern_bonus(query, doc_text)
            
            # Calculate G.R. number bonus
            gr_bonus = self._calculate_gr_number_bonus(query, doc_text)
            
            # Combine scores with weights
            enhanced_score = (
                cross_score * 0.6 +           # CrossEncoder base score
                legal_bonus * 0.2 +           # Legal keyword relevance
                pattern_bonus * 0.1 +         # Case pattern matching
                gr_bonus * 0.1                # G.R. number matching
            )
            
            enhanced_scores.append((doc, enhanced_score))
        
        # Sort by enhanced score
        enhanced_scores.sort(key=lambda x: x[1], reverse=True)
        
        return enhanced_scores
    
    def _extract_document_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from document for reranking"""
        title = doc.get('title', '')
        content = doc.get('content', '')
        
        # Combine title and content
        text = f"{title}\n\n{content}"
        
        # Clean and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _calculate_legal_relevance_bonus(self, query: str, doc_text: str) -> float:
        """Calculate bonus based on legal keyword relevance"""
        query_lower = query.lower()
        doc_lower = doc_text.lower()
        
        bonus = 0.0
        for keyword, weight in self.legal_keywords.items():
            if keyword in query_lower and keyword in doc_lower:
                bonus += weight
        
        # Normalize bonus (max possible is around 20+ for very relevant cases)
        return min(bonus / 10.0, 1.0)
    
    def _calculate_pattern_bonus(self, query: str, doc_text: str) -> float:
        """Calculate bonus based on case pattern matching"""
        query_lower = query.lower()
        doc_lower = doc_text.lower()
        
        bonus = 0.0
        for pattern in self.case_patterns:
            if re.search(pattern, query_lower) and re.search(pattern, doc_lower):
                bonus += 0.1
        
        return min(bonus, 1.0)
    
    def _calculate_gr_number_bonus(self, query: str, doc_text: str) -> float:
        """Calculate bonus for G.R. number matching"""
        # Extract G.R. numbers from query
        query_gr_numbers = re.findall(r'G\.R\.\s*No\.?\s*(\d+)', query, re.IGNORECASE)
        doc_gr_numbers = re.findall(r'G\.R\.\s*No\.?\s*(\d+)', doc_text, re.IGNORECASE)
        
        if not query_gr_numbers or not doc_gr_numbers:
            return 0.0
        
        # Check for exact matches
        for query_gr in query_gr_numbers:
            if query_gr in doc_gr_numbers:
                return 1.0  # Perfect match bonus
        
        return 0.0
