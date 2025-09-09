#!/usr/bin/env python3
"""
LLM-based Intent Processor for Legal Chatbot
Analyzes user queries to determine intent, complexity, and routing decisions
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class QueryIntent(Enum):
    """Query intent categories"""
    CASE_SEARCH = "case_search"           # Looking for specific cases
    LEGAL_TERM_DEFINITION = "legal_term"  # Define legal terms/concepts
    CASE_DIGEST = "case_digest"           # Full case analysis
    RULING_QUERY = "ruling_query"         # Specific ruling/disposition
    FACTS_QUERY = "facts_query"           # Case facts
    ISSUES_QUERY = "issues_query"         # Legal issues
    ARGUMENTS_QUERY = "arguments_query"   # Legal arguments
    GENERAL_LEGAL = "general_legal"       # General legal question
    NON_LEGAL = "non_legal"               # Non-legal question

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"      # Single concept, clear intent
    MODERATE = "moderate"  # Multiple concepts, some ambiguity
    COMPLEX = "complex"    # Multiple concepts, high ambiguity

@dataclass
class IntentAnalysis:
    """Result of intent analysis"""
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    confidence: float
    entities: Dict[str, List[str]]
    legal_terms: List[str]
    reformulated_queries: List[str]
    routing_decision: str
    context_requirements: List[str]
    suggested_filters: Dict[str, Any]

class LLMIntentProcessor:
    """LLM-based intent processor for legal queries"""
    
    def __init__(self):
        self.intent_keywords = {
            QueryIntent.CASE_SEARCH: [
                'case', 'decision', 'ruling', 'supreme court', 'court of appeals',
                'g.r. no', 'gr no', 'v.', 'vs', 'versus', 'petitioner', 'respondent'
            ],
            QueryIntent.LEGAL_TERM_DEFINITION: [
                'define', 'definition', 'meaning', 'what is', 'explain',
                'doctrine', 'principle', 'concept', 'term'
            ],
            QueryIntent.CASE_DIGEST: [
                'digest', 'summary', 'overview', 'full case', 'complete case',
                'case analysis', 'comprehensive'
            ],
            QueryIntent.RULING_QUERY: [
                'ruling', 'disposition', 'wherefore', 'so ordered', 'decision',
                'held', 'court held'
            ],
            QueryIntent.FACTS_QUERY: [
                'facts', 'factual', 'what happened', 'background', 'antecedent'
            ],
            QueryIntent.ISSUES_QUERY: [
                'issues', 'legal issues', 'questions', 'whether', 'problem'
            ],
            QueryIntent.ARGUMENTS_QUERY: [
                'arguments', 'reasoning', 'rationale', 'analysis', 'discussion'
            ]
        }
    
    def analyze_intent(self, query: str, history: List[Dict] = None) -> IntentAnalysis:
        """Analyze query intent using LLM and rule-based fallbacks"""
        if history is None:
            history = []
        
        # Extract context from history
        context = self._extract_context(history)
        
        # Rule-based analysis first (fast)
        rule_based_intent = self._rule_based_analysis(query, context)
        
        # LLM-based analysis for complex cases
        if rule_based_intent.complexity == QueryComplexity.COMPLEX or rule_based_intent.confidence < 0.7:
            llm_intent = self._llm_analysis(query, context, history)
            # Combine results, preferring LLM for complex cases
            final_intent = self._combine_analyses(rule_based_intent, llm_intent)
        else:
            final_intent = rule_based_intent
        
        # Determine routing decision
        routing_decision = self._determine_routing(final_intent, context)
        
        return IntentAnalysis(
            original_query=query,
            intent=final_intent.intent,
            complexity=final_intent.complexity,
            confidence=final_intent.confidence,
            entities=final_intent.entities,
            legal_terms=final_intent.legal_terms,
            reformulated_queries=final_intent.reformulated_queries,
            routing_decision=routing_decision,
            context_requirements=final_intent.context_requirements,
            suggested_filters=final_intent.suggested_filters
        )
    
    def _rule_based_analysis(self, query: str, context: str) -> IntentAnalysis:
        """Rule-based intent analysis as fallback"""
        query_lower = query.lower()
        
        
        # Check for case search indicators
        case_indicators = ['g.r. no', 'gr no', 'v.', 'vs', 'versus', 'case', 'decision']
        if any(indicator in query_lower for indicator in case_indicators):
            return IntentAnalysis(
                original_query=query,
                intent=QueryIntent.CASE_SEARCH,
                complexity=QueryComplexity.MODERATE,
                confidence=0.8,
                entities=self._extract_entities(query),
                legal_terms=self._extract_legal_terms(query),
                reformulated_queries=[query],
                routing_decision="jurisprudence",
                context_requirements=[],
                suggested_filters={}
            )
        
        # Check for legal term definition
        definition_indicators = ['define', 'definition', 'meaning', 'what is', 'explain']
        if any(indicator in query_lower for indicator in definition_indicators):
            return IntentAnalysis(
                original_query=query,
                intent=QueryIntent.LEGAL_TERM_DEFINITION,
                complexity=QueryComplexity.SIMPLE,
                confidence=0.8,
                entities=self._extract_entities(query),
                legal_terms=self._extract_legal_terms(query),
                reformulated_queries=[query],
                routing_decision="rule_based",
                context_requirements=[],
                suggested_filters={}
            )
        
        # Default to general legal
        return IntentAnalysis(
            original_query=query,
            intent=QueryIntent.GENERAL_LEGAL,
            complexity=QueryComplexity.MODERATE,
            confidence=0.5,
            entities=self._extract_entities(query),
            legal_terms=self._extract_legal_terms(query),
            reformulated_queries=[query],
            routing_decision="jurisprudence",
            context_requirements=[],
            suggested_filters={}
        )
    
    def _llm_analysis(self, query: str, context: str, history: List[Dict]) -> IntentAnalysis:
        """LLM-based intent analysis for complex queries"""
        try:
            from .generator import generate_response_from_messages

            # Create prompt for intent analysis
            prompt = self._create_intent_prompt(query, context, history)
            
            messages = [
                {"role": "system", "content": "You are a legal query intent analyzer. Analyze the user's query and return a JSON response with intent classification."},
                {"role": "user", "content": prompt}
            ]
            
            response = generate_response_from_messages(messages)
            
            # Parse LLM response
            return self._parse_llm_response(query, response)
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            # Fallback to rule-based
            return self._rule_based_analysis(query, context)
    
    def _create_intent_prompt(self, query: str, context: str, history: List[Dict]) -> str:
        """Create prompt for LLM intent analysis"""
        history_text = ""
        if history:
            history_text = "Previous conversation:\n"
            for msg in history[-3:]:  # Last 3 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role}: {content}\n"
        
        return f"""
Analyze this legal query and return a JSON response with the following structure:

{{
    "intent": "case_search|legal_term|case_digest|ruling_query|facts_query|issues_query|arguments_query|general_legal|non_legal",
    "complexity": "simple|moderate|complex",
    "confidence": 0.0-1.0,
    "entities": {{
        "gr_numbers": ["G.R. No. 123456"],
        "case_names": ["People vs. Smith"],
        "persons": ["John Doe"],
        "dates": ["2024-01-01"]
    }},
    "legal_terms": ["doctrine", "principle"],
    "reformulated_queries": ["alternative query 1", "alternative query 2"],
    "context_requirements": ["needs case context", "needs legal background"],
    "suggested_filters": {{
        "year": 2024,
        "case_type": "criminal",
        "ponente": "Ponente Name"
    }}
}}

Query: {query}
{history_text}
Context: {context}

Return only the JSON response:
"""
    
    def _parse_llm_response(self, query: str, response: str) -> IntentAnalysis:
        """Parse LLM response into IntentAnalysis"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Map intent string to enum
            intent_map = {
                'case_search': QueryIntent.CASE_SEARCH,
                'legal_term': QueryIntent.LEGAL_TERM_DEFINITION,
                'case_digest': QueryIntent.CASE_DIGEST,
                'ruling_query': QueryIntent.RULING_QUERY,
                'facts_query': QueryIntent.FACTS_QUERY,
                'issues_query': QueryIntent.ISSUES_QUERY,
                'arguments_query': QueryIntent.ARGUMENTS_QUERY,
                'general_legal': QueryIntent.GENERAL_LEGAL,
                'non_legal': QueryIntent.NON_LEGAL
            }
            
            # Map complexity string to enum
            complexity_map = {
                'simple': QueryComplexity.SIMPLE,
                'moderate': QueryComplexity.MODERATE,
                'complex': QueryComplexity.COMPLEX
            }
            
            return IntentAnalysis(
                original_query=query,
                intent=intent_map.get(data.get('intent', 'general_legal'), QueryIntent.GENERAL_LEGAL),
                complexity=complexity_map.get(data.get('complexity', 'moderate'), QueryComplexity.MODERATE),
                confidence=float(data.get('confidence', 0.5)),
                entities=data.get('entities', {}),
                legal_terms=data.get('legal_terms', []),
                reformulated_queries=data.get('reformulated_queries', [query]),
                routing_decision="",  # Will be determined later
                context_requirements=data.get('context_requirements', []),
                suggested_filters=data.get('suggested_filters', {})
            )
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Fallback to rule-based
            return self._rule_based_analysis(query, "")
    
    def _combine_analyses(self, rule_based: IntentAnalysis, llm: IntentAnalysis) -> IntentAnalysis:
        """Combine rule-based and LLM analyses"""
        # Prefer LLM for complex cases, rule-based for simple cases
        if rule_based.complexity == QueryComplexity.SIMPLE and rule_based.confidence > 0.8:
            return rule_based
        else:
            return llm
    
    def _determine_routing(self, intent_analysis: IntentAnalysis, context: str) -> str:
        """Determine routing decision based on intent analysis"""
        if intent_analysis.intent == QueryIntent.LEGAL_TERM_DEFINITION:
            return "rule_based"
        elif intent_analysis.intent in [QueryIntent.CASE_SEARCH, QueryIntent.CASE_DIGEST, 
                                       QueryIntent.RULING_QUERY, QueryIntent.FACTS_QUERY,
                                       QueryIntent.ISSUES_QUERY, QueryIntent.ARGUMENTS_QUERY]:
            return "jurisprudence"
        elif intent_analysis.intent == QueryIntent.GENERAL_LEGAL:
            return "jurisprudence"  # Try jurisprudence first, fallback to rule-based
        else:
            return "rule_based"
    
    def _extract_context(self, history: List[Dict]) -> str:
        """Extract relevant context from conversation history"""
        if not history:
            return ""
        
        context_parts = []
        for msg in history[-3:]:  # Last 3 messages
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user' and content:
                context_parts.append(content)
        
        return " ".join(context_parts)
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query using simple regex patterns"""
        entities = {
            'gr_numbers': [],
            'case_names': [],
            'persons': [],
            'dates': []
        }
        
        # G.R. numbers
        gr_pattern = r'G\.R\.\s+No\.?\s*\d+'
        entities['gr_numbers'] = re.findall(gr_pattern, query, re.IGNORECASE)
        
        # Case names (simple pattern)
        case_pattern = r'[A-Z][a-z]+\s+(?:v\.|vs\.?|versus)\s+[A-Z][a-z]+'
        entities['case_names'] = re.findall(case_pattern, query)
        
        # Dates
        date_pattern = r'\b(?:19|20)\d{2}\b'
        entities['dates'] = re.findall(date_pattern, query)
        
        return entities
    
    def _extract_legal_terms(self, query: str) -> List[str]:
        """Extract legal terms from query"""
        legal_terms = []
        
        # Common legal terms
        legal_patterns = [
            r'\b(?:doctrine|principle|concept|rule|law|statute|code)\b',
            r'\b(?:constitutional|criminal|civil|administrative)\b',
            r'\b(?:supreme court|court of appeals|trial court)\b',
            r'\b(?:petitioner|respondent|plaintiff|defendant)\b'
        ]
        
        for pattern in legal_patterns:
            terms = re.findall(pattern, query, re.IGNORECASE)
            legal_terms.extend(terms)
        
        return list(set(legal_terms))  # Remove duplicates

# Global instance
intent_processor = LLMIntentProcessor()
