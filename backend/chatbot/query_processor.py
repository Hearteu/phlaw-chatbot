# enhanced_query_processor.py — Advanced query processing with classification and reformulation
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class QueryType(Enum):
    """Types of legal queries"""
    CASE_DIGEST = "case_digest"
    FACT_QUERY = "fact_query"
    RULING_QUERY = "ruling_query"
    ISSUE_QUERY = "issue_query"
    ARGUMENT_QUERY = "argument_query"
    LEGAL_TERM_QUERY = "legal_term_query"
    CASE_SEARCH = "case_search"
    GENERAL_LEGAL = "general_legal"
    NON_LEGAL = "non_legal"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class QueryAnalysis:
    """Analysis result of a legal query"""
    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    confidence: float
    entities: Dict[str, List[str]]
    legal_terms: List[str]
    intent_flags: Dict[str, bool]
    reformulated_queries: List[str]
    suggested_filters: Dict[str, Any]
    context_requirements: List[str]

class EnhancedQueryProcessor:
    """Advanced query processing for legal document retrieval"""
    
    def __init__(self):
        self._init_query_patterns()
        self._init_legal_intent_keywords()
        self._init_complexity_indicators()
        self._init_query_reformulation_rules()
        self._init_philippine_legal_context()
    
    def _init_query_patterns(self):
        """Initialize query classification patterns"""
        self.query_patterns = {
            QueryType.CASE_DIGEST: [
                r'\b(?:digest|case digest|full digest|complete digest|comprehensive digest)\b',
                r'\b(?:summarize|summary|overview)\b.*\b(?:case|decision|ruling)\b',
                r'\b(?:give me|provide|show me)\b.*\b(?:digest|summary)\b',
            ],
            
            QueryType.FACT_QUERY: [
                r'\b(?:facts?|factual|what happened|background|story|narrative)\b',
                r'\b(?:tell me about|explain|describe)\b.*\b(?:facts?|what happened)\b',
                r'\b(?:who|what|when|where|why|how)\b.*\b(?:happened|occurred|did)\b',
            ],
            
            QueryType.RULING_QUERY: [
                r'\b(?:ruling|decision|held|court decided|judgment|verdict)\b',
                r'\b(?:what did the court decide|how was it decided|what was the ruling)\b',
                r'\b(?:wherefore|so ordered|disposition|dispositive)\b',
            ],
            
            QueryType.ISSUE_QUERY: [
                r'\b(?:issues?|legal questions?|questions? presented)\b',
                r'\b(?:what are the issues|what issues were raised|legal questions)\b',
                r'\b(?:whether|if|does|did|can|should|must)\b',
            ],
            
            QueryType.ARGUMENT_QUERY: [
                r'\b(?:arguments?|reasoning|legal reasoning|ratio decidendi)\b',
                r'\b(?:why did the court decide|court reasoning|legal arguments)\b',
                r'\b(?:basis|grounds|justification)\b.*\b(?:decision|ruling)\b',
            ],
            
            QueryType.LEGAL_TERM_QUERY: [
                r'\b(?:define|definition|meaning|what is|what does)\b',
                r'\b(?:legal term|doctrine|principle|concept)\b',
                r'\b(?:explain|clarify|elaborate)\b.*\b(?:term|concept|doctrine)\b',
            ],
            
            QueryType.CASE_SEARCH: [
                r'\b(?:G\.R\.|GR|case|decision)\b.*\b(?:No\.|number|#)\b',
                r'\b(?:find|search|look for|locate)\b.*\b(?:case|decision|ruling)\b',
                r'\b(?:case|decision|ruling)\b.*\b(?:v\.|vs\.|versus)\b',
            ],
            
            QueryType.GENERAL_LEGAL: [
                r'\b(?:legal|law|jurisprudence|court|supreme court)\b',
                r'\b(?:philippine|philippines|phil\.)\b.*\b(?:law|legal|court)\b',
                r'\b(?:constitutional|criminal|civil|administrative|labor)\b.*\b(?:law|right|case)\b',
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for query_type, patterns in self.query_patterns.items():
            self.compiled_patterns[query_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _init_legal_intent_keywords(self):
        """Initialize legal intent keywords"""
        self.intent_keywords = {
            'wants_digest': [
                'digest', 'case digest', 'full digest', 'complete digest', 'comprehensive digest',
                'summarize', 'summary', 'overview', 'give me', 'provide', 'show me'
            ],
            'wants_facts': [
                'facts', 'factual', 'what happened', 'background', 'story', 'narrative',
                'tell me about', 'explain', 'describe', 'who', 'what', 'when', 'where', 'why', 'how'
            ],
            'wants_ruling': [
                'ruling', 'decision', 'held', 'court decided', 'judgment', 'verdict',
                'what did the court decide', 'how was it decided', 'what was the ruling',
                'wherefore', 'so ordered', 'disposition', 'dispositive'
            ],
            'wants_issues': [
                'issues', 'legal questions', 'questions presented', 'what are the issues',
                'what issues were raised', 'legal questions', 'whether', 'if', 'does', 'did', 'can', 'should', 'must'
            ],
            'wants_arguments': [
                'arguments', 'reasoning', 'legal reasoning', 'ratio decidendi',
                'why did the court decide', 'court reasoning', 'legal arguments',
                'basis', 'grounds', 'justification'
            ],
            'wants_keywords': [
                'define', 'definition', 'meaning', 'what is', 'what does',
                'legal term', 'doctrine', 'principle', 'concept',
                'explain', 'clarify', 'elaborate'
            ],
            'wants_citations': [
                'cite', 'citation', 'authorities', 'legal authorities', 'case law',
                'precedent', 'jurisprudence', 'legal precedent'
            ]
        }
    
    def _init_complexity_indicators(self):
        """Initialize query complexity indicators"""
        self.complexity_indicators = {
            'simple': [
                'what', 'who', 'when', 'where', 'define', 'meaning',
                'facts', 'ruling', 'decision', 'case'
            ],
            'moderate': [
                'explain', 'describe', 'how', 'why', 'analyze', 'compare',
                'legal issues', 'court reasoning', 'legal arguments',
                'constitutional', 'criminal', 'civil', 'administrative'
            ],
            'complex': [
                'analyze the relationship', 'compare and contrast', 'evaluate the impact',
                'discuss the implications', 'examine the legal basis',
                'comprehensive analysis', 'detailed explanation', 'in-depth discussion',
                'legal doctrine', 'jurisprudential development', 'constitutional interpretation'
            ]
        }
    
    def _init_query_reformulation_rules(self):
        """Initialize query reformulation rules"""
        self.reformulation_rules = {
            # Expand abbreviations
            'abbreviations': {
                'G.R.': 'G.R. No.',
                'GR': 'G.R. No.',
                'SC': 'Supreme Court',
                'CA': 'Court of Appeals',
                'RTC': 'Regional Trial Court',
                'v.': 'versus',
                'vs.': 'versus'
            },
            
            # Add legal context
            'legal_context': {
                'case': 'Philippine Supreme Court case',
                'decision': 'Supreme Court decision',
                'ruling': 'Supreme Court ruling',
                'law': 'Philippine law',
                'court': 'Supreme Court'
            },
            
            # Expand legal terms
            'legal_terms': {
                'due process': 'due process constitutional right',
                'equal protection': 'equal protection constitutional right',
                'freedom of speech': 'freedom of speech constitutional right',
                'eminent domain': 'eminent domain expropriation',
                'land reform': 'land reform agrarian reform',
                'just compensation': 'just compensation fair compensation'
            }
        }
    
    def _init_philippine_legal_context(self):
        """Initialize Philippine legal context"""
        self.philippine_context = {
            'courts': [
                'Supreme Court', 'Court of Appeals', 'Regional Trial Court',
                'Municipal Trial Court', 'Sandiganbayan', 'Court of Tax Appeals'
            ],
            'legal_areas': [
                'Constitutional Law', 'Civil Law', 'Criminal Law', 'Administrative Law',
                'Labor Law', 'Commercial Law', 'Family Law', 'Special Laws'
            ],
            'legal_concepts': [
                'due process', 'equal protection', 'freedom of speech', 'bill of rights',
                'eminent domain', 'expropriation', 'just compensation', 'land reform',
                'separation of powers', 'checks and balances', 'judicial review'
            ]
        }
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> QueryAnalysis:
        """Process a legal query and return comprehensive analysis"""
        # Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Classify query type
        query_type, type_confidence = self._classify_query_type(cleaned_query)
        
        # Determine complexity
        complexity = self._determine_complexity(cleaned_query)
        
        # Extract entities
        entities = self._extract_entities(cleaned_query)
        
        # Extract legal terms
        legal_terms = self._extract_legal_terms(cleaned_query)
        
        # Determine intent flags
        intent_flags = self._determine_intent_flags(cleaned_query)
        
        # Reformulate queries
        reformulated_queries = self._reformulate_query(cleaned_query, query_type, entities)
        
        # Suggest filters
        suggested_filters = self._suggest_filters(query_type, entities, legal_terms)
        
        # Determine context requirements
        context_requirements = self._determine_context_requirements(query_type, complexity, conversation_history)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(query_type, type_confidence, entities, legal_terms)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            complexity=complexity,
            confidence=confidence,
            entities=entities,
            legal_terms=legal_terms,
            intent_flags=intent_flags,
            reformulated_queries=reformulated_queries,
            suggested_filters=suggested_filters,
            context_requirements=context_requirements
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove common query noise
        noise_patterns = [
            r'^(?:please|can you|could you|would you|will you)\s+',
            r'\s+(?:please|thanks|thank you|tia|ty)$',
            r'\s+(?:pls|plz|thx)\s*$'
        ]
        
        for pattern in noise_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        return query.strip()
    
    def _classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Classify the type of legal query"""
        query_lower = query.lower()
        scores = {}
        
        # Score each query type
        for query_type, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(query_lower):
                    score += 1
            scores[query_type] = score
        
        # Find the best match
        if not scores or max(scores.values()) == 0:
            # Check if it's a legal query at all
            if any(term in query_lower for term in ['legal', 'law', 'court', 'case', 'supreme']):
                return QueryType.GENERAL_LEGAL, 0.5
            else:
                return QueryType.NON_LEGAL, 0.1
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 3.0, 1.0)  # Normalize to 0-1
        
        return best_type, confidence
    
    def _determine_complexity(self, query: str) -> QueryComplexity:
        """Determine query complexity"""
        query_lower = query.lower()
        
        # Count complexity indicators
        simple_count = sum(1 for term in self.complexity_indicators['simple'] if term in query_lower)
        moderate_count = sum(1 for term in self.complexity_indicators['moderate'] if term in query_lower)
        complex_count = sum(1 for term in self.complexity_indicators['complex'] if term in query_lower)
        
        # Determine complexity based on indicators and query length
        if complex_count > 0 or len(query.split()) > 15:
            return QueryComplexity.COMPLEX
        elif moderate_count > 0 or len(query.split()) > 8:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {
            'gr_numbers': [],
            'case_names': [],
            'legal_terms': [],
            'years': [],
            'courts': [],
            'procedures': []
        }
        
        # Extract G.R. numbers
        gr_patterns = [
            r'G\.R\.\s*No\.?\s*(\d{6,7})',
            r'GR\s*No\.?\s*(\d{6,7})',
            r'(\d{6,7})'  # Fallback for 6-7 digit numbers
        ]
        
        for pattern in gr_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['gr_numbers'].extend(matches)
        
        # Extract case names
        case_pattern = r'([A-Z][A-Za-z\s&,\.]+?)\s+v\.?\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s*,\s*G\.R\.|\s*,\s*GR|\s*,\s*No\.|\s*,\s*\d{4}|\s*$)'
        case_matches = re.findall(case_pattern, query)
        for petitioner, respondent in case_matches:
            entities['case_names'].append(f"{petitioner} v. {respondent}")
        
        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        entities['years'] = re.findall(year_pattern, query)
        
        # Extract courts
        court_pattern = r'\b(?:Supreme Court|Court of Appeals|Regional Trial Court|Municipal Trial Court|Sandiganbayan|Court of Tax Appeals)\b'
        entities['courts'] = re.findall(court_pattern, query, re.IGNORECASE)
        
        # Extract legal terms
        entities['legal_terms'] = self._extract_legal_terms(query)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_legal_terms(self, query: str) -> List[str]:
        """Extract legal terms from query"""
        legal_terms = []
        query_lower = query.lower()
        
        # Check for legal concepts
        for concept in self.philippine_context['legal_concepts']:
            if concept in query_lower:
                legal_terms.append(concept)
        
        # Check for legal areas
        for area in self.philippine_context['legal_areas']:
            if area.lower() in query_lower:
                legal_terms.append(area)
        
        return legal_terms
    
    def _determine_intent_flags(self, query: str) -> Dict[str, bool]:
        """Determine intent flags for the query"""
        query_lower = query.lower()
        intent_flags = {}
        
        for intent, keywords in self.intent_keywords.items():
            intent_flags[intent] = any(keyword in query_lower for keyword in keywords)
        
        return intent_flags
    
    def _reformulate_query(self, query: str, query_type: QueryType, entities: Dict[str, List[str]]) -> List[str]:
        """Reformulate query for better retrieval"""
        reformulations = [query]  # Always include original
        
        # Apply abbreviation expansion
        expanded_query = query
        for abbr, expansion in self.reformulation_rules['abbreviations'].items():
            expanded_query = expanded_query.replace(abbr, expansion)
        if expanded_query != query:
            reformulations.append(expanded_query)
        
        # Add legal context
        contextual_query = query
        for term, context in self.reformulation_rules['legal_context'].items():
            if term in query.lower() and context not in query.lower():
                contextual_query += f" {context}"
        if contextual_query != query:
            reformulations.append(contextual_query)
        
        # Expand legal terms
        for term, expansion in self.reformulation_rules['legal_terms'].items():
            if term in query.lower():
                expanded_term_query = query.replace(term, expansion)
                reformulations.append(expanded_term_query)
        
        # Add Philippine context
        if not any(ph_term in query.lower() for ph_term in ['philippine', 'philippines', 'phil.']):
            philippine_query = f"Philippine {query}"
            reformulations.append(philippine_query)
        
        # Add query type specific reformulations
        if query_type == QueryType.CASE_DIGEST:
            reformulations.append(f"case digest {query}")
            reformulations.append(f"comprehensive digest {query}")
        elif query_type == QueryType.FACT_QUERY:
            reformulations.append(f"facts {query}")
            reformulations.append(f"factual background {query}")
        elif query_type == QueryType.RULING_QUERY:
            reformulations.append(f"ruling {query}")
            reformulations.append(f"decision {query}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_reformulations = []
        for reformulation in reformulations:
            if reformulation.lower() not in seen:
                seen.add(reformulation.lower())
                unique_reformulations.append(reformulation)
        
        return unique_reformulations[:5]  # Limit to 5 reformulations
    
    def _suggest_filters(self, query_type: QueryType, entities: Dict[str, List[str]], legal_terms: List[str]) -> Dict[str, Any]:
        """Suggest filters for query processing"""
        filters = {}
        
        # G.R. number filter
        if entities['gr_numbers']:
            filters['gr_numbers'] = entities['gr_numbers']
        
        # Case name filter
        if entities['case_names']:
            filters['case_names'] = entities['case_names']
        
        # Year filter
        if entities['years']:
            filters['years'] = entities['years']
        
        # Court filter
        if entities['courts']:
            filters['courts'] = entities['courts']
        
        # Legal area filter
        if legal_terms:
            filters['legal_areas'] = legal_terms
        
        # Query type specific filters
        if query_type == QueryType.CASE_DIGEST:
            filters['sections'] = ['facts', 'issues', 'ruling', 'arguments']
        elif query_type == QueryType.FACT_QUERY:
            filters['sections'] = ['facts']
        elif query_type == QueryType.RULING_QUERY:
            filters['sections'] = ['ruling']
        elif query_type == QueryType.ISSUE_QUERY:
            filters['sections'] = ['issues']
        elif query_type == QueryType.ARGUMENT_QUERY:
            filters['sections'] = ['arguments']
        
        return filters
    
    def _determine_context_requirements(self, query_type: QueryType, complexity: QueryComplexity, 
                                      conversation_history: Optional[List[Dict]]) -> List[str]:
        """Determine context requirements for the query"""
        requirements = []
        
        # Basic context requirements
        if query_type in [QueryType.CASE_DIGEST, QueryType.FACT_QUERY, QueryType.RULING_QUERY]:
            requirements.append('case_metadata')
            requirements.append('legal_sections')
        
        if query_type == QueryType.CASE_DIGEST:
            requirements.append('comprehensive_content')
            requirements.append('case_structure')
        
        if complexity == QueryComplexity.COMPLEX:
            requirements.append('detailed_analysis')
            requirements.append('legal_citations')
        
        # Conversation context requirements
        if conversation_history and len(conversation_history) > 0:
            requirements.append('conversation_context')
            requirements.append('previous_entities')
        
        return requirements
    
    def _calculate_confidence(self, query_type: QueryType, type_confidence: float, 
                            entities: Dict[str, List[str]], legal_terms: List[str]) -> float:
        """Calculate overall confidence for the query analysis"""
        confidence = type_confidence
        
        # Boost confidence for entity matches
        entity_boost = 0
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_boost += 0.1
        
        # Boost confidence for legal terms
        if legal_terms:
            confidence += 0.1
        
        # Boost confidence for specific query types
        if query_type in [QueryType.CASE_DIGEST, QueryType.CASE_SEARCH]:
            confidence += 0.1
        
        confidence += entity_boost
        
        return min(confidence, 1.0)
    
    def get_query_statistics(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get statistics about the query analysis"""
        return {
            'query_type': analysis.query_type.value,
            'complexity': analysis.complexity.value,
            'confidence': analysis.confidence,
            'entity_count': sum(len(entities) for entities in analysis.entities.values()),
            'legal_term_count': len(analysis.legal_terms),
            'intent_flag_count': sum(analysis.intent_flags.values()),
            'reformulation_count': len(analysis.reformulated_queries),
            'filter_count': len(analysis.suggested_filters),
            'context_requirement_count': len(analysis.context_requirements)
        }
