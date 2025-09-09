# enhanced_entity_extractor.py — Advanced legal entity extraction with NER and Philippine legal knowledge
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy


@dataclass
class LegalEntity:
    """Represents a legal entity with confidence and context"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    normalized: str = ""

class PhilippineLegalEntityExtractor:
    """Enhanced entity extractor for Philippine legal documents"""
    
    def __init__(self):
        self.nlp = None
        self._init_spacy_model()
        self._init_legal_patterns()
        self._init_philippine_legal_terms()
        self._init_court_hierarchy()
        self._init_legal_abbreviations()
    
    def _init_spacy_model(self):
        """Initialize spaCy model for NER"""
        try:
            # Try to load the best available model
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy model: en_core_web_sm")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_lg")
                print("✅ Loaded spaCy model: en_core_web_lg")
            except OSError:
                print("⚠️ No spaCy model found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def _init_legal_patterns(self):
        """Initialize legal-specific regex patterns"""
        self.patterns = {
            # G.R. Numbers
            'gr_number': [
                r'G\.R\.\s*No\.?\s*(\d{6,7})',
                r'GR\s*No\.?\s*(\d{6,7})',
                r'G\.R\.\s*Nos\.?\s*(\d{6,7}(?:[,\s]+and\s+)?\d{6,7})',
                r'(\d{6,7})',  # Fallback for 6-7 digit numbers
            ],
            
            # Case Names (Petitioner v. Respondent)
            'case_name': [
                r'([A-Z][A-Za-z\s&,\.]+?)\s+v\.?\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s*,\s*G\.R\.|\s*,\s*GR|\s*,\s*No\.|\s*,\s*\d{4}|\s*$)',
                r'([A-Z][A-Za-z\s&,\.]+?)\s+vs\.?\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s*,\s*G\.R\.|\s*,\s*GR|\s*,\s*No\.|\s*,\s*\d{4}|\s*$)',
                r'([A-Z][A-Za-z\s&,\.]+?)\s+versus\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s*,\s*G\.R\.|\s*,\s*GR|\s*,\s*No\.|\s*,\s*\d{4}|\s*$)',
            ],
            
            # Legal Citations
            'legal_citation': [
                r'(\d+\s+Phil\.\s+\d+)',
                r'(\d+\s+SCRA\s+\d+)',
                r'(\d+\s+OG\s+\d+)',
                r'(\d+\s+O\.G\.\s+\d+)',
                r'(\d+\s+Supp\.\s+\d+)',
            ],
            
            # Dates
            'date': [
                r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4})',
                r'(\d{4})',  # Year only
            ],
            
            # Legal Terms
            'legal_term': [
                r'\b(?:due process|equal protection|freedom of speech|bill of rights)\b',
                r'\b(?:eminent domain|expropriation|just compensation|land reform)\b',
                r'\b(?:writ of habeas corpus|writ of certiorari|writ of mandamus)\b',
                r'\b(?:res judicata|stare decisis|ratio decidendi|obiter dictum)\b',
            ],
            
            # Court References
            'court': [
                r'\b(?:Supreme Court|Court of Appeals|Regional Trial Court|Municipal Trial Court)\b',
                r'\b(?:Sandiganbayan|Court of Tax Appeals|Shari\'a District Court)\b',
                r'\b(?:En Banc|First Division|Second Division|Third Division)\b',
            ],
            
            # Legal Procedures
            'procedure': [
                r'\b(?:petition|motion|appeal|reconsideration|new trial)\b',
                r'\b(?:injunction|temporary restraining order|preliminary injunction)\b',
                r'\b(?:summary judgment|default judgment|consent judgment)\b',
            ]
        }
    
    def _init_philippine_legal_terms(self):
        """Initialize Philippine legal terminology database"""
        self.philippine_legal_terms = {
            # Constitutional Law
            'constitutional': {
                'terms': ['due process', 'equal protection', 'freedom of speech', 'bill of rights', 
                         'separation of powers', 'checks and balances', 'judicial review'],
                'weight': 0.9
            },
            
            # Civil Law
            'civil': {
                'terms': ['obligation', 'contract', 'tort', 'damages', 'property', 'ownership', 
                         'possession', 'succession', 'inheritance', 'marriage', 'divorce'],
                'weight': 0.8
            },
            
            # Criminal Law
            'criminal': {
                'terms': ['murder', 'homicide', 'theft', 'robbery', 'fraud', 'estafa', 
                         'malversation', 'bribery', 'corruption', 'penalty', 'imprisonment'],
                'weight': 0.8
            },
            
            # Administrative Law
            'administrative': {
                'terms': ['civil service', 'government', 'public officer', 'discipline', 
                         'dismissal', 'suspension', 'administrative case', 'ombudsman'],
                'weight': 0.8
            },
            
            # Labor Law
            'labor': {
                'terms': ['employment', 'wage', 'benefits', 'termination', 'unfair labor practice', 
                         'collective bargaining', 'strike', 'lockout', 'labor union'],
                'weight': 0.8
            },
            
            # Commercial Law
            'commercial': {
                'terms': ['corporation', 'partnership', 'bankruptcy', 'negotiable instrument', 
                         'insurance', 'securities', 'business', 'commerce'],
                'weight': 0.8
            },
            
            # Special Laws
            'special': {
                'terms': ['land reform', 'agrarian reform', 'eminent domain', 'expropriation', 
                         'environmental', 'pollution', 'natural resources', 'indigenous peoples'],
                'weight': 0.9
            }
        }
    
    def _init_court_hierarchy(self):
        """Initialize Philippine court hierarchy"""
        self.court_hierarchy = {
            'supreme_court': {
                'names': ['Supreme Court', 'SC', 'High Court'],
                'weight': 1.0,
                'abbreviations': ['SC']
            },
            'court_of_appeals': {
                'names': ['Court of Appeals', 'CA', 'Appellate Court'],
                'weight': 0.9,
                'abbreviations': ['CA']
            },
            'regional_trial_court': {
                'names': ['Regional Trial Court', 'RTC', 'Trial Court'],
                'weight': 0.8,
                'abbreviations': ['RTC']
            },
            'municipal_trial_court': {
                'names': ['Municipal Trial Court', 'MTC', 'Municipal Court'],
                'weight': 0.7,
                'abbreviations': ['MTC']
            },
            'sandiganbayan': {
                'names': ['Sandiganbayan', 'Anti-Graft Court'],
                'weight': 0.9,
                'abbreviations': ['SB']
            },
            'court_of_tax_appeals': {
                'names': ['Court of Tax Appeals', 'CTA', 'Tax Court'],
                'weight': 0.8,
                'abbreviations': ['CTA']
            }
        }
    
    def _init_legal_abbreviations(self):
        """Initialize legal abbreviations and their expansions"""
        self.legal_abbreviations = {
            'G.R.': 'G.R. No.',
            'GR': 'G.R. No.',
            'SC': 'Supreme Court',
            'CA': 'Court of Appeals',
            'RTC': 'Regional Trial Court',
            'MTC': 'Municipal Trial Court',
            'SB': 'Sandiganbayan',
            'CTA': 'Court of Tax Appeals',
            'Phil.': 'Philippine Reports',
            'SCRA': 'Supreme Court Reports Annotated',
            'OG': 'Official Gazette',
            'O.G.': 'Official Gazette',
            'Supp.': 'Supplement',
            'v.': 'versus',
            'vs.': 'versus',
            'versus': 'versus',
            'petitioner': 'Petitioner',
            'respondent': 'Respondent',
            'plaintiff': 'Plaintiff',
            'defendant': 'Defendant',
            'appellant': 'Appellant',
            'appellee': 'Appellee'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[LegalEntity]]:
        """Extract all legal entities from text"""
        entities = defaultdict(list)
        
        # Extract using regex patterns
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    entity = self._create_entity(match, entity_type, text)
                    if entity:
                        entities[entity_type].append(entity)
        
        # Extract using spaCy NER
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(text)
            for entity_type, entity_list in spacy_entities.items():
                entities[entity_type].extend(entity_list)
        
        # Extract legal terms with context
        legal_terms = self._extract_legal_terms_with_context(text)
        entities['legal_term'].extend(legal_terms)
        
        # Deduplicate and rank entities
        for entity_type in entities:
            entities[entity_type] = self._deduplicate_entities(entities[entity_type])
            entities[entity_type].sort(key=lambda x: x.confidence, reverse=True)
        
        return dict(entities)
    
    def _create_entity(self, match: re.Match, entity_type: str, text: str) -> Optional[LegalEntity]:
        """Create a LegalEntity from a regex match"""
        try:
            full_match = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract context (50 chars before and after)
            context_start = max(0, start_pos - 50)
            context_end = min(len(text), end_pos + 50)
            context = text[context_start:context_end]
            
            # Calculate confidence based on pattern specificity
            confidence = self._calculate_pattern_confidence(entity_type, full_match)
            
            # Normalize the entity
            normalized = self._normalize_entity(entity_type, full_match)
            
            return LegalEntity(
                text=full_match,
                entity_type=entity_type,
                confidence=confidence,
                start_pos=start_pos,
                end_pos=end_pos,
                context=context,
                normalized=normalized
            )
        except Exception as e:
            print(f"Error creating entity: {e}")
            return None
    
    def _extract_spacy_entities(self, text: str) -> Dict[str, List[LegalEntity]]:
        """Extract entities using spaCy NER"""
        entities = defaultdict(list)
        
        if not self.nlp:
            return entities
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Map spaCy entities to our legal entity types
                legal_type = self._map_spacy_to_legal_type(ent.label_)
                if legal_type:
                    entity = LegalEntity(
                        text=ent.text,
                        entity_type=legal_type,
                        confidence=0.7,  # Default confidence for spaCy
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        context=text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)],
                        normalized=ent.text
                    )
                    entities[legal_type].append(entity)
        except Exception as e:
            print(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def _map_spacy_to_legal_type(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to legal entity types"""
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'DATE': 'date',
            'LAW': 'legal_term',
            'CASE_NUMBER': 'gr_number'
        }
        return mapping.get(spacy_label)
    
    def _extract_legal_terms_with_context(self, text: str) -> List[LegalEntity]:
        """Extract legal terms with enhanced context analysis"""
        legal_entities = []
        text_lower = text.lower()
        
        for category, data in self.philippine_legal_terms.items():
            for term in data['terms']:
                if term in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(term, start)
                        if pos == -1:
                            break
                        
                        # Extract context
                        context_start = max(0, pos - 50)
                        context_end = min(len(text), pos + len(term) + 50)
                        context = text[context_start:context_end]
                        
                        # Calculate confidence based on context
                        confidence = self._calculate_legal_term_confidence(term, context, data['weight'])
                        
                        entity = LegalEntity(
                            text=term,
                            entity_type='legal_term',
                            confidence=confidence,
                            start_pos=pos,
                            end_pos=pos + len(term),
                            context=context,
                            normalized=term
                        )
                        legal_entities.append(entity)
                        start = pos + 1
        
        return legal_entities
    
    def _calculate_pattern_confidence(self, entity_type: str, text: str) -> float:
        """Calculate confidence based on pattern specificity"""
        base_confidence = {
            'gr_number': 0.9,
            'case_name': 0.8,
            'legal_citation': 0.9,
            'date': 0.7,
            'legal_term': 0.6,
            'court': 0.8,
            'procedure': 0.7
        }
        
        confidence = base_confidence.get(entity_type, 0.5)
        
        # Boost confidence for specific patterns
        if entity_type == 'gr_number' and 'G.R.' in text:
            confidence += 0.1
        elif entity_type == 'case_name' and ' v. ' in text:
            confidence += 0.1
        elif entity_type == 'legal_citation' and any(abbr in text for abbr in ['Phil.', 'SCRA', 'OG']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_legal_term_confidence(self, term: str, context: str, base_weight: float) -> float:
        """Calculate confidence for legal terms based on context"""
        confidence = base_weight
        
        # Boost confidence for legal context indicators
        legal_indicators = ['court', 'case', 'law', 'legal', 'supreme', 'decision', 'ruling']
        context_lower = context.lower()
        
        for indicator in legal_indicators:
            if indicator in context_lower:
                confidence += 0.1
        
        # Boost confidence for proximity to case references
        if any(ref in context_lower for ref in ['g.r.', 'gr no', 'case', 'petitioner', 'respondent']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _normalize_entity(self, entity_type: str, text: str) -> str:
        """Normalize entity text"""
        if entity_type == 'gr_number':
            # Normalize G.R. number format
            numbers = re.findall(r'\d{6,7}', text)
            if numbers:
                return f"G.R. No. {numbers[0]}"
        
        elif entity_type == 'case_name':
            # Normalize case name format
            if ' v. ' in text or ' vs. ' in text or ' versus ' in text:
                return text.replace(' vs. ', ' v. ').replace(' versus ', ' v. ')
        
        elif entity_type == 'court':
            # Normalize court names
            text_lower = text.lower()
            for court_type, data in self.court_hierarchy.items():
                for name in data['names']:
                    if name.lower() in text_lower:
                        return name
        
        # Apply abbreviation expansion
        normalized = text
        for abbr, expansion in self.legal_abbreviations.items():
            if abbr in normalized:
                normalized = normalized.replace(abbr, expansion)
        
        return normalized
    
    def _deduplicate_entities(self, entities: List[LegalEntity]) -> List[LegalEntity]:
        """Remove duplicate entities, keeping the highest confidence ones"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key based on normalized text and position
            key = (entity.normalized.lower(), entity.start_pos, entity.end_pos)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_entities_from_query(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from a query and return simplified format for retrieval"""
        entities = self.extract_entities(query)
        
        # Convert to simplified format
        simplified = {
            'gr_numbers': [e.normalized for e in entities.get('gr_number', [])],
            'case_names': [e.normalized for e in entities.get('case_name', [])],
            'legal_terms': [e.normalized for e in entities.get('legal_term', [])],
            'years': [e.normalized for e in entities.get('date', [])],
            'courts': [e.normalized for e in entities.get('court', [])],
            'procedures': [e.normalized for e in entities.get('procedure', [])]
        }
        
        # Remove duplicates
        for key in simplified:
            simplified[key] = list(set(simplified[key]))
        
        return simplified
    
    def get_entity_statistics(self, entities: Dict[str, List[LegalEntity]]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        stats = {
            'total_entities': sum(len(entity_list) for entity_list in entities.values()),
            'entity_types': {entity_type: len(entity_list) for entity_type, entity_list in entities.items()},
            'high_confidence_entities': sum(1 for entity_list in entities.values() for entity in entity_list if entity.confidence > 0.8),
            'average_confidence': 0.0
        }
        
        if stats['total_entities'] > 0:
            all_confidences = [entity.confidence for entity_list in entities.values() for entity in entity_list]
            stats['average_confidence'] = sum(all_confidences) / len(all_confidences)
        
        return stats