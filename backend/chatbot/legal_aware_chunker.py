# legal_aware_chunker.py — Legal document-aware chunking with concept boundary detection
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import nltk


class ChunkType(Enum):
    """Types of legal document chunks"""
    RULING = "ruling"
    FACTS = "facts"
    ISSUES = "issues"
    ARGUMENTS = "arguments"
    HEADER = "header"
    BODY = "body"
    LEGAL_CITATION = "legal_citation"
    CASE_REFERENCE = "case_reference"

@dataclass
class LegalChunk:
    """Represents a chunk of legal text with metadata"""
    text: str
    chunk_type: ChunkType
    start_pos: int
    end_pos: int
    confidence: float
    legal_concepts: List[str]
    case_references: List[str]
    legal_citations: List[str]
    priority: int  # Higher number = higher priority

class LegalAwareChunker:
    """Advanced chunking system for legal documents"""
    
    def __init__(self, 
                 base_chunk_size: int = 2000,
                 overlap_size: int = 200,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 4000):
        self.base_chunk_size = base_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        self._init_legal_boundaries()
        self._init_legal_concepts()
        self._init_sentence_segmentation()
        self._init_legal_citations()
    
    def _init_legal_boundaries(self):
        """Initialize legal document boundary patterns"""
        self.legal_boundaries = {
            # Ruling/Decision patterns
            'ruling': [
                r'WHEREFORE\s+',
                r'ACCORDINGLY\s+',
                r'IN\s+VIEW\s+OF\s+THE\s+FOREGOING\s+',
                r'PREMISES\s+CONSIDERED\s+',
                r'SO\s+ORDERED\.?',
                r'DECISION\s+IS\s+HEREBY\s+',
                r'JUDGMENT\s+IS\s+HEREBY\s+',
            ],
            
            # Facts patterns
            'facts': [
                r'FACTS\s*[:\-–]?\s*$',
                r'STATEMENT\s+OF\s+FACTS\s*[:\-–]?\s*$',
                r'ANTECEDENT\s+FACTS\s*[:\-–]?\s*$',
                r'FACTUAL\s+ANTECEDENTS\s*[:\-–]?\s*$',
                r'FACTUAL\s+BACKGROUND\s*[:\-–]?\s*$',
            ],
            
            # Issues patterns
            'issues': [
                r'ISSUES?\s*[:\-–]?\s*$',
                r'ISSUES?\s+FOR\s+RESOLUTION\s*[:\-–]?\s*$',
                r'QUESTIONS?\s+PRESENTED\s*[:\-–]?\s*$',
                r'LEGAL\s+QUESTIONS?\s*[:\-–]?\s*$',
                r'WHETHER\s+',
            ],
            
            # Arguments patterns
            'arguments': [
                r'ARGUMENTS?\s*[:\-–]?\s*$',
                r'LEGAL\s+ARGUMENTS?\s*[:\-–]?\s*$',
                r'REASONING\s*[:\-–]?\s*$',
                r'RATIO\s+DECIDENDI\s*[:\-–]?\s*$',
                r'LEGAL\s+REASONING\s*[:\-–]?\s*$',
            ],
            
            # Discussion patterns
            'discussion': [
                r'DISCUSSION\s*[:\-–]?\s*$',
                r'OPINION\s*[:\-–]?\s*$',
                r'CONCURRING\s+OPINION\s*[:\-–]?\s*$',
                r'DISSENTING\s+OPINION\s*[:\-–]?\s*$',
                r'SEPARATE\s+OPINION\s*[:\-–]?\s*$',
            ],
            
            # Citations patterns
            'citations': [
                r'CITATIONS?\s*[:\-–]?\s*$',
                r'AUTHORITIES?\s*[:\-–]?\s*$',
                r'LEGAL\s+AUTHORITIES?\s*[:\-–]?\s*$',
                r'CASE\s+LAW\s*[:\-–]?\s*$',
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_boundaries = {}
        for section_type, patterns in self.legal_boundaries.items():
            self.compiled_boundaries[section_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                for pattern in patterns
            ]
    
    def _init_legal_concepts(self):
        """Initialize legal concept patterns for better chunking"""
        self.legal_concepts = {
            # Constitutional concepts
            'constitutional': [
                'due process', 'equal protection', 'freedom of speech', 'bill of rights',
                'separation of powers', 'checks and balances', 'judicial review',
                'constitutional right', 'fundamental right'
            ],
            
            # Civil law concepts
            'civil': [
                'obligation', 'contract', 'tort', 'damages', 'property', 'ownership',
                'possession', 'succession', 'inheritance', 'marriage', 'divorce',
                'liability', 'negligence', 'breach of contract'
            ],
            
            # Criminal law concepts
            'criminal': [
                'murder', 'homicide', 'theft', 'robbery', 'fraud', 'estafa',
                'malversation', 'bribery', 'corruption', 'penalty', 'imprisonment',
                'criminal liability', 'mens rea', 'actus reus'
            ],
            
            # Administrative concepts
            'administrative': [
                'civil service', 'government', 'public officer', 'discipline',
                'dismissal', 'suspension', 'administrative case', 'ombudsman',
                'administrative liability', 'grave misconduct'
            ],
            
            # Labor concepts
            'labor': [
                'employment', 'wage', 'benefits', 'termination', 'unfair labor practice',
                'collective bargaining', 'strike', 'lockout', 'labor union',
                'labor standards', 'social justice'
            ]
        }
        
        # Create concept patterns
        self.concept_patterns = {}
        for category, concepts in self.legal_concepts.items():
            pattern = '|'.join(re.escape(concept) for concept in concepts)
            self.concept_patterns[category] = re.compile(
                r'\b(' + pattern + r')\b', re.IGNORECASE
            )
    
    def _init_sentence_segmentation(self):
        """Initialize sentence segmentation for better chunking"""
        try:
            nltk.download('punkt', quiet=True)
            self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except:
            # Fallback to simple sentence splitting
            self.sentence_tokenizer = None
    
    def _init_legal_citations(self):
        """Initialize legal citation patterns"""
        self.citation_patterns = [
            # Philippine Reports
            r'\d+\s+Phil\.\s+\d+',
            # Supreme Court Reports Annotated
            r'\d+\s+SCRA\s+\d+',
            # Official Gazette
            r'\d+\s+OG\s+\d+',
            r'\d+\s+O\.G\.\s+\d+',
            # Supplement
            r'\d+\s+Supp\.\s+\d+',
            # G.R. Numbers
            r'G\.R\.\s+No\.\s+\d+',
            r'GR\s+No\.\s+\d+',
        ]
        
        self.compiled_citations = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.citation_patterns
        ]
    
    def chunk_document(self, text: str, document_metadata: Optional[Dict] = None) -> List[LegalChunk]:
        """Chunk a legal document with awareness of legal structure"""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # First, identify legal sections
        sections = self._identify_legal_sections(text)
        
        # Chunk each section appropriately
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, text)
            chunks.extend(section_chunks)
        
        # If no sections identified, chunk the entire text
        if not chunks:
            chunks = self._chunk_entire_text(text)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks, text)
        
        return chunks
    
    def _identify_legal_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify legal sections in the document"""
        sections = []
        
        for section_type, patterns in self.compiled_boundaries.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    section = {
                        'type': section_type,
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'confidence': self._calculate_section_confidence(section_type, match.group())
                    }
                    sections.append(section)
        
        # Sort sections by position
        sections.sort(key=lambda x: x['start'])
        
        # Merge overlapping sections
        sections = self._merge_overlapping_sections(sections)
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any], full_text: str) -> List[LegalChunk]:
        """Chunk a specific legal section"""
        section_type = section['type']
        start_pos = section['start']
        end_pos = section['end']
        
        # Determine chunking strategy based on section type
        if section_type == 'ruling':
            return self._chunk_ruling_section(section, full_text)
        elif section_type in ['facts', 'issues', 'arguments']:
            return self._chunk_structured_section(section, full_text)
        else:
            return self._chunk_general_section(section, full_text)
    
    def _chunk_ruling_section(self, section: Dict[str, Any], full_text: str) -> List[LegalChunk]:
        """Special chunking for ruling sections"""
        chunks = []
        
        # Extract the full ruling text
        ruling_start = section['start']
        ruling_end = self._find_ruling_end(full_text, ruling_start)
        ruling_text = full_text[ruling_start:ruling_end]
        
        # Ruling sections are usually kept as single chunks due to their importance
        if len(ruling_text) <= self.max_chunk_size:
            chunk = LegalChunk(
                text=ruling_text,
                chunk_type=ChunkType.RULING,
                start_pos=ruling_start,
                end_pos=ruling_end,
                confidence=section['confidence'],
                legal_concepts=self._extract_legal_concepts(ruling_text),
                case_references=self._extract_case_references(ruling_text),
                legal_citations=self._extract_legal_citations(ruling_text),
                priority=10  # Highest priority
            )
            chunks.append(chunk)
        else:
            # If too large, chunk with high overlap to preserve context
            sub_chunks = self._chunk_with_high_overlap(ruling_text, ruling_start)
            for sub_chunk in sub_chunks:
                sub_chunk.chunk_type = ChunkType.RULING
                sub_chunk.priority = 10
                chunks.append(sub_chunk)
        
        return chunks
    
    def _chunk_structured_section(self, section: Dict[str, Any], full_text: str) -> List[LegalChunk]:
        """Chunk structured sections (facts, issues, arguments)"""
        chunks = []
        
        # Extract section text
        section_start = section['start']
        section_end = self._find_section_end(full_text, section_start)
        section_text = full_text[section_start:section_end]
        
        # Chunk by paragraphs first
        paragraphs = self._split_into_paragraphs(section_text)
        
        current_chunk = ""
        current_start = section_start
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, create a chunk
            if len(current_chunk) + len(paragraph) > self.base_chunk_size and current_chunk:
                chunk = self._create_chunk_from_text(
                    current_chunk, current_start, section['type'], full_text
                )
                chunks.append(chunk)
                current_chunk = paragraph
                current_start = section_start + len(current_chunk)
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining text as final chunk
        if current_chunk:
            chunk = self._create_chunk_from_text(
                current_chunk, current_start, section['type'], full_text
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_general_section(self, section: Dict[str, Any], full_text: str) -> List[LegalChunk]:
        """Chunk general sections using standard chunking"""
        section_text = full_text[section['start']:section['end']]
        
        if len(section_text) <= self.base_chunk_size:
            chunk = self._create_chunk_from_text(
                section_text, section['start'], section['type'], full_text
            )
            return [chunk]
        else:
            return self._chunk_with_sentence_boundaries(section_text, section['start'])
    
    def _chunk_entire_text(self, text: str) -> List[LegalChunk]:
        """Chunk entire text when no sections are identified"""
        return self._chunk_with_sentence_boundaries(text, 0)
    
    def _chunk_with_sentence_boundaries(self, text: str, start_offset: int) -> List[LegalChunk]:
        """Chunk text respecting sentence boundaries"""
        chunks = []
        
        if self.sentence_tokenizer:
            sentences = self.sentence_tokenizer.tokenize(text)
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_start = start_offset
        chunk_start_pos = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, create a chunk
            if len(current_chunk) + len(sentence) > self.base_chunk_size and current_chunk:
                chunk = self._create_chunk_from_text(
                    current_chunk, current_start, 'body', text
                )
                chunks.append(chunk)
                current_chunk = sentence
                chunk_start_pos = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining text as final chunk
        if current_chunk:
            chunk = self._create_chunk_from_text(
                current_chunk, current_start, 'body', text
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_with_high_overlap(self, text: str, start_offset: int) -> List[LegalChunk]:
        """Chunk with high overlap for important sections"""
        chunks = []
        step_size = self.base_chunk_size - self.overlap_size * 2  # High overlap
        
        for i in range(0, len(text), step_size):
            chunk_text = text[i:i + self.base_chunk_size]
            if len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk_from_text(
                    chunk_text, start_offset + i, 'ruling', text
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_text(self, text: str, start_pos: int, section_type: str, full_text: str) -> LegalChunk:
        """Create a LegalChunk from text"""
        # Map section type to ChunkType
        chunk_type_mapping = {
            'ruling': ChunkType.RULING,
            'facts': ChunkType.FACTS,
            'issues': ChunkType.ISSUES,
            'arguments': ChunkType.ARGUMENTS,
            'header': ChunkType.HEADER,
            'body': ChunkType.BODY
        }
        
        chunk_type = chunk_type_mapping.get(section_type, ChunkType.BODY)
        
        # Calculate priority based on chunk type
        priority_mapping = {
            ChunkType.RULING: 10,
            ChunkType.ISSUES: 8,
            ChunkType.FACTS: 7,
            ChunkType.ARGUMENTS: 6,
            ChunkType.HEADER: 5,
            ChunkType.BODY: 3
        }
        
        priority = priority_mapping.get(chunk_type, 3)
        
        return LegalChunk(
            text=text.strip(),
            chunk_type=chunk_type,
            start_pos=start_pos,
            end_pos=start_pos + len(text),
            confidence=0.8,  # Default confidence
            legal_concepts=self._extract_legal_concepts(text),
            case_references=self._extract_case_references(text),
            legal_citations=self._extract_legal_citations(text),
            priority=priority
        )
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        for category, pattern in self.concept_patterns.items():
            matches = pattern.findall(text_lower)
            concepts.extend(matches)
        
        return list(set(concepts))  # Remove duplicates
    
    def _extract_case_references(self, text: str) -> List[str]:
        """Extract case references from text"""
        case_refs = []
        
        # G.R. number patterns
        gr_patterns = [
            r'G\.R\.\s+No\.\s+\d+',
            r'GR\s+No\.\s+\d+',
            r'G\.R\.\s+Nos\.\s+\d+(?:[,\s]+and\s+)?\d+'
        ]
        
        for pattern in gr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            case_refs.extend(matches)
        
        # Case name patterns
        case_name_pattern = r'([A-Z][A-Za-z\s&,\.]+?)\s+v\.?\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s*,\s*G\.R\.|\s*,\s*GR|\s*,\s*No\.|\s*,\s*\d{4}|\s*$)'
        case_names = re.findall(case_name_pattern, text)
        for petitioner, respondent in case_names:
            case_refs.append(f"{petitioner} v. {respondent}")
        
        return list(set(case_refs))
    
    def _extract_legal_citations(self, text: str) -> List[str]:
        """Extract legal citations from text"""
        citations = []
        
        for pattern in self.compiled_citations:
            matches = pattern.findall(text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _find_ruling_end(self, text: str, ruling_start: int) -> int:
        """Find the end of a ruling section"""
        # Look for "SO ORDERED" or similar ending patterns
        end_patterns = [
            r'SO\s+ORDERED\.?',
            r'IT\s+IS\s+SO\s+ORDERED\.?',
            r'DECISION\s+IS\s+HEREBY\s+RENDERED\.?',
            r'JUDGMENT\s+IS\s+HEREBY\s+RENDERED\.?'
        ]
        
        search_text = text[ruling_start:]
        for pattern in end_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return ruling_start + match.end()
        
        # If no clear end found, use a reasonable length
        return min(ruling_start + self.max_chunk_size, len(text))
    
    def _find_section_end(self, text: str, section_start: int) -> int:
        """Find the end of a section"""
        # Look for next section header or end of document
        next_section_pattern = r'\n\s*(?:FACTS|ISSUES|ARGUMENTS|DISCUSSION|CITATIONS|RULING|DECISION)\s*[:\-–]?\s*$'
        
        search_text = text[section_start:]
        match = re.search(next_section_pattern, search_text, re.IGNORECASE | re.MULTILINE)
        
        if match:
            return section_start + match.start()
        else:
            # Look for next major heading
            heading_pattern = r'\n\s*[A-Z][A-Z\s]{10,}\s*$'
            match = re.search(heading_pattern, search_text, re.MULTILINE)
            if match:
                return section_start + match.start()
        
        # Default to reasonable length
        return min(section_start + self.max_chunk_size, len(text))
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _calculate_section_confidence(self, section_type: str, text: str) -> float:
        """Calculate confidence for section identification"""
        base_confidence = {
            'ruling': 0.9,
            'facts': 0.8,
            'issues': 0.8,
            'arguments': 0.7,
            'discussion': 0.7,
            'citations': 0.8
        }
        
        confidence = base_confidence.get(section_type, 0.5)
        
        # Boost confidence for specific indicators
        text_lower = text.lower()
        if section_type == 'ruling' and 'so ordered' in text_lower:
            confidence += 0.1
        elif section_type == 'facts' and any(word in text_lower for word in ['factual', 'background', 'antecedent']):
            confidence += 0.1
        elif section_type == 'issues' and 'whether' in text_lower:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _merge_overlapping_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping sections"""
        if not sections:
            return sections
        
        merged = [sections[0]]
        
        for current in sections[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current['start'] < last['end']:
                # Merge sections
                last['end'] = max(last['end'], current['end'])
                last['text'] = last['text'] + " " + current['text']
                last['confidence'] = max(last['confidence'], current['confidence'])
            else:
                merged.append(current)
        
        return merged
    
    def _post_process_chunks(self, chunks: List[LegalChunk], full_text: str) -> List[LegalChunk]:
        """Post-process chunks for quality and consistency"""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean up text
            chunk.text = self._clean_chunk_text(chunk.text)
            
            # Skip chunks that are too short or empty
            if len(chunk.text.strip()) < self.min_chunk_size:
                continue
            
            # Update legal concepts, case references, and citations
            chunk.legal_concepts = self._extract_legal_concepts(chunk.text)
            chunk.case_references = self._extract_case_references(chunk.text)
            chunk.legal_citations = self._extract_legal_citations(chunk.text)
            
            # Calculate final confidence
            chunk.confidence = self._calculate_chunk_confidence(chunk)
            
            processed_chunks.append(chunk)
        
        # Sort by priority and confidence
        processed_chunks.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        return processed_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean up chunk text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'—\s*(?:body|header|facts|issues|ruling|arguments|keywords)\s*—\s*[^—\n]*',
            r'—\s*N/A\s*',
            r'Supreme Court E-Library[^—]*',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _calculate_chunk_confidence(self, chunk: LegalChunk) -> float:
        """Calculate confidence for a chunk"""
        base_confidence = 0.5
        
        # Boost confidence for legal concepts
        if chunk.legal_concepts:
            base_confidence += min(len(chunk.legal_concepts) * 0.1, 0.3)
        
        # Boost confidence for case references
        if chunk.case_references:
            base_confidence += min(len(chunk.case_references) * 0.1, 0.2)
        
        # Boost confidence for legal citations
        if chunk.legal_citations:
            base_confidence += min(len(chunk.legal_citations) * 0.1, 0.2)
        
        # Boost confidence for high-priority chunk types
        if chunk.chunk_type in [ChunkType.RULING, ChunkType.ISSUES, ChunkType.FACTS]:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def get_chunk_statistics(self, chunks: List[LegalChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'chunk_types': {},
            'average_length': sum(len(chunk.text) for chunk in chunks) / len(chunks),
            'high_confidence_chunks': sum(1 for chunk in chunks if chunk.confidence > 0.8),
            'total_legal_concepts': sum(len(chunk.legal_concepts) for chunk in chunks),
            'total_case_references': sum(len(chunk.case_references) for chunk in chunks),
            'total_legal_citations': sum(len(chunk.legal_citations) for chunk in chunks)
        }
        
        # Count chunk types
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        return stats
