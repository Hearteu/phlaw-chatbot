# chunker.py — Structure-aware legal document chunker for 21k text cases
import hashlib
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Dispositive patterns for extraction
DISPOSITIVE_HDR = r"(?:WHEREFORE|ACCORDINGLY|IN VIEW OF THE FOREGOING|IN VIEW WHEREOF|THUS|HENCE|PREMISES CONSIDERED)"
SO_ORDERED = r"SO\s+ORDERED\.?"
RULING_REGEX = re.compile(
    rf"{DISPOSITIVE_HDR}[\s\S]{{0,4000}}?{SO_ORDERED}",
    re.IGNORECASE,
)

class LegalDocumentChunker:
    """Structure-aware chunker for legal documents with sliding-window fallback"""
    
    def __init__(self, 
                 chunk_size: int = 640,  # tokens
                 overlap_ratio: float = 0.15,
                 min_chunk_size: int = 200,
                 max_dispositive_size: int = 1200):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_size = min_chunk_size
        self.max_dispositive_size = max_dispositive_size
        self.overlap_tokens = int(chunk_size * overlap_ratio)
    
    def chunk_case(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single case using structure-aware + sliding-window strategy"""
        chunks = []
        
        # Extract metadata
        metadata = self._extract_case_metadata(case_data)
        
        # 1. Create mini-summary chunk (always first)
        summary_chunk = self._create_mini_summary(case_data, metadata)
        chunks.append(summary_chunk)
        
        # 2. Extract and chunk by sections
        clean_text = case_data.get('clean_text', '')
        if not clean_text:
            return chunks
        
        # Extract sections using structure awareness
        sections = self._extract_sections(clean_text, case_data)
        
        # 3. Process each section with appropriate strategy
        for section_name, section_content in sections.items():
            section_chunks = self._chunk_section(
                section_content, 
                section_name, 
                metadata,
                case_data
            )
            chunks.extend(section_chunks)
        
        # 4. Add chunk indices and validate
        for i, chunk in enumerate(chunks):
            chunk['chunk_index'] = i
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    def _extract_case_metadata(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract core metadata for all chunks"""
        return {
            'case_id': case_data.get('gr_number') or case_data.get('special_number') or case_data.get('id', ''),
            'gr_number': case_data.get('gr_number', ''),
            'special_number': case_data.get('special_number', ''),
            'title': case_data.get('case_title', ''),
            'date': case_data.get('date', ''),
            'ponente': case_data.get('ponente', ''),
            'case_type': case_data.get('case_type', ''),
            'division': case_data.get('division', ''),
            'en_banc': case_data.get('en_banc', False),
            'source_url': case_data.get('source_url', ''),
            'promulgation_year': case_data.get('promulgation_year', ''),
            'is_administrative': case_data.get('is_administrative', False)
        }
    
    def _create_mini_summary(self, case_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact summary chunk with key case information"""
        
        # Extract core holding/ruling for summary
        clean_text = case_data.get('clean_text', '')
        ruling_excerpt = self._extract_dispositive_excerpt(clean_text)
        
        # Build summary content (target 200-350 tokens)
        summary_parts = []
        
        # Case identification
        if metadata['title']:
            summary_parts.append(f"Case: {metadata['title']}")
        
        if metadata['gr_number']:
            summary_parts.append(f"G.R. Number: {metadata['gr_number']}")
        elif metadata['special_number']:
            summary_parts.append(f"Special Number: {metadata['special_number']}")
        
        if metadata['date']:
            summary_parts.append(f"Date: {metadata['date']}")
        
        if metadata['ponente']:
            summary_parts.append(f"Ponente: {metadata['ponente']}")
        
        if metadata['case_type']:
            summary_parts.append(f"Case Type: {metadata['case_type']}")
        
        # Add brief ruling/holding if available
        if ruling_excerpt:
            summary_parts.append(f"Ruling: {ruling_excerpt}")
        
        # Add case subtypes if available
        case_subtypes = case_data.get('case_subtypes', [])
        if case_subtypes:
            summary_parts.append(f"Legal Areas: {', '.join(case_subtypes[:3])}")
        
        content = ". ".join(summary_parts)
        content_preview = content[:300] + "..." if len(content) > 300 else content
        
        # Create stable UUID for summary
        summary_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{metadata['case_id']}_summary"))
        
        return {
            'id': summary_id,
            'content': content,
            'content_preview': content_preview,
            'section': 'summary',
            'section_type': 'mini_summary',
            'paragraph_index': 0,
            'token_count': self._estimate_tokens(content),
            'metadata': metadata.copy(),
            'chunk_type': 'summary'
        }
    
    def _extract_sections(self, clean_text: str, case_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract structured sections from clean text"""
        sections = {}
        
        # Try to extract from existing sections dict first
        if 'sections' in case_data and isinstance(case_data['sections'], dict):
            for section_name, content in case_data['sections'].items():
                if content and isinstance(content, str) and content.strip():
                    sections[section_name.lower()] = content.strip()
        
        # If no sections dict, parse from clean_text
        if not sections and clean_text:
            sections = self._parse_sections_from_text(clean_text)
        
        # Ensure we have at least body content
        if not sections and clean_text:
            sections['body'] = clean_text
        
        return sections
    
    def _parse_sections_from_text(self, text: str) -> Dict[str, str]:
        """Parse sections from clean text using patterns"""
        sections = {}
        
        # Extract dispositive/ruling first (highest priority)
        ruling_match = RULING_REGEX.search(text)
        if ruling_match:
            sections['ruling'] = ruling_match.group(0).strip()
        
        # Simple heuristic patterns for other sections
        # Look for common section headers
        section_patterns = [
            (r'THE FACTS?[\s\S]*?(?=THE ISSUE|THE RULING|WHEREFORE|$)', 'facts'),
            (r'ANTECEDENT FACTS?[\s\S]*?(?=THE ISSUE|THE RULING|WHEREFORE|$)', 'facts'),
            (r'THE ISSUE[S]?[\s\S]*?(?=THE RULING|WHEREFORE|$)', 'issues'),
            (r'ISSUE[S]?[\s\S]*?(?=THE RULING|WHEREFORE|$)', 'issues'),
            (r'THE RULING[\s\S]*?(?=WHEREFORE|$)', 'arguments'),
            (r'DISCUSSION[\s\S]*?(?=WHEREFORE|$)', 'arguments'),
        ]
        
        for pattern, section_name in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and section_name not in sections:
                content = match.group(0).strip()
                if len(content) > 100:  # Only add substantial content
                    sections[section_name] = content
        
        # If no structured sections found, treat as body
        if not sections:
            sections['body'] = text
        
        return sections
    
    def _chunk_section(self, content: str, section_name: str, 
                      metadata: Dict[str, Any], case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a section using appropriate strategy based on section type"""
        if not content or not content.strip():
            return []
        
        content = content.strip()
        chunks = []
        
        if section_name == 'ruling' or section_name == 'dispositive':
            # Special handling for dispositive - keep intact if possible
            chunks = self._chunk_dispositive(content, metadata)
        elif section_name in ['facts', 'arguments', 'body']:
            # Use sliding window for long narrative sections
            chunks = self._chunk_with_sliding_window(content, section_name, metadata)
        elif section_name == 'issues':
            # Keep issues as single chunk if reasonable size
            if self._estimate_tokens(content) <= self.max_dispositive_size:
                chunks = [self._create_chunk(content, section_name, 0, metadata)]
            else:
                chunks = self._chunk_with_sliding_window(content, section_name, metadata)
        else:
            # Default sliding window for other sections
            chunks = self._chunk_with_sliding_window(content, section_name, metadata)
        
        return chunks
    
    def _chunk_dispositive(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special chunking for dispositive/ruling sections"""
        token_count = self._estimate_tokens(content)
        
        if token_count <= self.max_dispositive_size:
            # Keep as single chunk
            return [self._create_chunk(content, 'ruling', 0, metadata)]
        else:
            # Split at sentence boundaries with overlap
            sentences = self._split_into_sentences(content)
            return self._create_chunks_from_sentences(sentences, 'ruling', metadata)
    
    def _chunk_with_sliding_window(self, content: str, section_name: str, 
                                  metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk content using sliding window with sentence boundaries"""
        # Split into sentences first
        sentences = self._split_into_sentences(content)
        if not sentences:
            return []
        
        return self._create_chunks_from_sentences(sentences, section_name, metadata)
    
    def _create_chunks_from_sentences(self, sentences: List[str], section_name: str, 
                                    metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from sentences using sliding window"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        paragraph_index = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_chunk and current_tokens + sentence_tokens > self.chunk_size:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                if self._estimate_tokens(chunk_content) >= self.min_chunk_size:
                    chunk = self._create_chunk(chunk_content, section_name, paragraph_index, metadata)
                    chunks.append(chunk)
                    paragraph_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = sum(self._estimate_tokens(s) for s in overlap_sentences)
            
            # Add current sentence
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if self._estimate_tokens(chunk_content) >= self.min_chunk_size:
                chunk = self._create_chunk(chunk_content, section_name, paragraph_index, metadata)
                chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences:
            return []
        
        overlap_sentences = []
        tokens = 0
        
        # Take sentences from the end until we reach overlap token limit
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if tokens + sentence_tokens <= self.overlap_tokens:
                overlap_sentences.insert(0, sentence)
                tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(self, content: str, section: str, paragraph_index: int, 
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized chunk dictionary"""
        content_preview = content[:300] + "..." if len(content) > 300 else content
        
        # Create stable UUID for chunk
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{metadata['case_id']}_{section}_{paragraph_index}"))
        
        return {
            'id': chunk_id,
            'content': content,
            'content_preview': content_preview,
            'section': section,
            'section_type': self._classify_section_type(section),
            'paragraph_index': paragraph_index,
            'token_count': self._estimate_tokens(content),
            'metadata': metadata.copy(),
            'chunk_type': 'content'
        }
    
    def _classify_section_type(self, section: str) -> str:
        """Classify section type for retrieval boosting"""
        section_lower = section.lower()
        
        if section_lower in ['ruling', 'dispositive', 'decision']:
            return 'dispositive'
        elif section_lower in ['facts', 'antecedent_facts']:
            return 'factual'
        elif section_lower in ['issues', 'issue']:
            return 'issues'
        elif section_lower in ['arguments', 'discussion', 'analysis']:
            return 'legal_analysis'
        elif section_lower == 'summary':
            return 'summary'
        else:
            return 'general'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with legal document awareness"""
        if not text:
            return []
        
        # Simple sentence splitting - can be enhanced with legal-specific rules
        # Handle common legal abbreviations
        text = re.sub(r'\bG\.R\.\s+No\.', 'GR_No', text)  # Protect G.R. No.
        text = re.sub(r'\bA\.M\.\s+No\.', 'AM_No', text)  # Protect A.M. No.
        text = re.sub(r'\bU\.S\.', 'US', text)  # Protect U.S.
        text = re.sub(r'\bPhil\.', 'Phil', text)  # Protect Phil.
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected abbreviations
        sentences = [s.replace('GR_No', 'G.R. No.').replace('AM_No', 'A.M. No.')
                    .replace('US', 'U.S.').replace('Phil', 'Phil.') for s in sentences]
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def _extract_dispositive_excerpt(self, text: str) -> str:
        """Extract a brief excerpt from the dispositive for summary"""
        ruling_match = RULING_REGEX.search(text)
        if ruling_match:
            full_ruling = ruling_match.group(0)
            # Take first 200 characters as excerpt
            excerpt = full_ruling[:200].strip()
            if len(full_ruling) > 200:
                # Try to end at word boundary
                last_space = excerpt.rfind(' ')
                if last_space > 150:
                    excerpt = excerpt[:last_space] + "..."
            return excerpt
        return ""
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        sections = {}
        
        for chunk in chunks:
            section = chunk.get('section', 'unknown')
            if section not in sections:
                sections[section] = {'count': 0, 'tokens': 0}
            sections[section]['count'] += 1
            sections[section]['tokens'] += chunk.get('token_count', 0)
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': total_tokens / len(chunks) if chunks else 0,
            'sections': sections,
            'chunk_types': list(set(chunk.get('chunk_type', 'content') for chunk in chunks))
        }


def chunk_legal_document(case_data: Dict[str, Any], 
                        chunk_size: int = 640,
                        overlap_ratio: float = 0.15) -> List[Dict[str, Any]]:
    """Convenience function to chunk a single legal document"""
    chunker = LegalDocumentChunker(
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio
    )
    return chunker.chunk_case(case_data)


def chunk_legal_documents(cases: List[Dict[str, Any]], 
                         chunk_size: int = 640,
                         overlap_ratio: float = 0.15) -> List[Dict[str, Any]]:
    """Chunk multiple legal documents"""
    chunker = LegalDocumentChunker(
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio
    )
    
    all_chunks = []
    for case in cases:
        chunks = chunker.chunk_case(case)
        all_chunks.extend(chunks)
    
    return all_chunks
