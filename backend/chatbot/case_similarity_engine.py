# case_similarity_engine.py — Case similarity and recommendation system
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarCase:
    """Represents a similar case with similarity metrics"""
    case_id: str
    title: str
    gr_number: str
    year: int
    similarity_score: float
    similarity_type: str
    shared_concepts: List[str]
    shared_citations: List[str]
    legal_area: str
    court_division: str
    ponente: str

@dataclass
class CaseProfile:
    """Profile of a case for similarity analysis"""
    case_id: str
    title: str
    gr_number: str
    year: int
    legal_concepts: List[str]
    legal_citations: List[str]
    case_references: List[str]
    legal_area: str
    court_division: str
    ponente: str
    content_vector: Optional[np.ndarray] = None
    topic_vector: Optional[np.ndarray] = None

class CaseSimilarityEngine:
    """Advanced case similarity and recommendation engine"""
    
    def __init__(self, vectorizer_model_path: Optional[str] = None):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Reduced for small datasets
            stop_words='english',
            ngram_range=(1, 2),  # Reduced ngram range
            min_df=1,  # Allow single document terms
            max_df=1.0  # Allow all terms
        )
        self.lda_model = None
        self.case_profiles = {}
        self.legal_area_classifier = None
        self.similarity_cache = {}
        self.model_path = vectorizer_model_path or "backend/chatbot/similarity_models"
        
        # Initialize legal area mapping
        self._init_legal_areas()
        self._init_legal_concept_weights()
        
        # Load or initialize models
        self._load_or_initialize_models()
    
    def _init_legal_areas(self):
        """Initialize legal area classification"""
        self.legal_areas = {
            'constitutional': {
                'keywords': ['constitutional', 'constitution', 'bill of rights', 'due process', 
                           'equal protection', 'freedom of speech', 'separation of powers'],
                'weight': 1.0
            },
            'criminal': {
                'keywords': ['criminal', 'murder', 'homicide', 'theft', 'robbery', 'fraud', 
                           'estafa', 'malversation', 'bribery', 'penalty', 'imprisonment'],
                'weight': 0.9
            },
            'civil': {
                'keywords': ['civil', 'contract', 'obligation', 'damages', 'property', 'ownership', 
                           'possession', 'tort', 'liability', 'negligence', 'marriage', 'divorce'],
                'weight': 0.9
            },
            'administrative': {
                'keywords': ['administrative', 'government', 'public officer', 'civil service', 
                           'discipline', 'dismissal', 'suspension', 'ombudsman'],
                'weight': 0.8
            },
            'labor': {
                'keywords': ['labor', 'employment', 'wage', 'benefits', 'termination', 
                           'unfair labor practice', 'collective bargaining', 'strike'],
                'weight': 0.8
            },
            'commercial': {
                'keywords': ['commercial', 'corporation', 'partnership', 'bankruptcy', 
                           'negotiable instrument', 'insurance', 'securities'],
                'weight': 0.8
            },
            'family': {
                'keywords': ['family', 'marriage', 'divorce', 'annulment', 'custody', 
                           'support', 'adoption', 'domestic relations'],
                'weight': 0.8
            },
            'special': {
                'keywords': ['land reform', 'agrarian reform', 'eminent domain', 'expropriation', 
                           'environmental', 'pollution', 'natural resources', 'indigenous peoples'],
                'weight': 0.9
            }
        }
    
    def _init_legal_concept_weights(self):
        """Initialize weights for legal concepts"""
        self.concept_weights = {
            'constitutional_rights': 1.0,
            'procedural_law': 0.9,
            'substantive_law': 0.9,
            'remedies': 0.8,
            'jurisdiction': 0.8,
            'evidence': 0.7,
            'appeals': 0.7,
            'enforcement': 0.6
        }
    
    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing models
        try:
            self._load_models()
            print("✅ Loaded existing similarity models")
        except:
            print("🔄 Initializing new similarity models")
            self._initialize_models()
    
    def _load_models(self):
        """Load existing models from disk"""
        vectorizer_path = os.path.join(self.model_path, "vectorizer.pkl")
        lda_path = os.path.join(self.model_path, "lda_model.pkl")
        profiles_path = os.path.join(self.model_path, "case_profiles.json")
        
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        if os.path.exists(lda_path):
            with open(lda_path, 'rb') as f:
                self.lda_model = pickle.load(f)
        
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
                self.case_profiles = {
                    case_id: CaseProfile(**profile) 
                    for case_id, profile in profiles_data.items()
                }
    
    def _initialize_models(self):
        """Initialize new models"""
        # Initialize LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=min(10, 5),  # Use fewer components for small datasets
            random_state=42,
            max_iter=50
        )
    
    def _save_models(self):
        """Save models to disk"""
        vectorizer_path = os.path.join(self.model_path, "vectorizer.pkl")
        lda_path = os.path.join(self.model_path, "lda_model.pkl")
        profiles_path = os.path.join(self.model_path, "case_profiles.json")
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        if self.lda_model:
            with open(lda_path, 'wb') as f:
                pickle.dump(self.lda_model, f)
        
        # Convert case profiles to serializable format
        profiles_data = {}
        for case_id, profile in self.case_profiles.items():
            profiles_data[case_id] = {
                'case_id': profile.case_id,
                'title': profile.title,
                'gr_number': profile.gr_number,
                'year': profile.year,
                'legal_concepts': profile.legal_concepts,
                'legal_citations': profile.legal_citations,
                'case_references': profile.case_references,
                'legal_area': profile.legal_area,
                'court_division': profile.court_division,
                'ponente': profile.ponente
            }
        
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, indent=2)
    
    def add_case(self, case_data: Dict[str, Any]) -> str:
        """Add a case to the similarity engine"""
        case_id = case_data.get('id', f"case_{len(self.case_profiles)}")
        
        # Extract legal concepts
        legal_concepts = self._extract_legal_concepts(case_data)
        
        # Extract legal citations
        legal_citations = self._extract_legal_citations(case_data)
        
        # Extract case references
        case_references = self._extract_case_references(case_data)
        
        # Classify legal area
        legal_area = self._classify_legal_area(case_data, legal_concepts)
        
        # Create case profile
        profile = CaseProfile(
            case_id=case_id,
            title=case_data.get('title', ''),
            gr_number=case_data.get('gr_number', ''),
            year=case_data.get('year', 0),
            legal_concepts=legal_concepts,
            legal_citations=legal_citations,
            case_references=case_references,
            legal_area=legal_area,
            court_division=case_data.get('division', ''),
            ponente=case_data.get('ponente', '')
        )
        
        self.case_profiles[case_id] = profile
        
        # Update models if needed
        self._update_models()
        
        return case_id
    
    def _extract_legal_concepts(self, case_data: Dict[str, Any]) -> List[str]:
        """Extract legal concepts from case data"""
        concepts = []
        
        # Extract from title
        title = case_data.get('title', '')
        concepts.extend(self._extract_concepts_from_text(title))
        
        # Extract from content
        content = case_data.get('content', '')
        if isinstance(content, dict):
            # If content is structured, extract from all sections
            for section_content in content.values():
                if isinstance(section_content, str):
                    concepts.extend(self._extract_concepts_from_text(section_content))
        elif isinstance(content, str):
            concepts.extend(self._extract_concepts_from_text(content))
        
        # Remove duplicates and return
        return list(set(concepts))
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract legal concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        for area, data in self.legal_areas.items():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    concepts.append(keyword)
        
        return concepts
    
    def _extract_legal_citations(self, case_data: Dict[str, Any]) -> List[str]:
        """Extract legal citations from case data"""
        citations = []
        
        # Citation patterns
        citation_patterns = [
            r'\d+\s+Phil\.\s+\d+',
            r'\d+\s+SCRA\s+\d+',
            r'\d+\s+OG\s+\d+',
            r'\d+\s+O\.G\.\s+\d+',
            r'\d+\s+Supp\.\s+\d+'
        ]
        
        content = case_data.get('content', '')
        if isinstance(content, dict):
            content = ' '.join(str(v) for v in content.values() if isinstance(v, str))
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _extract_case_references(self, case_data: Dict[str, Any]) -> List[str]:
        """Extract case references from case data"""
        references = []
        
        # G.R. number patterns
        gr_patterns = [
            r'G\.R\.\s+No\.\s+\d+',
            r'GR\s+No\.\s+\d+'
        ]
        
        content = case_data.get('content', '')
        if isinstance(content, dict):
            content = ' '.join(str(v) for v in content.values() if isinstance(v, str))
        
        for pattern in gr_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))
    
    def _classify_legal_area(self, case_data: Dict[str, Any], legal_concepts: List[str]) -> str:
        """Classify the legal area of a case"""
        # Combine title and concepts for classification
        text = case_data.get('title', '') + ' ' + ' '.join(legal_concepts)
        text_lower = text.lower()
        
        area_scores = {}
        for area, data in self.legal_areas.items():
            score = 0
            for keyword in data['keywords']:
                if keyword in text_lower:
                    score += data['weight']
            area_scores[area] = score
        
        if area_scores and max(area_scores.values()) > 0:
            return max(area_scores, key=area_scores.get)
        else:
            return 'general'
    
    def _update_models(self):
        """Update similarity models with new cases"""
        if len(self.case_profiles) < 2:
            return
        
        # Prepare text data for vectorization
        texts = []
        for profile in self.case_profiles.values():
            text = f"{profile.title} {' '.join(profile.legal_concepts)}"
            texts.append(text)
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        
        # Transform texts to vectors
        vectors = self.vectorizer.transform(texts)
        
        # Update case profiles with vectors
        for i, (case_id, profile) in enumerate(self.case_profiles.items()):
            profile.content_vector = vectors[i].toarray().flatten()
        
        # Fit LDA model
        if len(texts) >= 2:  # Reduced minimum documents for LDA
            try:
                # Adjust n_components based on number of documents
                n_components = min(self.lda_model.n_components, len(texts) - 1)
                if n_components < 2:
                    n_components = 2
                
                # Create a new LDA model with adjusted components
                from sklearn.decomposition import LatentDirichletAllocation
                adjusted_lda = LatentDirichletAllocation(
                    n_components=n_components,
                    random_state=42,
                    max_iter=50
                )
                
                adjusted_lda.fit(vectors)
                self.lda_model = adjusted_lda
                
                # Get topic distributions
                topic_distributions = self.lda_model.transform(vectors)
                
                # Update case profiles with topic vectors
                for i, (case_id, profile) in enumerate(self.case_profiles.items()):
                    profile.topic_vector = topic_distributions[i]
            except Exception as e:
                print(f"Warning: LDA model fitting failed: {e}")
                # Continue without LDA model
        
        # Save updated models
        self._save_models()
    
    def find_similar_cases(self, case_id: str, top_k: int = 10) -> List[SimilarCase]:
        """Find similar cases to a given case"""
        if case_id not in self.case_profiles:
            return []
        
        target_profile = self.case_profiles[case_id]
        similar_cases = []
        
        for other_id, other_profile in self.case_profiles.items():
            if other_id == case_id:
                continue
            
            # Calculate similarity scores
            content_similarity = self._calculate_content_similarity(target_profile, other_profile)
            concept_similarity = self._calculate_concept_similarity(target_profile, other_profile)
            citation_similarity = self._calculate_citation_similarity(target_profile, other_profile)
            topic_similarity = self._calculate_topic_similarity(target_profile, other_profile)
            
            # Weighted similarity score
            total_similarity = (
                content_similarity * 0.4 +
                concept_similarity * 0.3 +
                citation_similarity * 0.2 +
                topic_similarity * 0.1
            )
            
            if total_similarity > 0.1:  # Minimum similarity threshold
                similar_case = SimilarCase(
                    case_id=other_id,
                    title=other_profile.title,
                    gr_number=other_profile.gr_number,
                    year=other_profile.year,
                    similarity_score=total_similarity,
                    similarity_type=self._determine_similarity_type(
                        content_similarity, concept_similarity, citation_similarity
                    ),
                    shared_concepts=self._get_shared_concepts(target_profile, other_profile),
                    shared_citations=self._get_shared_citations(target_profile, other_profile),
                    legal_area=other_profile.legal_area,
                    court_division=other_profile.court_division,
                    ponente=other_profile.ponente
                )
                similar_cases.append(similar_case)
        
        # Sort by similarity score and return top k
        similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_cases[:top_k]
    
    def find_similar_cases_by_query(self, query: str, top_k: int = 10) -> List[SimilarCase]:
        """Find similar cases based on a query"""
        # Extract concepts from query
        query_concepts = self._extract_concepts_from_text(query)
        
        # Create a temporary profile for the query
        query_profile = CaseProfile(
            case_id="query",
            title=query,
            gr_number="",
            year=0,
            legal_concepts=query_concepts,
            legal_citations=[],
            case_references=[],
            legal_area=self._classify_legal_area({'title': query}, query_concepts),
            court_division="",
            ponente=""
        )
        
        similar_cases = []
        
        for case_id, profile in self.case_profiles.items():
            # Calculate similarity scores
            concept_similarity = self._calculate_concept_similarity(query_profile, profile)
            area_similarity = 1.0 if query_profile.legal_area == profile.legal_area else 0.0
            
            # Weighted similarity score
            total_similarity = concept_similarity * 0.7 + area_similarity * 0.3
            
            if total_similarity > 0.1:  # Minimum similarity threshold
                similar_case = SimilarCase(
                    case_id=case_id,
                    title=profile.title,
                    gr_number=profile.gr_number,
                    year=profile.year,
                    similarity_score=total_similarity,
                    similarity_type="concept_based",
                    shared_concepts=self._get_shared_concepts(query_profile, profile),
                    shared_citations=[],
                    legal_area=profile.legal_area,
                    court_division=profile.court_division,
                    ponente=profile.ponente
                )
                similar_cases.append(similar_case)
        
        # Sort by similarity score and return top k
        similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_cases[:top_k]
    
    def _calculate_content_similarity(self, profile1: CaseProfile, profile2: CaseProfile) -> float:
        """Calculate content similarity between two case profiles"""
        if profile1.content_vector is None or profile2.content_vector is None:
            return 0.0
        
        try:
            similarity = cosine_similarity(
                profile1.content_vector.reshape(1, -1),
                profile2.content_vector.reshape(1, -1)
            )[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_concept_similarity(self, profile1: CaseProfile, profile2: CaseProfile) -> float:
        """Calculate concept similarity between two case profiles"""
        concepts1 = set(profile1.legal_concepts)
        concepts2 = set(profile2.legal_concepts)
        
        if not concepts1 or not concepts2:
            return 0.0
        
        intersection = len(concepts1.intersection(concepts2))
        union = len(concepts1.union(concepts2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_citation_similarity(self, profile1: CaseProfile, profile2: CaseProfile) -> float:
        """Calculate citation similarity between two case profiles"""
        citations1 = set(profile1.legal_citations)
        citations2 = set(profile2.legal_citations)
        
        if not citations1 or not citations2:
            return 0.0
        
        intersection = len(citations1.intersection(citations2))
        union = len(citations1.union(citations2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_topic_similarity(self, profile1: CaseProfile, profile2: CaseProfile) -> float:
        """Calculate topic similarity between two case profiles"""
        if profile1.topic_vector is None or profile2.topic_vector is None:
            return 0.0
        
        try:
            similarity = cosine_similarity(
                profile1.topic_vector.reshape(1, -1),
                profile2.topic_vector.reshape(1, -1)
            )[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _determine_similarity_type(self, content_sim: float, concept_sim: float, citation_sim: float) -> str:
        """Determine the type of similarity between cases"""
        if citation_sim > 0.5:
            return "citation_based"
        elif concept_sim > 0.7:
            return "concept_based"
        elif content_sim > 0.6:
            return "content_based"
        else:
            return "mixed"
    
    def _get_shared_concepts(self, profile1: CaseProfile, profile2: CaseProfile) -> List[str]:
        """Get shared legal concepts between two profiles"""
        concepts1 = set(profile1.legal_concepts)
        concepts2 = set(profile2.legal_concepts)
        return list(concepts1.intersection(concepts2))
    
    def _get_shared_citations(self, profile1: CaseProfile, profile2: CaseProfile) -> List[str]:
        """Get shared legal citations between two profiles"""
        citations1 = set(profile1.legal_citations)
        citations2 = set(profile2.legal_citations)
        return list(citations1.intersection(citations2))
    
    def get_case_recommendations(self, case_id: str, recommendation_type: str = "similar") -> List[SimilarCase]:
        """Get case recommendations based on different criteria"""
        if case_id not in self.case_profiles:
            return []
        
        target_profile = self.case_profiles[case_id]
        
        if recommendation_type == "similar":
            return self.find_similar_cases(case_id, top_k=10)
        elif recommendation_type == "same_area":
            return self._get_cases_by_legal_area(target_profile.legal_area, case_id)
        elif recommendation_type == "same_ponente":
            return self._get_cases_by_ponente(target_profile.ponente, case_id)
        elif recommendation_type == "same_year":
            return self._get_cases_by_year(target_profile.year, case_id)
        else:
            return self.find_similar_cases(case_id, top_k=10)
    
    def _get_cases_by_legal_area(self, legal_area: str, exclude_id: str) -> List[SimilarCase]:
        """Get cases from the same legal area"""
        similar_cases = []
        
        for case_id, profile in self.case_profiles.items():
            if case_id == exclude_id or profile.legal_area != legal_area:
                continue
            
            similar_case = SimilarCase(
                case_id=case_id,
                title=profile.title,
                gr_number=profile.gr_number,
                year=profile.year,
                similarity_score=0.8,  # High score for same area
                similarity_type="legal_area",
                shared_concepts=[],
                shared_citations=[],
                legal_area=profile.legal_area,
                court_division=profile.court_division,
                ponente=profile.ponente
            )
            similar_cases.append(similar_case)
        
        return similar_cases[:10]
    
    def _get_cases_by_ponente(self, ponente: str, exclude_id: str) -> List[SimilarCase]:
        """Get cases by the same ponente"""
        similar_cases = []
        
        for case_id, profile in self.case_profiles.items():
            if case_id == exclude_id or profile.ponente != ponente:
                continue
            
            similar_case = SimilarCase(
                case_id=case_id,
                title=profile.title,
                gr_number=profile.gr_number,
                year=profile.year,
                similarity_score=0.7,  # High score for same ponente
                similarity_type="ponente",
                shared_concepts=[],
                shared_citations=[],
                legal_area=profile.legal_area,
                court_division=profile.court_division,
                ponente=profile.ponente
            )
            similar_cases.append(similar_case)
        
        return similar_cases[:10]
    
    def _get_cases_by_year(self, year: int, exclude_id: str) -> List[SimilarCase]:
        """Get cases from the same year"""
        similar_cases = []
        
        for case_id, profile in self.case_profiles.items():
            if case_id == exclude_id or profile.year != year:
                continue
            
            similar_case = SimilarCase(
                case_id=case_id,
                title=profile.title,
                gr_number=profile.gr_number,
                year=profile.year,
                similarity_score=0.6,  # Medium score for same year
                similarity_type="temporal",
                shared_concepts=[],
                shared_citations=[],
                legal_area=profile.legal_area,
                court_division=profile.court_division,
                ponente=profile.ponente
            )
            similar_cases.append(similar_case)
        
        return similar_cases[:10]
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get statistics about the similarity engine"""
        if not self.case_profiles:
            return {}
        
        # Count by legal area
        area_counts = Counter(profile.legal_area for profile in self.case_profiles.values())
        
        # Count by year
        year_counts = Counter(profile.year for profile in self.case_profiles.values())
        
        # Count by ponente
        ponente_counts = Counter(profile.ponente for profile in self.case_profiles.values())
        
        return {
            'total_cases': len(self.case_profiles),
            'legal_areas': dict(area_counts),
            'years': dict(year_counts),
            'ponentes': dict(ponente_counts.most_common(10)),
            'model_loaded': self.lda_model is not None,
            'vectorizer_fitted': hasattr(self.vectorizer, 'vocabulary_')
        }
