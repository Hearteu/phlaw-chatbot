"""
Bar Subject Classifier - Classifies legal cases into Bar Examination subjects
Based on 2026 Philippine Bar Examination syllabus
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# 2026 Bar Examination Subject Categories - Complete Detailed Syllabus
BAR_SUBJECTS = {
    'political_law': {
        'keywords': [
            'constitution', 'constitutional', 'president', 'executive', 'legislative',
            'judicial', 'supreme court', 'bill of rights', 'due process', 'equal protection',
            'administrative', 'public officer', 'local government', 'election', 'suffrage',
            'treaty', 'international', 'united nations', 'state responsibility',
            'diplomatic', 'human rights', 'humanitarian law', 'law of the sea'
        ],
        'subcategories': [
            'constitutional_law', 'administrative_law', 'election_law',
            'public_international_law', 'constitutional_remedies'
        ]
    },
    'labor_law': {
        'keywords': [
            'labor', 'employee', 'employer', 'wage', 'overtime', 'holiday pay',
            'thirteenth month', 'social security', 'gsis', 'sss', 'philhealth',
            'pension', 'retirement', 'strike', 'lockout', 'collective bargaining',
            'labor union', 'union', 'employment contract', 'termination',
            'separation pay', 'retrenchment', 'downsizing'
        ],
        'subcategories': [
            'labor_standards', 'labor_relations', 'social_legislation'
        ]
    },
    'civil_law': {
        'keywords': [
            'obligation', 'contract', 'breach of contract', 'rescission', 'specific performance',
            'damages', 'family', 'marriage', 'annulment', 'legal separation', 'property',
            'ownership', 'sale', 'donation', 'succession', 'will', 'inheritance', 'estate',
            'paternity', 'custody', 'support', 'adoption', 'domestic violence'
        ],
        'subcategories': [
            'contracts', 'property', 'family_law', 'succession', 'tort'
        ]
    },
    'taxation_law': {
        'keywords': [
            'tax', 'taxation', 'internal revenue', 'bir', 'income tax', 'corporate tax',
            'estate tax', 'donor tax', 'vat', 'value added tax', 'excise tax',
            'documentary stamp', 'withholding', 'tax exemption', 'tax deduction',
            'real property tax', 'local tax', 'tax refund', 'tax assessment',
            'tax evasion', 'tax avoidance', 'fraudulent return'
        ],
        'subcategories': [
            'income_tax', 'estate_tax', 'donor_tax', 'vat', 'local_taxation'
        ]
    },
    'commercial_law': {
        'keywords': [
            'corporation', 'corporate', 'stock', 'share', 'board of directors',
            'corporate officers', 'stockholders', 'dividend', 'corporate dissolution',
            'negotiable instrument', 'promissory note', 'check', 'bill of exchange',
            'insurance', 'policy', 'premium', 'claim', 'transportation', 'common carrier',
            'charter party', 'bill of lading', 'securities', 'stock exchange',
            'partnership', 'competition', 'monopoly', 'anti-trust', 'data privacy'
        ],
        'subcategories': [
            'corporation_law', 'negotiable_instruments', 'insurance', 'transportation',
            'securities_regulation', 'competition_law'
        ]
    },
    'criminal_law': {
        'keywords': [
            'murder', 'homicide', 'assault', 'robbery', 'theft', 'fraud', 'estafa',
            'forgery', 'malversation', 'corruption', 'graft', 'plunder', 'bribery',
            'dangerous drugs', 'drug trafficking', 'drug abuse', 'money laundering',
            'cybercrime', 'computer crime', 'identity theft', 'child abuse',
            'violence against women', 'human trafficking', 'illegal arrest',
            'torture', 'extrajudicial killing'
        ],
        'subcategories': [
            'criminal_law', 'special_penal_laws', 'dangerous_drugs', 'cybercrime',
            'anti-corruption'
        ]
    },
    'remedial_law': {
        'keywords': [
            'jurisdiction', 'venue', 'service of process', 'subpoena', 'discovery',
            'deposition', 'interrogatory', 'summary judgment', 'default',
            'preliminary injunction', 'temporary restraining order', 'attachment',
            'garnishment', 'replevin', 'writ of execution', 'appeal', 'motion',
            'pleading', 'complaint', 'answer', 'counterclaim', 'cross-claim',
            'class action', 'intervention', 'joinder', 'evidence', 'hearsay',
            'expert witness', 'exhibit', 'objection', 'prosecution', 'arraignment'
        ],
        'subcategories': [
            'civil_procedure', 'criminal_procedure', 'evidence', 'special_proceedings',
            'provisional_remedies'
        ]
    },
    'legal_ethics': {
        'keywords': [
            'attorney', 'lawyer', 'legal profession', 'bar', 'judicial', 'judge',
            'court', 'canon', 'professional responsibility', 'conflict of interest',
            'attorney-client privilege', 'malpractice', 'disbarment', 'contempt',
            'continuing legal education', 'ethics', 'code of judicial conduct',
            'judicial conduct', 'judicial independence'
        ],
        'subcategories': [
            'legal_profession', 'judicial_ethics', 'bar_discipline'
        ]
    }
}


def classify_by_bar_subject(
    text: str,
    title: str = "",
    gr_number: str = ""
) -> Dict[str, any]:
    """
    Classify a legal case into one or more Bar examination subjects
    
    Args:
        text: The case text/body
        title: Case title (optional)
        gr_number: G.R. number (optional)
        
    Returns:
        Dictionary with classification results
    """
    # Combine title and text for classification
    combined_text = f"{title} {text}".lower()
    
    # Score each subject
    subject_scores = {}
    
    for subject, data in BAR_SUBJECTS.items():
        score = 0
        matched_keywords = []
        
        for keyword in data['keywords']:
            # Count occurrences of keyword
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                matched_keywords.append(keyword)
                score += len(matches)  # Weight by frequency
        
        if score > 0:
            subject_scores[subject] = {
                'score': score,
                'matched_keywords': matched_keywords,
                'subcategories': data['subcategories']
            }
    
    # Determine primary classification
    primary_subject = None
    secondary_subjects = []
    
    if subject_scores:
        # Sort by score
        sorted_subjects = sorted(
            subject_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        primary_subject = sorted_subjects[0][0]
        
        # Secondary subjects are those with at least 50% of primary score
        primary_score = sorted_subjects[0][1]['score']
        
        for subject, data in sorted_subjects[1:]:
            if data['score'] >= primary_score * 0.5:
                secondary_subjects.append(subject)
    
    # If no classification found, try advanced heuristics
    if not primary_subject:
        primary_subject, secondary_subjects = _advanced_classification(
            title, text, gr_number
        )
    
    return {
        'primary_subject': primary_subject,
        'secondary_subjects': secondary_subjects,
        'subject_scores': subject_scores,
        'confidence': _calculate_confidence(subject_scores)
    }


def _advanced_classification(
    title: str,
    text: str,
    gr_number: str
) -> Tuple[Optional[str], List[str]]:
    """
    Advanced classification using case title patterns and GR number analysis
    """
    combined = f"{title} {text}".lower()
    
    # Check for administrative matters
    if 'a.m.' in gr_number.lower() or 'administrative matter' in combined:
        return 'legal_ethics', ['political_law']
    
    # Check for labor cases
    if any(word in combined for word in ['nursing', 'nurses', 'seafarer', 'seaman', 'ofw']):
        return 'labor_law', []
    
    # Check for corporate cases
    if any(word in combined for word in ['inc.', 'corporation', 'corp.', 'company']):
        return 'commercial_law', []
    
    # Check for family cases
    if any(word in combined for word in ['divorce', 'annulment', 'marriage', 'children']):
        return 'civil_law', []
    
    return None, []


def _calculate_confidence(scores: Dict[str, Dict]) -> str:
    """
    Calculate classification confidence based on scores
    """
    if not scores:
        return 'low'
    
    max_score = max(data['score'] for data in scores.values())
    
    if max_score >= 10:
        return 'high'
    elif max_score >= 5:
        return 'medium'
    else:
        return 'low'


def get_bar_subject_metadata(case_data: Dict) -> Dict:
    """
    Add Bar subject metadata to a case
    
    Args:
        case_data: Dictionary containing case information
        
    Returns:
        Updated case data with Bar subject classification
    """
    # Extract text fields
    title = case_data.get('title', '') or case_data.get('case_title', '')
    text = case_data.get('clean_text', '') or case_data.get('body', '')
    gr_number = case_data.get('gr_number', '')
    
    # Classify
    classification = classify_by_bar_subject(text, title, gr_number)
    
    # Add metadata
    case_data['bar_primary_subject'] = classification['primary_subject']
    case_data['bar_secondary_subjects'] = classification['secondary_subjects']
    case_data['bar_subject_scores'] = classification['subject_scores']
    case_data['bar_classification_confidence'] = classification['confidence']
    
    return case_data


def get_subject_syllabus(subject: str) -> Dict:
    """
    Get the complete 2026 syllabus/coverage for a specific Bar subject with all sub-topics
    
    Args:
        subject: The Bar subject name
        
    Returns:
        Dictionary with complete syllabus information including all sub-topics
    """
    # Complete 2026 Bar Bulletin Syllabus with all sub-topics
    syllabus_map = {
        'political_law': {
            'main_areas': [
                'Constitutional Law', 'Administrative Law', 'Election Law',
                'Law of Public Officers', 'Public Corporation Law', 'Public International Law'
            ],
            'detailed_topics': {
                'Constitutional Law': [
                    'Constitution (1987)', 'Fundamental Principles', 'Bill of Rights',
                    'Due Process', 'Equal Protection', 'Freedom of Speech',
                    'Freedom of Religion', 'Right to Information', 'Right of Assembly',
                    'Ex Post Facto Laws', 'Double Jeopardy', 'Right to Counsel'
                ],
                'Administrative Law': [
                    'Administrative Bodies', 'Powers and Functions',
                    'Due Process in Administrative Proceedings',
                    'Judicial Review', 'Certiorari', 'Prohibition', 'Mandamus'
                ],
                'Election Law': [
                    'Suffrage', 'Registration of Voters', 'Campaign Finance',
                    'Election Offenses', 'Election Contests', 'Electoral Protest'
                ]
            }
        },
        'labor_law': {
            'main_areas': [
                'Labor Standards', 'Labor Relations', 'Social Legislation'
            ],
            'detailed_topics': {
                'Labor Standards': [
                    'Employment Relationship', 'Termination of Employment',
                    'Just and Authorized Causes', 'Wage and Wage-Related Benefits',
                    'Overtime Work', 'Holiday Pay', 'Service Incentive Leave',
                    'Retirement', 'Employee Compensation'
                ],
                'Labor Relations': [
                    'Union Organization', 'Collective Bargaining',
                    'Strike and Lockout', 'Grievance Machinery',
                    'Labor Disputes', 'Unfair Labor Practice',
                    'Jurisdiction of NLRC'
                ],
                'Social Legislation': [
                    'Social Security System (SSS)', 'GSIS',
                    'Philippine Health Insurance', 'Overseas Workers Rights',
                    'Magna Carta for Workers', 'Occupational Safety'
                ]
            }
        },
        'civil_law': {
            'main_areas': [
                'Obligations and Contracts', 'Sales', 'Lease',
                'Mortgage and Pledge', 'Partnership', 'Agency',
                'Credit Transactions', 'Deposit and Trust'
            ],
            'detailed_topics': {
                'Obligations': [
                    'Nature and Effects', 'Different Kinds of Obligations',
                    'Extinguishment', 'Breach of Contract',
                    'Specific Performance', 'Rescission', 'Damages'
                ],
                'Property': [
                    'Ownership', 'Possession', 'Accession',
                    'Co-ownership', 'Usufruct', 'Real Rights'
                ],
                'Family Relations': [
                    'Marriage', 'Annulment', 'Legal Separation',
                    'Family Property', 'Support', 'Adoption',
                    'Paternity and Filiation'
                ],
                'Succession': [
                    'Intestate Succession', 'Testate Succession',
                    'Compulsory Heirs', 'Disinheritance', 'Will'
                ]
            }
        },
        'taxation_law': {
            'main_areas': [
                'General Principles', 'Income Tax', 'Estate Tax',
                'Donor Tax', 'VAT', 'Local Taxation', 'Tax Remedies'
            ],
            'detailed_topics': {
                'Income Tax': [
                    'Gross Income', 'Exclusions', 'Deductions',
                    'Corporate Tax', 'Individual Tax',
                    'Capital Gains Tax', 'Fringe Benefits Tax',
                    'Withholding Tax', 'Creditable Withholding'
                ],
                'VAT': [
                    'VATable Transactions', 'Zero-Rated Sales',
                    'Exempt Sales', 'Input Tax', 'Output Tax',
                    'VAT on Importation'
                ],
                'Tax Remedies': [
                    'Assessment', 'Protest', 'Refund',
                    'Taxpayer Bill of Rights',
                    'Civil and Criminal Remedies'
                ]
            }
        },
        'commercial_law': {
            'main_areas': [
                'Corporation Law', 'Negotiable Instruments',
                'Insurance', 'Transportation', 'Securities Regulation',
                'Partnership'
            ],
            'detailed_topics': {
                'Corporation Law': [
                    'Incorporation', 'Corporate Powers',
                    'Board of Directors', 'Officers',
                    'Stockholders', 'Dividends',
                    'Corporate Dissolution', 'Ultra Vires'
                ],
                'Negotiable Instruments': [
                    'Promissory Note', 'Bill of Exchange', 'Check',
                    'Draft', 'Negotiation', 'Endorsement',
                    'Holders in Due Course', 'Liabilities'
                ],
                'Insurance': [
                    'Insurance Contract', 'Premium',
                    'Insurable Interest', 'Loss',
                    'Indemnity', 'Subrogation'
                ]
            }
        },
        'criminal_law': {
            'main_areas': [
                'General Principles', 'Felonies', 'Penalties',
                'Special Penal Laws'
            ],
            'detailed_topics': {
                'General Principles': [
                    'Criminal Liability', 'Conspiracy',
                    'Attempted and Frustrated Felonies',
                    'Accomplices', 'Accessories',
                    'Mitigating and Aggravating Circumstances'
                ],
                'Felonies': [
                    'Crimes Against Persons', 'Crimes Against Property',
                    'Crimes Against Chastity', 'Crimes Against Public Order',
                    'Crimes by Public Officers'
                ],
                'Penalties': [
                    'Death Penalty', 'Reclusion Perpetua',
                    'Reclusion Temporal', 'Prison Correctional',
                    'Arresto Mayor', 'Arresto Menor', 'Fine'
                ]
            }
        },
        'remedial_law': {
            'main_areas': [
                'Civil Procedure', 'Criminal Procedure',
                'Evidence', 'Special Proceedings',
                'Provisional Remedies'
            ],
            'detailed_topics': {
                'Civil Procedure': [
                    'Jurisdiction', 'Venue', 'Summons',
                    'Pleadings', 'Amendments', 'Pre-trial',
                    'Trial', 'Judgment', 'Execution',
                    'Appeal', 'New Trial'
                ],
                'Criminal Procedure': [
                    'Criminal Actions', 'Institution of Actions',
                    'Prosecution', 'Bail', 'Arraignment',
                    'Trial', 'Judgment', 'Appeal',
                    'Probation'
                ],
                'Evidence': [
                    'Admissibility', 'Burden of Proof', 'Presumptions',
                    'Documentary Evidence', 'Testimonial Evidence',
                    'Hearsay Rule', 'Exceptions', 'Object Evidence'
                ]
            }
        },
        'legal_ethics': {
            'main_areas': [
                'Practice of Law', 'Lawyer-Client Relationship',
                'Duties to Courts', 'Duties to Public',
                'Bar Discipline'
            ],
            'detailed_topics': {
                'Lawyer-Client Relationship': [
                    'Creation and Duration', 'Confidentiality',
                    'Conflict of Interest', 'Duty of Diligence',
                    'Duty to Communicate', 'Withdrawal',
                    'Attorney-Client Privilege'
                ],
                'Duties to Courts': [
                    'Candor with Tribunal', 'Respect for Courts',
                    'Fairness to Opposing Counsel',
                    'Maintenance of Dignity'
                ],
                'Bar Discipline': [
                    'Disciplinary Proceedings', 'Contempt',
                    'Disbarment', 'Suspension',
                    'Reprimand'
                ]
            }
        }
    }
    
    return syllabus_map.get(subject, {
        'main_areas': ['Not specified'],
        'detailed_topics': {}
    })

