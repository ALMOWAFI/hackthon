#!/usr/bin/env python3
"""
Advanced Paper Grading System

This module provides a comprehensive paper grading system that simulates 
how an experienced teacher would mark and provide feedback on academic papers.
It follows a four-phase assessment process:
1. Initial paper markup with annotation symbols
2. Detailed feedback document
3. Rubric-based assessment
4. Summary feedback and next steps
"""

import os
import re
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance
from collections import defaultdict

# Ensure NLTK dependencies are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class PaperGradingSystem:
    """
    Implements a comprehensive paper grading system that replicates
    the paper grading process of an experienced teacher.
    """
    
    def __init__(self, subject_area="general", education_level="high_school"):
        """
        Initialize the paper grading system.
        
        Args:
            subject_area (str): The academic subject area (STEM, humanities, etc.)
            education_level (str): Educational level (elementary, high_school, etc.)
        """
        self.subject_area = subject_area
        self.education_level = education_level
        self.markup_symbols = self._initialize_markup_symbols()
        self.grammar_patterns = self._initialize_grammar_patterns()
        self.spelling_dictionary = self._initialize_spelling_dictionary()
        
    def _initialize_markup_symbols(self):
        """Define standard markup symbols used by teachers."""
        return {
            "grammar_error": "underline",
            "spelling_error": "circle", 
            "awkward_phrasing": "wavy_underline",
            "unclear_section": "question_mark",
            "formatting_issue": "box",
            "strong_point": "checkmark",
            "exceptional_point": "double_checkmark",
            "redundant": "strikethrough",
            "margin_comment": "note",
            "citation_error": "double_underline"
        }
        
    def _initialize_grammar_patterns(self):
        """Define patterns for common grammar errors."""
        return {
            "subject_verb_agreement": re.compile(r'\b(they|he|she|it)\s+(are|am|were|was)\b', re.IGNORECASE),
            "run_on_sentence": re.compile(r'[^.!?;]+\s+and\s+[^.!?;]+\s+and\s+[^.!?;]+'),
            "sentence_fragment": re.compile(r'^\s*[A-Z][^.!?]*(?:[.!?]|\s*$)'),
            "double_negative": re.compile(r'\b(?:not|n\'t)\b.*\b(?:no|none|nobody|nothing|nowhere|never|neither|nor)\b'),
            # Additional patterns would be added
        }
        
    def _initialize_spelling_dictionary(self):
        """Initialize common word spellings."""
        # In a full implementation, this would load a comprehensive dictionary
        # For now, we'll use WordNet as a basic dictionary
        return set(word.lower() for word in wordnet.words())
    
    def mark_paper(self, paper_text):
        """
        Phase 1: Generate markup annotations for the paper.
        
        Args:
            paper_text (str): The text content of the paper
            
        Returns:
            dict: Markup annotations with positions and types
        """
        paragraphs = paper_text.split('\n\n')
        markup = {
            "annotations": [],
            "paragraph_indices": self._paragraph_indices(paper_text)
        }
        
        # Process each paragraph
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = sent_tokenize(paragraph)
            
            # Check each sentence
            for sent_idx, sentence in enumerate(sentences):
                # Mark grammar errors
                grammar_issues = self._check_grammar(sentence)
                for issue in grammar_issues:
                    markup["annotations"].append({
                        "type": "grammar_error",
                        "paragraph": para_idx,
                        "sentence": sent_idx,
                        "start": issue["start"],
                        "end": issue["end"],
                        "description": issue["description"],
                        "symbol": self.markup_symbols["grammar_error"]
                    })
                
                # Mark spelling errors
                spelling_issues = self._check_spelling(sentence)
                for issue in spelling_issues:
                    markup["annotations"].append({
                        "type": "spelling_error",
                        "paragraph": para_idx,
                        "sentence": sent_idx,
                        "start": issue["start"],
                        "end": issue["end"],
                        "word": issue["word"],
                        "symbol": self.markup_symbols["spelling_error"]
                    })
                    
                # Check for awkward phrasing
                awkward_phrases = self._check_phrasing(sentence)
                for phrase in awkward_phrases:
                    markup["annotations"].append({
                        "type": "awkward_phrasing",
                        "paragraph": para_idx,
                        "sentence": sent_idx,
                        "start": phrase["start"],
                        "end": phrase["end"],
                        "text": phrase["text"],
                        "symbol": self.markup_symbols["awkward_phrasing"]
                    })
                    
            # Check paragraph-level issues
            structure_issues = self._check_paragraph_structure(paragraph)
            for issue in structure_issues:
                markup["annotations"].append({
                    "type": issue["type"],
                    "paragraph": para_idx,
                    "description": issue["description"],
                    "symbol": self.markup_symbols[issue["type"]]
                })
        
        # Identify strengths
        strengths = self._identify_strengths(paper_text)
        for strength in strengths:
            markup["annotations"].append({
                "type": strength["type"],
                "paragraph": strength["paragraph"],
                "sentence": strength.get("sentence"),
                "start": strength.get("start"),
                "end": strength.get("end"),
                "description": strength["description"],
                "symbol": self.markup_symbols[strength["type"]]
            })
            
        return markup
    
    def _paragraph_indices(self, text):
        """Calculate the start and end indices of each paragraph in the text."""
        indices = []
        start = 0
        for paragraph in text.split('\n\n'):
            end = start + len(paragraph)
            indices.append({"start": start, "end": end})
            start = end + 2  # +2 for the '\n\n'
        return indices
    
    def _check_grammar(self, sentence):
        """Check for grammar issues in a sentence."""
        issues = []
        
        # Check for subject-verb agreement
        for match in self.grammar_patterns["subject_verb_agreement"].finditer(sentence):
            issues.append({
                "start": match.start(),
                "end": match.end(),
                "description": "Subject-verb agreement error"
            })
            
        # Check for run-on sentences
        if self.grammar_patterns["run_on_sentence"].search(sentence):
            issues.append({
                "start": 0,
                "end": len(sentence),
                "description": "Run-on sentence"
            })
            
        # Additional grammar checks would be implemented here
        
        return issues
    
    def _check_spelling(self, sentence):
        """Check for spelling errors in a sentence."""
        issues = []
        words = word_tokenize(sentence)
        
        # Track character position in the sentence
        char_pos = 0
        for word in words:
            # Skip punctuation and other non-alphabetic tokens
            if not word.isalpha():
                char_pos += len(word) + 1  # +1 for space
                continue
                
            # Check if word is in dictionary
            if word.lower() not in self.spelling_dictionary:
                # Find actual position in original sentence
                word_start = sentence[char_pos:].find(word) + char_pos
                if word_start >= 0:
                    issues.append({
                        "start": word_start,
                        "end": word_start + len(word),
                        "word": word
                    })
            
            char_pos += len(word) + 1  # +1 for space
            
        return issues
    
    def _check_phrasing(self, sentence):
        """Identify awkward phrasing in a sentence."""
        awkward_phrases = []
        
        # This would use NLP techniques to identify awkward phrasing
        # Simplified approach for demonstration
        potential_awkward = [
            "in order to", "due to the fact that", "in regards to",
            "as to", "for the purpose of", "in the event that"
        ]
        
        for phrase in potential_awkward:
            if phrase in sentence.lower():
                start = sentence.lower().find(phrase)
                awkward_phrases.append({
                    "start": start,
                    "end": start + len(phrase),
                    "text": sentence[start:start + len(phrase)]
                })
                
        return awkward_phrases
    
    def _check_paragraph_structure(self, paragraph):
        """Check for structural issues in a paragraph."""
        issues = []
        
        # Check for very short paragraphs
        if len(paragraph.split()) < 3:
            issues.append({
                "type": "formatting_issue",
                "description": "Paragraph too short"
            })
            
        # Check for very long paragraphs
        elif len(paragraph.split()) > 200:
            issues.append({
                "type": "formatting_issue",
                "description": "Paragraph too long"
            })
            
        # Check for missing topic sentence
        sentences = sent_tokenize(paragraph)
        if len(sentences) > 1:
            first_sentence = sentences[0].lower()
            if any(word in first_sentence for word in ["it", "this", "that", "they", "these"]):
                if not any(word in first_sentence for word in ["is", "are", "means", "shows", "demonstrates", "illustrates"]):
                    issues.append({
                        "type": "unclear_section",
                        "description": "Unclear topic sentence"
                    })
                    
        return issues
    
    def _identify_strengths(self, text):
        """Identify strong points in the paper."""
        strengths = []
        
        # This would use NLP to identify strengths
        # For demonstration, we'll use simple heuristics
        paragraphs = text.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = sent_tokenize(paragraph)
            
            # Look for well-structured topic sentences
            if len(sentences) > 0:
                first_sent = sentences[0]
                if len(first_sent.split()) > 5 and len(first_sent.split()) < 25:
                    if not any(x in first_sent.lower() for x in ["i think", "i believe", "in my opinion"]):
                        strengths.append({
                            "type": "strong_point",
                            "paragraph": para_idx,
                            "sentence": 0,
                            "description": "Clear topic sentence"
                        })
            
            # Look for good transitions between paragraphs
            if para_idx > 0 and len(sentences) > 0:
                transition_words = ["therefore", "consequently", "furthermore", "however", "in addition", "similarly"]
                first_sent = sentences[0].lower()
                if any(word in first_sent for word in transition_words):
                    strengths.append({
                        "type": "strong_point",
                        "paragraph": para_idx,
                        "sentence": 0,
                        "description": "Effective transition"
                    })
                    
        return strengths
        
    def generate_detailed_feedback(self, paper_text, markup):
        """
        Phase 2: Create a detailed feedback document based on the markup.
        
        Args:
            paper_text (str): The original paper text
            markup (dict): The markup annotations from Phase 1
            
        Returns:
            dict: Structured detailed feedback
        """
        paragraphs = paper_text.split('\n\n')
        annotations = markup["annotations"]
        
        # Organize annotations by paragraph
        para_annotations = defaultdict(list)
        for annotation in annotations:
            para_annotations[annotation["paragraph"]].append(annotation)
        
        # Generate section-by-section analysis
        section_analysis = []
        error_patterns = defaultdict(list)
        strengths_analysis = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_feedback = {
                "paragraph_number": para_idx + 1,
                "text_excerpt": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph,
                "issues": [],
                "strengths": []
            }
            
            # Process annotations for this paragraph
            for annotation in para_annotations[para_idx]:
                if annotation["type"] in ["grammar_error", "spelling_error", "awkward_phrasing", 
                                       "unclear_section", "formatting_issue", "citation_error"]:
                    # Add to paragraph issues
                    issue = {
                        "type": annotation["type"],
                        "description": annotation.get("description", "")
                    }
                    
                    # Add specific text examples if available
                    if "start" in annotation and "end" in annotation:
                        para_start = markup["paragraph_indices"][para_idx]["start"]
                        absolute_start = para_start + annotation["start"]
                        absolute_end = para_start + annotation["end"]
                        issue["text"] = paper_text[absolute_start:absolute_end]
                    
                    para_feedback["issues"].append(issue)
                    
                    # Track error patterns
                    error_type = annotation["type"]
                    if "description" in annotation:
                        error_patterns[error_type].append({
                            "description": annotation["description"],
                            "example": issue.get("text", "")
                        })
                
                elif annotation["type"] in ["strong_point", "exceptional_point"]:
                    # Add to paragraph strengths
                    para_feedback["strengths"].append({
                        "type": annotation["type"],
                        "description": annotation.get("description", "")
                    })
                    
                    # Track strengths for overall analysis
                    strengths_analysis.append({
                        "paragraph": para_idx + 1,
                        "description": annotation.get("description", "")
                    })
            
            section_analysis.append(para_feedback)
        
        # Summarize error patterns
        error_summary = []
        for error_type, errors in error_patterns.items():
            if errors:  # Only include error types that were found
                summary = {
                    "type": error_type,
                    "frequency": len(errors),
                    "description": self._get_error_description(error_type),
                    "examples": [e["example"] for e in errors[:3] if e["example"]],  # Up to 3 examples
                    "improvement_suggestions": self._get_improvement_suggestions(error_type)
                }
                error_summary.append(summary)
        
        # Generate complete detailed feedback
        detailed_feedback = {
            "section_analysis": section_analysis,
            "error_patterns": error_summary,
            "strengths": strengths_analysis,
            "improvement_suggestions": self._generate_improvement_suggestions(error_patterns.keys())
        }
        
        return detailed_feedback
    
    def _get_error_description(self, error_type):
        """Get standardized description for an error type."""
        descriptions = {
            "grammar_error": "Grammatical errors affect sentence structure and can impede understanding.",
            "spelling_error": "Spelling errors can distract the reader and appear unprofessional.",
            "awkward_phrasing": "Awkward phrasing makes your writing less clear and readable.",
            "unclear_section": "Unclear sections leave the reader confused about your meaning or intent.",
            "formatting_issue": "Formatting issues affect the presentation and organization of your work.",
            "citation_error": "Citation errors may constitute academic dishonesty and fail to give proper credit."
        }
        return descriptions.get(error_type, "This type of error affects the quality of your work.")
    
    def _get_improvement_suggestions(self, error_type):
        """Get improvement suggestions for an error type."""
        suggestions = {
            "grammar_error": ["Review basic grammar rules", "Read your work aloud to catch errors", 
                           "Use grammar checking tools as a first step"],
            "spelling_error": ["Use spell check", "Create a personal list of words you commonly misspell", 
                             "Proofread carefully after writing"],
            "awkward_phrasing": ["Simplify complex sentences", "Read your work aloud", 
                               "Ask someone else to read your work and mark confusing sections"],
            "unclear_section": ["Start with a clear topic sentence", "Use concrete examples", 
                              "Connect ideas explicitly"],
            "formatting_issue": ["Follow the required style guide", "Use consistent formatting throughout", 
                               "Break long paragraphs into smaller units"],
            "citation_error": ["Consult a style guide for proper citation format", 
                            "Take careful notes when researching", 
                            "Use citation management software"]
        }
        return suggestions.get(error_type, ["Review relevant guidelines", "Practice regularly", "Seek feedback"])
    
    def _generate_improvement_suggestions(self, error_types):
        """Generate overall improvement suggestions based on error types found."""
        suggestions = []
        
        if "grammar_error" in error_types or "spelling_error" in error_types:
            suggestions.append({
                "focus_area": "Mechanical Accuracy",
                "suggestions": [
                    "Set aside dedicated proofreading time after writing",
                    "Read your work backwards sentence by sentence to focus on mechanics",
                    "Consider working with a writing tutor or using a grammar checking tool"
                ]
            })
            
        if "awkward_phrasing" in error_types or "unclear_section" in error_types:
            suggestions.append({
                "focus_area": "Clarity and Readability",
                "suggestions": [
                    "Read your work aloud to identify awkward phrasing",
                    "Ask someone else to read your work and identify confusing sections",
                    "Simplify complex sentences and use more precise vocabulary"
                ]
            })
            
        if "formatting_issue" in error_types:
            suggestions.append({
                "focus_area": "Document Structure",
                "suggestions": [
                    "Review the assignment guidelines or style guide",
                    "Create an outline before writing to organize your thoughts",
                    "Use section headings and transitions to guide the reader"
                ]
            })
            
        # Add general suggestions if we don't have specific ones
        if not suggestions:
            suggestions.append({
                "focus_area": "General Writing Improvement",
                "suggestions": [
                    "Read exemplary works in your field to absorb effective writing patterns",
                    "Practice regular writing with focused revision",
                    "Seek feedback from peers and instructors"
                ]
            })
            
        return suggestions
        
    def apply_rubric(self, paper_text, markup, rubric=None):
        """
        Phase 3: Apply a rubric to evaluate the paper and provide scores.
        
        Args:
            paper_text (str): The original paper text
            markup (dict): The markup annotations from Phase 1
            rubric (dict): Optional rubric with criteria. If None, a default rubric is used.
            
        Returns:
            dict: Rubric assessment with scores and justifications
        """
        # Use default rubric if none provided
        if rubric is None:
            rubric = self._get_default_rubric()
            
        # Count annotation types
        annotation_counts = defaultdict(int)
        for annotation in markup["annotations"]:
            annotation_counts[annotation["type"]] += 1
            
        # Calculate total paragraphs and words
        paragraphs = paper_text.split('\n\n')
        total_paragraphs = len(paragraphs)
        words = paper_text.split()
        total_words = len(words)
        
        # Evaluate each rubric criterion
        criteria_scores = []
        total_points = 0
        total_possible = 0
        
        for criterion in rubric["criteria"]:
            criterion_id = criterion["id"]
            max_points = criterion["max_points"]
            total_possible += max_points
            
            # Determine score based on criterion type and annotation counts
            score, justification = self._score_criterion(
                criterion_id, 
                max_points,
                annotation_counts,
                total_paragraphs,
                total_words
            )
            
            # Record the score and justification
            criteria_scores.append({
                "criterion_id": criterion_id,
                "criterion_name": criterion["name"],
                "description": criterion["description"],
                "max_points": max_points,
                "score": score,
                "justification": justification
            })
            
            total_points += score
        
        # Calculate final grade
        percentage = (total_points / total_possible) * 100 if total_possible > 0 else 0
        letter_grade = self._calculate_letter_grade(percentage)
        
        # Create the assessment
        assessment = {
            "rubric_name": rubric["name"],
            "criteria_scores": criteria_scores,
            "total_points": total_points,
            "total_possible": total_possible,
            "percentage": percentage,
            "letter_grade": letter_grade,
            "overall_assessment": self._generate_overall_assessment(
                percentage, criteria_scores, annotation_counts
            )
        }
        
        return assessment
    
    def _get_default_rubric(self):
        """Get a default rubric based on subject area and education level."""
        # Basic rubric template
        rubric = {
            "name": f"{self.subject_area.capitalize()} Paper Rubric - {self.education_level.capitalize()} Level",
            "criteria": [
                {
                    "id": "content",
                    "name": "Content & Development",
                    "description": "The depth, breadth, and relevance of ideas presented",
                    "max_points": 30
                },
                {
                    "id": "organization",
                    "name": "Organization & Structure",
                    "description": "The logical flow and effective structuring of ideas",
                    "max_points": 25
                },
                {
                    "id": "language",
                    "name": "Language & Style",
                    "description": "Grammar, spelling, sentence structure, and word choice",
                    "max_points": 25
                },
                {
                    "id": "formatting",
                    "name": "Formatting & Presentation",
                    "description": "Adherence to style guidelines and visual presentation",
                    "max_points": 10
                },
                {
                    "id": "critical_thinking",
                    "name": "Critical Thinking",
                    "description": "Depth of analysis, synthesis, and evaluation",
                    "max_points": 10
                }
            ]
        }
        
        # Subject-specific adjustments
        if self.subject_area == "STEM":
            # Modify criteria for STEM papers
            for criterion in rubric["criteria"]:
                if criterion["id"] == "content":
                    criterion["description"] = "Technical accuracy and application of concepts"
                elif criterion["id"] == "critical_thinking":
                    criterion["description"] = "Problem-solving methodology and logical reasoning"
                    criterion["max_points"] = 15  # Increase weight for STEM
                elif criterion["id"] == "language":
                    criterion["max_points"] = 20  # Decrease slightly for STEM
        
        elif self.subject_area == "humanities":
            # Modify criteria for humanities papers
            for criterion in rubric["criteria"]:
                if criterion["id"] == "content":
                    criterion["description"] = "Depth of argument and use of evidence"
                elif criterion["id"] == "critical_thinking":
                    criterion["description"] = "Analysis depth and interpretive skill"
                    criterion["max_points"] = 15  # Increase for humanities
        
        # Education level adjustments
        if self.education_level == "elementary":
            # Simplify for younger students
            for criterion in rubric["criteria"]:
                if criterion["id"] == "critical_thinking":
                    criterion["max_points"] = 5  # Less emphasis for elementary
                elif criterion["id"] == "content":
                    criterion["max_points"] = 35  # More emphasis on basic content
        
        elif self.education_level == "graduate":
            # Increase expectations for graduate level
            for criterion in rubric["criteria"]:
                if criterion["id"] == "critical_thinking":
                    criterion["max_points"] = 20  # Much higher for graduate level
                elif criterion["id"] == "content":
                    criterion["description"] += " and contribution to the field"
                
        return rubric
    
    def _score_criterion(self, criterion_id, max_points, annotation_counts, total_paragraphs, total_words):
        """Score a specific rubric criterion based on annotations."""
        
        if criterion_id == "content":
            # Content scoring based on length and issues
            content_score = max_points
            
            # Adjust for paper length
            if total_words < 250:  # Very short
                content_score -= max_points * 0.3
                justification = "Paper is quite short, limiting content development."
            elif total_words < 500:  # Short
                content_score -= max_points * 0.1
                justification = "Paper could be developed further with additional content."
            else:
                justification = "Paper has sufficient length for content development."
                
            # Adjust for unclear sections
            unclear_count = annotation_counts["unclear_section"]
            if unclear_count > 5:
                content_score -= max_points * 0.3
                justification += " Several unclear sections limit content effectiveness."
            elif unclear_count > 2:
                content_score -= max_points * 0.15
                justification += " Some unclear sections impact content clarity."
                
            # Ensure score is in bounds
            content_score = max(0, min(content_score, max_points))
            return content_score, justification
            
        elif criterion_id == "organization":
            # Organization scoring based on structure issues
            org_score = max_points
            
            # Check for structure
            if total_paragraphs < 3:  # Too few paragraphs
                org_score -= max_points * 0.2
                justification = "Paper has limited paragraph structure."
            else:
                justification = "Paper has adequate paragraph structure."
                
            # Adjust for formatting issues
            format_count = annotation_counts["formatting_issue"]
            if format_count > 3:
                org_score -= max_points * 0.2
                justification += " Multiple formatting issues affect organization."
            elif format_count > 1:
                org_score -= max_points * 0.1
                justification += " Some formatting issues affect organization."
                
            # Ensure score is in bounds
            org_score = max(0, min(org_score, max_points))
            return org_score, justification
            
        elif criterion_id == "language":
            # Language scoring based on grammar and style issues
            lang_score = max_points
            
            # Count grammatical errors
            grammar_count = annotation_counts["grammar_error"]
            spelling_count = annotation_counts["spelling_error"]
            awkward_count = annotation_counts["awkward_phrasing"]
            
            total_language_issues = grammar_count + spelling_count + awkward_count
            
            # Normalize by paper length (issues per 500 words)
            normalized_issues = (total_language_issues / total_words) * 500 if total_words > 0 else 0
            
            if normalized_issues > 10:  # Many issues
                lang_score -= max_points * 0.4
                justification = "Numerous grammatical, spelling, and phrasing issues."
            elif normalized_issues > 5:  # Moderate issues
                lang_score -= max_points * 0.2
                justification = "Several grammatical, spelling, and phrasing issues throughout."
            elif normalized_issues > 2:  # Few issues
                lang_score -= max_points * 0.1
                justification = "Some grammatical, spelling, and phrasing issues."
            else:  # Very few issues
                justification = "Very few language issues; generally well-written."
                
            # Ensure score is in bounds
            lang_score = max(0, min(lang_score, max_points))
            return lang_score, justification
            
        elif criterion_id == "formatting":
            # Formatting scoring
            format_score = max_points
            format_count = annotation_counts["formatting_issue"]
            
            if format_count > 5:  # Many issues
                format_score -= max_points * 0.7
                justification = "Significant formatting issues throughout."
            elif format_count > 3:  # Several issues
                format_score -= max_points * 0.4
                justification = "Several formatting issues affect presentation."
            elif format_count > 1:  # Few issues
                format_score -= max_points * 0.2
                justification = "A few formatting issues present."
            else:  # Very few issues
                justification = "Formatting is generally consistent and appropriate."
                
            # Ensure score is in bounds
            format_score = max(0, min(format_score, max_points))
            return format_score, justification
            
        elif criterion_id == "critical_thinking":
            # Critical thinking is more subjective
            # Use ratio of strong points to total annotations as a proxy
            strong_points = annotation_counts["strong_point"] + annotation_counts["exceptional_point"]
            total_annotations = sum(annotation_counts.values())
            
            if total_annotations == 0:
                ratio = 0.5  # Default if no annotations
            else:
                ratio = strong_points / total_annotations
                
            # Score based on ratio of strong points
            if ratio > 0.3:  # Many strong points
                critical_score = max_points * 0.9
                justification = "Demonstrates strong critical thinking and analysis."
            elif ratio > 0.2:  # Good amount
                critical_score = max_points * 0.7
                justification = "Shows good critical thinking in several areas."
            elif ratio > 0.1:  # Some
                critical_score = max_points * 0.5
                justification = "Demonstrates some critical thinking but could be developed further."
            else:  # Few
                critical_score = max_points * 0.3
                justification = "Limited evidence of critical thinking and analysis."
                
            # Ensure score is in bounds
            critical_score = max(0, min(critical_score, max_points))
            return critical_score, justification
            
        # Default case
        return max_points * 0.6, "Score based on general assessment of this criterion."
    
    def _calculate_letter_grade(self, percentage):
        """Convert a percentage score to a letter grade."""
        if percentage >= 97:
            return "A+"
        elif percentage >= 93:
            return "A"
        elif percentage >= 90:
            return "A-"
        elif percentage >= 87:
            return "B+"
        elif percentage >= 83:
            return "B"
        elif percentage >= 80:
            return "B-"
        elif percentage >= 77:
            return "C+"
        elif percentage >= 73:
            return "C"
        elif percentage >= 70:
            return "C-"
        elif percentage >= 67:
            return "D+"
        elif percentage >= 63:
            return "D"
        elif percentage >= 60:
            return "D-"
        else:
            return "F"
    
    def _generate_overall_assessment(self, percentage, criteria_scores, annotation_counts):
        """Generate an overall assessment summary."""
        
        # Identify strongest and weakest areas
        sorted_scores = sorted(criteria_scores, key=lambda x: x["score"] / x["max_points"])
        weakest = sorted_scores[0]
        strongest = sorted_scores[-1]
        
        # Generate assessment based on overall score
        if percentage >= 90:
            quality = "excellent"
            tone = "This is a strong paper that "
        elif percentage >= 80:
            quality = "good"
            tone = "This is a solid paper that "
        elif percentage >= 70:
            quality = "satisfactory"
            tone = "This paper meets basic requirements and "
        elif percentage >= 60:
            quality = "marginal"
            tone = "This paper needs improvement but "
        else:
            quality = "unsatisfactory"
            tone = "This paper requires significant revision and "
            
        # Construct assessment
        assessment = f"{tone}demonstrates {quality} work in {strongest['criterion_name']}. "
        assessment += f"The primary area needing improvement is {weakest['criterion_name']}. "
        
        # Add specific detail about errors
        total_errors = (
            annotation_counts.get("grammar_error", 0) +
            annotation_counts.get("spelling_error", 0) +
            annotation_counts.get("awkward_phrasing", 0) +
            annotation_counts.get("unclear_section", 0) +
            annotation_counts.get("formatting_issue", 0)
        )
        
        if total_errors > 15:
            assessment += "There are numerous technical issues throughout the paper that should be addressed. "
        elif total_errors > 5:
            assessment += "There are several technical issues that should be addressed. "
        elif total_errors > 0:
            assessment += "There are a few technical issues to address. "
        else:
            assessment += "The paper is technically sound with minimal issues. "
            
        # Add strengths
        strong_points = annotation_counts.get("strong_point", 0)
        exceptional_points = annotation_counts.get("exceptional_point", 0)
        
        if exceptional_points > 0:
            assessment += "The paper has some exceptional elements that demonstrate advanced thinking. "
        elif strong_points > 5:
            assessment += "The paper has multiple strengths that enhance its effectiveness. "
        elif strong_points > 0:
            assessment += "The paper has some strong points that can be built upon. "
            
        return assessment
        
    def create_summary(self, detailed_feedback, rubric_assessment):
        """
        Phase 4: Create a summary feedback page with final assessment and next steps.
        
        Args:
            detailed_feedback (dict): The detailed feedback from Phase 2
            rubric_assessment (dict): The rubric assessment from Phase 3
            
        Returns:
            dict: A summary feedback document
        """
        # Extract key information
        error_patterns = detailed_feedback["error_patterns"]
        strengths = detailed_feedback["strengths"]
        improvements = detailed_feedback["improvement_suggestions"]
        
        # Get overall assessment
        overall = rubric_assessment["overall_assessment"]
        percentage = rubric_assessment["percentage"]
        letter_grade = rubric_assessment["letter_grade"]
        
        # Generate key strengths list
        key_strengths = []
        if strengths:
            # Group similar strengths
            strength_types = {}
            for strength in strengths:
                desc = strength["description"]
                if desc in strength_types:
                    strength_types[desc]["count"] += 1
                else:
                    strength_types[desc] = {"description": desc, "count": 1}
                    
            # Get top 3 strengths
            sorted_strengths = sorted(strength_types.values(), key=lambda x: x["count"], reverse=True)
            for strength in sorted_strengths[:3]:
                key_strengths.append({
                    "description": strength["description"],
                    "frequency": f"Observed {strength['count']} times"
                })
        
        # If we don't have enough strengths, add some generic ones based on highest scoring criteria
        if len(key_strengths) < 2:
            # Get highest scoring criteria
            sorted_criteria = sorted(
                rubric_assessment["criteria_scores"],
                key=lambda x: x["score"] / x["max_points"],
                reverse=True
            )
            
            for criterion in sorted_criteria[:2]:
                # Check if we already have a strength for this
                if not any(strength["description"].lower() in criterion["criterion_name"].lower() 
                       for strength in key_strengths):
                    key_strengths.append({
                        "description": f"Good {criterion['criterion_name']}",
                        "justification": criterion["justification"]
                    })
        
        # Generate key areas for improvement
        improvement_areas = []
        
        # Start with lowest scoring criteria
        sorted_criteria = sorted(
            rubric_assessment["criteria_scores"],
            key=lambda x: x["score"] / x["max_points"]
        )
        
        for criterion in sorted_criteria[:2]:  # Top 2 weakest areas
            improvement_areas.append({
                "area": criterion["criterion_name"],
                "justification": criterion["justification"],
                "score": f"{criterion['score']}/{criterion['max_points']} points"
            })
            
        # Add most frequent error patterns
        if error_patterns:
            sorted_errors = sorted(error_patterns, key=lambda x: x["frequency"], reverse=True)
            for error in sorted_errors[:2]:  # Top 2 error patterns
                if not any(area["area"].lower() in error["type"].lower() for area in improvement_areas):
                    improvement_areas.append({
                        "area": error["type"].replace("_", " ").title(),
                        "justification": error["description"],
                        "frequency": f"Observed {error['frequency']} times"
                    })
        
        # Generate actionable next steps
        next_steps = []
        
        # Add specific improvement suggestions
        for focus_area in improvements:
            next_steps.append({
                "focus": focus_area["focus_area"],
                "actions": focus_area["suggestions"]
            })
            
        # Add curriculum connections based on subject area
        if self.subject_area == "STEM":
            next_steps.append({
                "focus": "Mathematical/Scientific Precision",
                "actions": [
                    "Review technical terminology used in the paper",
                    "Ensure all equations and formulas are correctly presented",
                    "Connect concepts to broader scientific principles"
                ]
            })
        elif self.subject_area == "humanities":
            next_steps.append({
                "focus": "Critical Analysis",
                "actions": [
                    "Strengthen thesis statement with more nuanced language",
                    "Incorporate additional scholarly perspectives",
                    "Connect analysis to broader theoretical frameworks"
                ]
            })
        else:  # General
            next_steps.append({
                "focus": "Academic Rigor",
                "actions": [
                    "Strengthen connections between evidence and claims",
                    "Incorporate more specific examples to support points",
                    "Consider alternative perspectives on your topic"
                ]
            })
            
        # Create encouraging closing comment based on grade
        if percentage >= 90:
            closing = "Excellent work! Your paper demonstrates strong understanding and execution."
        elif percentage >= 80:
            closing = "Good job on this assignment. With some targeted improvements, your next paper could be excellent."
        elif percentage >= 70:
            closing = "You've met the basic requirements. Focus on the suggested improvements to strengthen your writing."
        elif percentage >= 60:
            closing = "While there's room for improvement, you've shown potential. Work on the key areas identified."
        else:
            closing = "This paper needs significant revision. Focus on the fundamentals and don't hesitate to seek additional help."
            
        # Compile the summary
        summary = {
            "final_grade": {
                "percentage": percentage,
                "letter_grade": letter_grade,
                "points": f"{rubric_assessment['total_points']}/{rubric_assessment['total_possible']}"
            },
            "overall_assessment": overall,
            "key_strengths": key_strengths,
            "improvement_areas": improvement_areas,
            "next_steps": next_steps,
            "closing_comment": closing
        }
        
        return summary
