#!/usr/bin/env python3
"""
Simplified Paper Grading System Demo

This script demonstrates the advanced paper grading system with a simpler implementation
that doesn't rely on complex NLP dependencies.
"""

import os
import json
import re
from collections import defaultdict

class SimplePaperGrader:
    """A simplified version of the paper grading system for demonstration purposes."""
    
    def __init__(self, subject_area="general", education_level="high_school"):
        self.subject_area = subject_area
        self.education_level = education_level
        self.common_errors = self._initialize_error_patterns()
    
    def _initialize_error_patterns(self):
        """Initialize patterns for common writing errors."""
        return {
            "spelling": [
                ("there", "their", "Possible incorrect use of 'there'"),
                ("their", "there", "Possible incorrect use of 'their'"),
                ("your", "you're", "Possible incorrect use of 'your'"),
                ("its", "it's", "Possible incorrect use of 'its'"),
                ("affect", "effect", "Possible confusion between 'affect' and 'effect'"),
                ("occuring", "occurring", "Misspelling: 'occurring' has two 'r's"),
                ("accomodate", "accommodate", "Misspelling: 'accommodate' has two 'c's and two 'm's")
            ],
            "grammar": [
                (r"\b(is|are|was|were)\s+being\s+\w+ed\b", "Awkward passive construction"),
                (r"\b(they|he|she|it)\s+(don't|do|does|did)\b", "Subject-verb agreement error"),
                (r"\b(effect|impact|change)\s+\w+\b", "Possible incorrect use of noun as verb"),
                (r"\bthat\s+which\b", "Incorrect relative pronoun combination")
            ],
            "punctuation": [
                (r",\s*and", "Possible comma splice with conjunction"),
                (r"\w+\s+\w+\s+\w+\s*[.!?]\s*[a-z]", "Possible sentence boundary error"),
                (r"\s+[.!?]", "Space before terminal punctuation")
            ],
            "style": [
                (r"\bvery\s+\w+\b", "Vague intensifier 'very'"),
                (r"\breally\s+\w+\b", "Vague intensifier 'really'"),
                (r"\bin order to\b", "Wordy phrase 'in order to'"),
                (r"\bdue to the fact that\b", "Wordy phrase 'due to the fact that'")
            ]
        }
    
    def grade_paper(self, paper_text):
        """Grade a paper and provide detailed feedback."""
        # Phase 1: Mark the paper with annotations
        markup = self._mark_paper(paper_text)
        
        # Phase 2: Generate detailed feedback
        detailed_feedback = self._generate_feedback(paper_text, markup)
        
        # Phase 3: Apply rubric assessment
        assessment = self._apply_rubric(paper_text, markup)
        
        # Phase 4: Create summary feedback
        summary = self._create_summary(detailed_feedback, assessment)
        
        return {
            "markup": markup,
            "detailed_feedback": detailed_feedback,
            "assessment": assessment,
            "summary": summary
        }
    
    def _mark_paper(self, paper_text):
        """Mark the paper with annotations for errors and strengths."""
        # Split paper into paragraphs
        paragraphs = [p for p in paper_text.split('\n\n') if p.strip()]
        
        # Initialize markup
        markup = {
            "annotations": [],
            "paragraph_count": len(paragraphs),
            "word_count": len(paper_text.split())
        }
        
        # Check for errors in each paragraph
        for para_idx, paragraph in enumerate(paragraphs):
            # Check for spelling errors
            for error_word, correct_word, description in self.common_errors["spelling"]:
                if re.search(r'\b' + error_word + r'\b', paragraph, re.IGNORECASE):
                    markup["annotations"].append({
                        "type": "spelling_error",
                        "paragraph": para_idx,
                        "word": error_word,
                        "correction": correct_word,
                        "description": description
                    })
            
            # Check for grammar errors
            for pattern, description in self.common_errors["grammar"]:
                matches = re.finditer(pattern, paragraph, re.IGNORECASE)
                for match in matches:
                    markup["annotations"].append({
                        "type": "grammar_error",
                        "paragraph": para_idx,
                        "text": match.group(0),
                        "description": description
                    })
            
            # Check for punctuation errors
            for pattern, description in self.common_errors["punctuation"]:
                matches = re.finditer(pattern, paragraph)
                for match in matches:
                    markup["annotations"].append({
                        "type": "punctuation_error",
                        "paragraph": para_idx,
                        "text": match.group(0),
                        "description": description
                    })
            
            # Check for style issues
            for pattern, description in self.common_errors["style"]:
                matches = re.finditer(pattern, paragraph, re.IGNORECASE)
                for match in matches:
                    markup["annotations"].append({
                        "type": "style_issue",
                        "paragraph": para_idx,
                        "text": match.group(0),
                        "description": description
                    })
            
            # Identify strengths (simplified)
            # Check for good topic sentences
            sentences = paragraph.split('.')
            if len(sentences) > 0 and len(sentences[0].split()) > 5 and len(sentences[0].split()) < 30:
                if not re.search(r'\b(I think|I believe|In my opinion)\b', sentences[0], re.IGNORECASE):
                    markup["annotations"].append({
                        "type": "strong_point",
                        "paragraph": para_idx,
                        "description": "Clear topic sentence"
                    })
            
            # Check for good transitions
            if para_idx > 0:
                transition_words = ["therefore", "consequently", "furthermore", "however", "in addition", "similarly"]
                first_sentence = sentences[0].lower() if sentences else ""
                if any(word in first_sentence for word in transition_words):
                    markup["annotations"].append({
                        "type": "strong_point",
                        "paragraph": para_idx,
                        "description": "Effective transition"
                    })
        
        return markup
    
    def _generate_feedback(self, paper_text, markup):
        """Generate detailed feedback based on markup."""
        # Group annotations by type and paragraph
        annotations_by_type = defaultdict(list)
        annotations_by_para = defaultdict(list)
        
        for annotation in markup["annotations"]:
            annotations_by_type[annotation["type"]].append(annotation)
            annotations_by_para[annotation["paragraph"]].append(annotation)
        
        # Generate paragraph-by-paragraph feedback
        paragraph_feedback = []
        paragraphs = [p for p in paper_text.split('\n\n') if p.strip()]
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_annotations = annotations_by_para[para_idx]
            issues = [a for a in para_annotations if a["type"] not in ["strong_point", "exceptional_point"]]
            strengths = [a for a in para_annotations if a["type"] in ["strong_point", "exceptional_point"]]
            
            feedback = {
                "paragraph_number": para_idx + 1,
                "text_excerpt": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph,
                "issues": [{"type": issue["type"], "description": issue["description"]} for issue in issues],
                "strengths": [{"type": strength["type"], "description": strength["description"]} for strength in strengths]
            }
            paragraph_feedback.append(feedback)
        
        # Identify error patterns
        error_patterns = []
        for error_type, annotations in annotations_by_type.items():
            if error_type not in ["strong_point", "exceptional_point"] and annotations:
                pattern = {
                    "type": error_type,
                    "frequency": len(annotations),
                    "examples": [a.get("text", a.get("word", "")) for a in annotations[:3] if a.get("text") or a.get("word")]
                }
                error_patterns.append(pattern)
        
        # Generate improvement suggestions
        suggestions = []
        if annotations_by_type["spelling_error"]:
            suggestions.append({
                "focus_area": "Spelling and Word Choice",
                "suggestions": [
                    "Use spell check before submitting",
                    "Review commonly confused words like there/their/they're",
                    "Keep a personal list of words you frequently misspell"
                ]
            })
        
        if annotations_by_type["grammar_error"]:
            suggestions.append({
                "focus_area": "Grammar",
                "suggestions": [
                    "Review subject-verb agreement rules",
                    "Read your work aloud to catch awkward constructions",
                    "Consider shorter, clearer sentences when expressing complex ideas"
                ]
            })
        
        if annotations_by_type["style_issue"]:
            suggestions.append({
                "focus_area": "Writing Style",
                "suggestions": [
                    "Replace vague intensifiers (very, really) with specific descriptions",
                    "Eliminate wordy phrases",
                    "Vary sentence structure for reader engagement"
                ]
            })
        
        return {
            "section_analysis": paragraph_feedback,
            "error_patterns": error_patterns,
            "strengths": [a for a in markup["annotations"] if a["type"] in ["strong_point", "exceptional_point"]],
            "improvement_suggestions": suggestions
        }
    
    def _apply_rubric(self, paper_text, markup):
        """Apply a rubric to evaluate the paper."""
        # Get basic counts
        annotations = markup["annotations"]
        word_count = markup["word_count"]
        paragraph_count = markup["paragraph_count"]
        
        # Count different types of annotations
        counts = defaultdict(int)
        for annotation in annotations:
            counts[annotation["type"]] += 1
        
        # Define rubric criteria
        criteria = [
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
        
        # Score each criterion
        criteria_scores = []
        total_points = 0
        total_possible = 0
        
        for criterion in criteria:
            max_points = criterion["max_points"]
            total_possible += max_points
            
            if criterion["id"] == "content":
                # Score content based on length and clarity
                if word_count < 300:
                    score = max_points * 0.6
                    justification = "Paper is quite short, limiting depth of content."
                elif word_count < 600:
                    score = max_points * 0.8
                    justification = "Moderate length allows for some content development."
                else:
                    score = max_points * 0.9
                    justification = "Good length providing space for detailed content."
            
            elif criterion["id"] == "organization":
                # Score organization based on paragraph structure and transitions
                transitions = counts["strong_point"]  # Assuming all "strong_point" are transitions for simplicity
                
                if paragraph_count < 3:
                    score = max_points * 0.6
                    justification = "Limited paragraph structure affects organization."
                elif transitions < 2:
                    score = max_points * 0.7
                    justification = "Could benefit from more explicit transitions."
                else:
                    score = max_points * 0.85
                    justification = "Good paragraph structure and transitions."
            
            elif criterion["id"] == "language":
                # Score language based on grammar and spelling errors
                grammar_errors = counts["grammar_error"]
                spelling_errors = counts["spelling_error"]
                style_issues = counts["style_issue"]
                
                # Normalize by word count (per 500 words)
                normalized_errors = ((grammar_errors + spelling_errors + style_issues) / word_count) * 500 if word_count > 0 else 0
                
                if normalized_errors > 10:
                    score = max_points * 0.5
                    justification = "Numerous language errors affect readability."
                elif normalized_errors > 5:
                    score = max_points * 0.7
                    justification = "Several language errors present."
                else:
                    score = max_points * 0.9
                    justification = "Few language errors; generally well-written."
            
            elif criterion["id"] == "formatting":
                # For simplicity, give a default score
                score = max_points * 0.8
                justification = "Paper follows basic formatting expectations."
            
            elif criterion["id"] == "critical_thinking":
                # Simplified scoring for critical thinking
                score = max_points * 0.75
                justification = "Shows acceptable critical thinking."
                
            # Ensure score is in bounds
            score = max(0, min(round(score), max_points))
            
            criteria_scores.append({
                "criterion_id": criterion["id"],
                "criterion_name": criterion["name"],
                "description": criterion["description"],
                "max_points": max_points,
                "score": score,
                "justification": justification
            })
            
            total_points += score
        
        # Calculate final grade
        percentage = (total_points / total_possible) * 100 if total_possible > 0 else 0
        
        # Get letter grade
        if percentage >= 90:
            letter_grade = "A"
        elif percentage >= 80:
            letter_grade = "B"
        elif percentage >= 70:
            letter_grade = "C"
        elif percentage >= 60:
            letter_grade = "D"
        else:
            letter_grade = "F"
        
        # Generate overall assessment
        strongest = max(criteria_scores, key=lambda x: x["score"] / x["max_points"])
        weakest = min(criteria_scores, key=lambda x: x["score"] / x["max_points"])
        
        if percentage >= 80:
            tone = "This is a strong paper that "
        elif percentage >= 70:
            tone = "This paper meets requirements and "
        else:
            tone = "This paper needs improvement and "
        
        assessment = f"{tone}demonstrates good work in {strongest['criterion_name']}. "
        assessment += f"The primary area needing improvement is {weakest['criterion_name']}."
        
        return {
            "rubric_name": f"{self.subject_area.capitalize()} Paper Rubric - {self.education_level.capitalize()} Level",
            "criteria_scores": criteria_scores,
            "total_points": total_points,
            "total_possible": total_possible,
            "percentage": percentage,
            "letter_grade": letter_grade,
            "overall_assessment": assessment
        }
    
    def _create_summary(self, detailed_feedback, assessment):
        """Create a summary with final assessment and next steps."""
        # Extract key information
        error_patterns = detailed_feedback["error_patterns"]
        strengths = detailed_feedback["strengths"]
        improvements = detailed_feedback["improvement_suggestions"]
        
        # Get overall scores
        overall = assessment["overall_assessment"]
        percentage = assessment["percentage"]
        letter_grade = assessment["letter_grade"]
        
        # Generate key strengths list (simplifying from the full implementation)
        key_strengths = []
        strength_counts = defaultdict(int)
        
        for strength in strengths:
            strength_counts[strength["description"]] += 1
        
        for desc, count in sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            key_strengths.append({
                "description": desc,
                "frequency": f"Observed {count} times"
            })
        
        # Generate improvement areas
        improvement_areas = []
        
        # Start with lowest scoring criteria
        sorted_criteria = sorted(assessment["criteria_scores"], key=lambda x: x["score"] / x["max_points"])
        
        for criterion in sorted_criteria[:2]:  # Top 2 weakest areas
            improvement_areas.append({
                "area": criterion["criterion_name"],
                "justification": criterion["justification"],
                "score": f"{criterion['score']}/{criterion['max_points']} points"
            })
        
        # Create encouraging closing comment
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
                "points": f"{assessment['total_points']}/{assessment['total_possible']}"
            },
            "overall_assessment": overall,
            "key_strengths": key_strengths,
            "improvement_areas": improvement_areas,
            "next_steps": improvements,
            "closing_comment": closing
        }
        
        return summary


def main():
    print("\nSIMPLIFIED PAPER GRADING SYSTEM DEMONSTRATION\n")
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Sample paper text (abbreviated for demo purposes)
    sample_paper = """
The Impact of Climate Change on Marine Ecosystems

Climate change is one of the most pressing issues facing our planet today. Rising temperatures, ocean acidification, and changing weather patterns are having a profound effect on marine ecosystems worldwide. This paper will explore the various ways in which climate change impacts marine life and the potential long-term consequences for our oceans.

Ocean acidification is occuring due to increased CO2 levels in the atmosphere. When carbon dioxide dissolves in seawater, it forms carbonic acid, which lowers the pH of the ocean. This process has serious implications for marine organisms with calcium carbonate shells or skeletons, such as coral, mollusks, and some plankton species. As seawater becomes more acidic, these organisms struggle to build and maintain there shells, which can lead to population declines and disruptions in the food web.

Rising sea temperatures are another major concern. Many marine species have specific temperature requirements and cannot adapt quickly to changing conditions. Coral bleaching, for example, occurs when water temperatures rise above a certain threshold, causing coral to expel the algae that provide them with nutrients. Without these algae, coral turn white (hence the term "bleaching") and are at risk of starvation and death. Given that coral reefs are home to approximately 25% of all marine species, the loss of these ecosystems would be catastrophic.

Changes in ocean currents and weather patterns also effect marine life. Currents distribute nutrients, regulate temperature, and influence the migration and breeding patterns of many species. Alterations to these currents can disrupt entire ecosystems. Furthermore, more frequent and severe storms can damage coastal habitats such as mangroves and seagrass beds, which provide important nursery areas for many marine species.

In conclusion, climate change poses a significant threat to marine ecosystems. The combined effects of ocean acidification, rising temperatures, and changing weather patterns are already having observable impacts on marine life. Without urgent action to reduce greenhouse gas emissions and mitigate the effects of climate change, the future of our oceans and the countless species that depend on them is uncertain.
"""

    # Initialize the paper grading system
    print("Initializing paper grading system...")
    grader = SimplePaperGrader(subject_area="STEM", education_level="high_school")
    
    # Grade the paper
    print("\nGrading paper...")
    results = grader.grade_paper(sample_paper)
    
    # Save results to files
    print("\nSaving results to files...")
    with open("results/simple_markup.json", "w") as f:
        json.dump(results["markup"], f, indent=2)
    
    with open("results/simple_feedback.json", "w") as f:
        json.dump(results["detailed_feedback"], f, indent=2)
    
    with open("results/simple_assessment.json", "w") as f:
        json.dump(results["assessment"], f, indent=2)
    
    with open("results/simple_summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2)
    
    # Display results
    markup = results["markup"]
    assessment = results["assessment"]
    summary = results["summary"]
    
    print("\n" + "="*80)
    print("PAPER GRADING RESULTS")
    print("="*80)
    
    print(f"\nAnnotations: {len(markup['annotations'])}")
    
    # Count by type
    annotation_counts = defaultdict(int)
    for annotation in markup["annotations"]:
        annotation_counts[annotation["type"]] += 1
    
    print("\nAnnotation breakdown:")
    for anno_type, count in annotation_counts.items():
        print(f"  {anno_type}: {count}")
    
    print(f"\nFinal Grade: {assessment['letter_grade']} ({assessment['percentage']:.1f}%)")
    print(f"Points: {assessment['total_points']}/{assessment['total_possible']}")
    
    print("\nScores by criterion:")
    for criterion in assessment["criteria_scores"]:
        print(f"  {criterion['criterion_name']}: {criterion['score']}/{criterion['max_points']}")
        print(f"    {criterion['justification']}")
    
    print("\nOverall Assessment:")
    print(summary["overall_assessment"])
    
    print("\nKey Strengths:")
    for i, strength in enumerate(summary["key_strengths"]):
        print(f"{i+1}. {strength['description']}")
        if "frequency" in strength:
            print(f"   {strength['frequency']}")
    
    print("\nAreas for Improvement:")
    for i, area in enumerate(summary["improvement_areas"]):
        print(f"{i+1}. {area['area']}")
        print(f"   {area['justification']}")
    
    print("\nNext Steps:")
    for i, step in enumerate(summary["next_steps"]):
        print(f"{i+1}. {step['focus_area']}:")
        for suggestion in step["suggestions"]:
            print(f"   • {suggestion}")
    
    print("\nClosing Comment:")
    print(summary["closing_comment"])
    
    print("\n" + "="*80)
    print("All results have been saved to the 'results' directory.")
    print("="*80)

if __name__ == "__main__":
    main()
