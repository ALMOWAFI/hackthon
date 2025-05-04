#!/usr/bin/env python3
"""
Paper Grading System Demo

This script demonstrates the advanced paper grading system that provides
teacher-like feedback and assessment for academic papers.
"""

import os
import json
import nltk

# Ensure required NLTK data packages are downloaded
print("Downloading required NLTK data packages...")
# The simple nltk.download with a GUI may be more reliable for getting all necessary data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    # Make sure words() is available
    from nltk.corpus import wordnet
    wordnet.words()
    print("NLTK data packages successfully downloaded")
except Exception as e:
    print(f"Error during NLTK data download: {str(e)}")
    print("Attempting alternative download method...")
    # Alternative approach if the first one fails
    nltk.download('all', quiet=False)

from math_analyzer.paper_grading import PaperGradingSystem

def main():
    print("\nADVANCED PAPER GRADING SYSTEM DEMONSTRATION\n")
    
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
    grader = PaperGradingSystem(subject_area="STEM", education_level="high_school")
    
    # Phase 1: Mark the paper
    print("\nPHASE 1: MARKING THE PAPER")
    markup = grader.mark_paper(sample_paper)
    print(f"Generated {len(markup['annotations'])} annotations")
    
    # Save markup to file
    with open("results/paper_markup.json", "w") as f:
        json.dump(markup, f, indent=2)
    
    # Count annotation types
    annotation_counts = {}
    for annotation in markup["annotations"]:
        anno_type = annotation["type"]
        annotation_counts[anno_type] = annotation_counts.get(anno_type, 0) + 1
    
    print("\nAnnotation breakdown:")
    for anno_type, count in annotation_counts.items():
        print(f"  {anno_type}: {count}")
    
    # Phase 2: Generate detailed feedback
    print("\nPHASE 2: GENERATING DETAILED FEEDBACK")
    detailed_feedback = grader.generate_detailed_feedback(sample_paper, markup)
    
    # Save detailed feedback to file
    with open("results/detailed_feedback.json", "w") as f:
        json.dump(detailed_feedback, f, indent=2)
    
    print("\nDetailed feedback generated:")
    print(f"  Section analyses: {len(detailed_feedback['section_analysis'])}")
    print(f"  Error patterns identified: {len(detailed_feedback['error_patterns'])}")
    print(f"  Strengths highlighted: {len(detailed_feedback['strengths'])}")
    print(f"  Improvement suggestions: {len(detailed_feedback['improvement_suggestions'])}")
    
    # Phase 3: Apply rubric
    print("\nPHASE 3: APPLYING RUBRIC")
    assessment = grader.apply_rubric(sample_paper, markup)
    
    # Save assessment to file
    with open("results/rubric_assessment.json", "w") as f:
        json.dump(assessment, f, indent=2)
    
    print(f"\nRubric assessment:")
    print(f"  Final grade: {assessment['letter_grade']} ({assessment['percentage']:.1f}%)")
    print(f"  Total points: {assessment['total_points']}/{assessment['total_possible']}")
    
    print("\nScores by criterion:")
    for criterion in assessment["criteria_scores"]:
        print(f"  {criterion['criterion_name']}: {criterion['score']}/{criterion['max_points']}")
    
    # Phase 4: Create summary
    print("\nPHASE 4: CREATING SUMMARY FEEDBACK")
    summary = grader.create_summary(detailed_feedback, assessment)
    
    # Save summary to file
    with open("results/feedback_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary feedback:")
    print(f"  Final grade: {summary['final_grade']['letter_grade']} ({summary['final_grade']['percentage']:.1f}%)")
    print(f"  Key strengths identified: {len(summary['key_strengths'])}")
    print(f"  Areas for improvement: {len(summary['improvement_areas'])}")
    print(f"  Next steps provided: {len(summary['next_steps'])}")
    
    # Display sample of the grading
    print("\n" + "="*80)
    print("SAMPLE PAPER GRADING OUTPUT")
    print("="*80)
    
    print("\nOVERALL ASSESSMENT:")
    print(summary["overall_assessment"])
    
    print("\nKEY STRENGTHS:")
    for i, strength in enumerate(summary["key_strengths"]):
        print(f"{i+1}. {strength['description']}")
        if "justification" in strength:
            print(f"   {strength['justification']}")
        elif "frequency" in strength:
            print(f"   {strength['frequency']}")
    
    print("\nAREAS FOR IMPROVEMENT:")
    for i, area in enumerate(summary["improvement_areas"]):
        print(f"{i+1}. {area['area']}")
        print(f"   {area['justification']}")
        if "score" in area:
            print(f"   Score: {area['score']}")
        elif "frequency" in area:
            print(f"   {area['frequency']}")
    
    print("\nNEXT STEPS:")
    for i, step in enumerate(summary["next_steps"]):
        print(f"{i+1}. {step['focus']}:")
        for action in step["actions"]:
            print(f"   • {action}")
    
    print("\nCLOSING COMMENT:")
    print(summary["closing_comment"])
    
    print("\n" + "="*80)
    print("All assessment results have been saved to the 'results' directory.")
    print("="*80)

if __name__ == "__main__":
    main()
