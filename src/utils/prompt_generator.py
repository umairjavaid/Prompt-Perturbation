"""
Prompt generator module for app store review analysis.
Creates and manages different prompt variants for analyzing reviews.
"""

from typing import List, Dict, Optional
import json
from pathlib import Path

class PromptGenerator:
    """Class for generating and managing prompt variants for review analysis."""
    
    def __init__(self):
        """Initialize the prompt generator with base templates."""
        self.base_templates = {
            'sentiment': {
                'formal': "Analyze the sentiment of this app store review: '{review}'. Classify it as positive, neutral, or negative.",
                'casual': "Hey, what's the vibe of this review? '{review}' Is it positive, neutral, or negative?",
                'detailed': "Please evaluate the sentiment expressed in the following app store review: '{review}'. Consider the user's tone, word choice, and overall message to classify it as positive, neutral, or negative. Provide your classification.",
                'structured': "INPUT: '{review}'\nTASK: Sentiment Analysis\nOUTPUT FORMAT: One of [positive, neutral, negative]\nANALYSIS:",
                'minimal': "Review: '{review}'\nSentiment (positive/neutral/negative):"
            },
            'feature_request': {
                'formal': "Does this app store review contain a feature request? Review: '{review}'. Please identify any requested features.",
                'casual': "Can you spot any feature requests in this review? '{review}' What does the user want to be added?",
                'detailed': "Analyze this app store review for feature requests: '{review}'. Identify any specific features, functionality, or improvements the user is asking for.",
                'structured': "INPUT: '{review}'\nTASK: Feature Request Detection\nOUTPUT FORMAT: [Yes/No] + [Requested Feature]\nANALYSIS:",
                'minimal': "Review: '{review}'\nContains feature request? (yes/no):"
            },
            'bug_report': {
                'formal': "Identify if this app store review reports any bugs or technical issues: '{review}'",
                'casual': "Hey, does this review mention any bugs or problems? '{review}'",
                'detailed': "Please analyze this app store review for bug reports or technical issues: '{review}'. Identify specific problems mentioned by the user.",
                'structured': "INPUT: '{review}'\nTASK: Bug Report Detection\nOUTPUT FORMAT: [Yes/No] + [Description of Issue]\nANALYSIS:",
                'minimal': "Review: '{review}'\nContains bug report? (yes/no):"
            }
        }
        
    def get_prompt_variants(self, task: str, review_text: str) -> List[Dict[str, str]]:
        """
        Generate different variants of prompts for a specific task and review.
        
        Args:
            task (str): The analysis task ('sentiment', 'feature_request', or 'bug_report')
            review_text (str): The review text to analyze
            
        Returns:
            List[Dict[str, str]]: List of prompt variants with their style labels
        """
        if task not in self.base_templates:
            raise ValueError(f"Unknown task: {task}")
            
        variants = []
        for style, template in self.base_templates[task].items():
            variants.append({
                'style': style,
                'prompt': template.format(review=review_text)
            })
            
        return variants
    
    def create_custom_prompt(self, template: str, review_text: str) -> str:
        """
        Create a prompt using a custom template.
        
        Args:
            template (str): Custom prompt template with {review} placeholder
            review_text (str): The review text to analyze
            
        Returns:
            str: Formatted prompt
        """
        return template.format(review=review_text)
    
    def get_task_list(self) -> List[str]:
        """
        Get list of available analysis tasks.
        
        Returns:
            List[str]: List of task names
        """
        return list(self.base_templates.keys())
    
    def get_style_list(self) -> List[str]:
        """
        Get list of available prompt styles.
        
        Returns:
            List[str]: List of style names
        """
        return list(self.base_templates['sentiment'].keys())
