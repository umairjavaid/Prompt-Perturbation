"""
Model runner module for executing foundation models with different prompts.
Supports both OpenAI and Hugging Face models for app review analysis.
"""

from typing import List, Dict, Any, Optional, Union
import os
import json
from pathlib import Path
import openai
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

class ModelRunner:
    """Class for running different foundation models with various prompts."""
    
    def __init__(self):
        """Initialize the model runner with available models."""
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
            
        # Initialize Hugging Face pipelines
        self.hf_models = {
            'sentiment': pipeline(
                'text-classification',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                top_k=None
            ),
            'zero_shot': pipeline(
                'zero-shot-classification',
                model='facebook/bart-large-mnli'
            )
        }
        
    async def run_openai_model(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Run analysis using OpenAI model.
        
        Args:
            prompt (str): The prompt to send to the model
            model (str): OpenAI model identifier
            temperature (float): Sampling temperature (0-1)
            
        Returns:
            Dict[str, Any]: Model response and metadata
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not found in environment variables")
            
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing app store reviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        return {
            'model': model,
            'response': response.choices[0].message.content,
            'finish_reason': response.choices[0].finish_reason
        }
    
    def run_huggingface_model(
        self,
        text: str,
        task: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run analysis using Hugging Face model.
        
        Args:
            text (str): The text to analyze
            task (str): Analysis task ('sentiment' or 'zero_shot')
            labels (Optional[List[str]]): Labels for zero-shot classification
            
        Returns:
            Dict[str, Any]: Model predictions and metadata
        """
        if task == 'sentiment':
            # Use sentiment analysis pipeline
            result = self.hf_models['sentiment'](text)
            return {
                'model': 'distilbert-sst2',
                'predictions': result
            }
        elif task == 'zero_shot':
            # Use zero-shot classification for feature requests and bug reports
            if not labels:
                labels = ['feature request', 'bug report', 'general feedback']
            result = self.hf_models['zero_shot'](text, labels)
            return {
                'model': 'bart-large-mnli',
                'predictions': {
                    'labels': result['labels'],
                    'scores': result['scores']
                }
            }
        else:
            raise ValueError(f"Unknown task: {task}")
    
    async def analyze_review(
        self,
        review: str,
        prompt: str,
        task: str,
        use_openai: bool = True,
        use_hf: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a review using both OpenAI and Hugging Face models.
        
        Args:
            review (str): The review text to analyze
            prompt (str): The prompt to use for OpenAI
            task (str): The analysis task
            use_openai (bool): Whether to use OpenAI model
            use_hf (bool): Whether to use Hugging Face model
            
        Returns:
            Dict[str, Any]: Results from both models
        """
        results = {'review': review, 'task': task}
        
        if use_openai:
            try:
                openai_result = await self.run_openai_model(prompt)
                results['openai'] = openai_result
            except Exception as e:
                results['openai_error'] = str(e)
                
        if use_hf:
            try:
                if task == 'sentiment':
                    hf_result = self.run_huggingface_model(review, 'sentiment')
                else:
                    # Use zero-shot for feature requests and bug reports
                    hf_result = self.run_huggingface_model(
                        review,
                        'zero_shot',
                        labels=['contains ' + task, 'no ' + task]
                    )
                results['huggingface'] = hf_result
            except Exception as e:
                results['huggingface_error'] = str(e)
                
        return results
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models for each provider.
        
        Returns:
            Dict[str, List[str]]: Available models by provider
        """
        models = {
            'openai': ['gpt-3.5-turbo', 'gpt-4'] if self.openai_client else [],
            'huggingface': [
                'distilbert-base-uncased-finetuned-sst-2-english',
                'facebook/bart-large-mnli'
            ]
        }
        return models
