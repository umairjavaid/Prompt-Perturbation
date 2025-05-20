"""
Evaluator module for measuring model robustness and consistency across prompt variants.
Analyzes how different prompt formulations affect model outputs.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import cohen_kappa_score
import json
from pathlib import Path

class ModelEvaluator:
    """Class for evaluating model robustness across prompt variants."""
    
    def __init__(self):
        """Initialize the evaluator with metric definitions."""
        self.metrics = {
            'sentiment': self._evaluate_sentiment_consistency,
            'feature_request': self._evaluate_feature_request_consistency,
            'bug_report': self._evaluate_bug_report_consistency
        }
        
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate model robustness across all tasks and prompt variants.
        
        Args:
            results (List[Dict[str, Any]]): Analysis results from model runner
            
        Returns:
            Dict[str, Any]: Evaluation metrics and statistics
        """
        evaluation = {
            'overall_metrics': {},
            'per_task_metrics': {},
            'per_review_metrics': {}
        }
        
        # Evaluate each task
        for task in ['sentiment', 'feature_request', 'bug_report']:
            task_metrics = self._evaluate_task(results, task)
            evaluation['per_task_metrics'][task] = task_metrics
            
        # Calculate overall robustness metrics
        evaluation['overall_metrics'] = self._calculate_overall_metrics(
            evaluation['per_task_metrics']
        )
        
        # Calculate per-review metrics
        evaluation['per_review_metrics'] = self._calculate_per_review_metrics(results)
        
        return evaluation
    
    def _evaluate_task(self, results: List[Dict[str, Any]], task: str) -> Dict[str, float]:
        """
        Evaluate model performance for a specific task.
        
        Args:
            results (List[Dict[str, Any]]): Analysis results
            task (str): Task name
            
        Returns:
            Dict[str, float]: Task-specific metrics
        """
        task_results = []
        for review_result in results:
            if task in review_result['tasks']:
                task_results.extend(review_result['tasks'][task])
                
        metrics = {
            'prompt_variation_consistency': self._calculate_prompt_consistency(task_results),
            'model_agreement': self._calculate_model_agreement(task_results),
            'response_stability': self._calculate_response_stability(task_results)
        }
        
        return metrics
    
    def _evaluate_sentiment_consistency(
        self,
        variants: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate consistency of sentiment analysis across prompt variants.
        
        Args:
            variants (List[Dict[str, Any]]): Results for different prompt variants
            
        Returns:
            Dict[str, float]: Consistency metrics
        """
        sentiments = []
        for variant in variants:
            if 'openai' in variant['result']:
                response = variant['result']['openai']['response'].lower()
                if 'positive' in response:
                    sentiments.append(1)
                elif 'negative' in response:
                    sentiments.append(-1)
                else:
                    sentiments.append(0)
                    
        if not sentiments:
            return {'consistency': 0.0}
            
        # Calculate variance in sentiment predictions
        consistency = 1.0 - np.var(sentiments) / 2.0  # Normalize to [0,1]
        return {'consistency': consistency}
    
    def _evaluate_feature_request_consistency(
        self,
        variants: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate consistency of feature request detection across prompt variants.
        
        Args:
            variants (List[Dict[str, Any]]): Results for different prompt variants
            
        Returns:
            Dict[str, float]: Consistency metrics
        """
        feature_requests = []
        for variant in variants:
            if 'openai' in variant['result']:
                response = variant['result']['openai']['response'].lower()
                feature_requests.append(1 if 'yes' in response else 0)
                
        if not feature_requests:
            return {'consistency': 0.0}
            
        # Calculate agreement ratio
        consistency = 1.0 - np.var(feature_requests)
        return {'consistency': consistency}
    
    def _evaluate_bug_report_consistency(
        self,
        variants: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate consistency of bug report detection across prompt variants.
        
        Args:
            variants (List[Dict[str, Any]]): Results for different prompt variants
            
        Returns:
            Dict[str, float]: Consistency metrics
        """
        bug_reports = []
        for variant in variants:
            if 'openai' in variant['result']:
                response = variant['result']['openai']['response'].lower()
                bug_reports.append(1 if 'yes' in response else 0)
                
        if not bug_reports:
            return {'consistency': 0.0}
            
        # Calculate agreement ratio
        consistency = 1.0 - np.var(bug_reports)
        return {'consistency': consistency}
    
    def _calculate_prompt_consistency(
        self,
        task_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate how consistent model responses are across different prompt variants.
        
        Args:
            task_results (List[Dict[str, Any]]): Results for a specific task
            
        Returns:
            float: Consistency score
        """
        responses = []
        for result in task_results:
            if 'openai' in result['result']:
                responses.append(result['result']['openai']['response'])
                
        if len(responses) < 2:
            return 0.0
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._text_similarity(responses[i], responses[j])
                similarities.append(similarity)
                
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_model_agreement(
        self,
        task_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate agreement between OpenAI and Hugging Face models.
        
        Args:
            task_results (List[Dict[str, Any]]): Results for a specific task
            
        Returns:
            float: Agreement score
        """
        agreements = []
        for result in task_results:
            if 'openai' in result['result'] and 'huggingface' in result['result']:
                agreement = self._compare_model_outputs(
                    result['result']['openai'],
                    result['result']['huggingface']
                )
                agreements.append(agreement)
                
        return np.mean(agreements) if agreements else 0.0
    
    def _calculate_response_stability(
        self,
        task_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate stability of responses across prompt variations.
        
        Args:
            task_results (List[Dict[str, Any]]): Results for a specific task
            
        Returns:
            float: Stability score
        """
        # Group results by prompt style
        style_responses = {}
        for result in task_results:
            style = result['style']
            if style not in style_responses:
                style_responses[style] = []
            if 'openai' in result['result']:
                style_responses[style].append(result['result']['openai']['response'])
                
        # Calculate variance in response length and content
        variances = []
        for responses in style_responses.values():
            if responses:
                lengths = [len(r) for r in responses]
                variances.append(np.var(lengths) / np.mean(lengths))  # Coefficient of variation
                
        return 1.0 / (1.0 + np.mean(variances)) if variances else 0.0
    
    def _calculate_overall_metrics(
        self,
        task_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate overall robustness metrics across all tasks.
        
        Args:
            task_metrics (Dict[str, Dict[str, float]]): Metrics for each task
            
        Returns:
            Dict[str, float]: Overall metrics
        """
        overall = {}
        for metric in ['prompt_variation_consistency', 'model_agreement', 'response_stability']:
            values = []
            for task_dict in task_metrics.values():
                if metric in task_dict:
                    values.append(task_dict[metric])
            overall[f'average_{metric}'] = np.mean(values) if values else 0.0
            
        return overall
    
    def _calculate_per_review_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each individual review.
        
        Args:
            results (List[Dict[str, Any]]): Analysis results
            
        Returns:
            Dict[str, Dict[str, float]]: Metrics per review
        """
        per_review = {}
        for review in results:
            review_id = review['review_id']
            metrics = {}
            
            for task, task_results in review['tasks'].items():
                task_fn = self.metrics.get(task)
                if task_fn:
                    metrics[task] = task_fn(task_results)
                    
            per_review[str(review_id)] = metrics
            
        return per_review
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        Calculate simple text similarity score.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score
        """
        # Convert to sets of words for simple overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def _compare_model_outputs(
        openai_result: Dict[str, Any],
        hf_result: Dict[str, Any]
    ) -> float:
        """
        Compare outputs from OpenAI and Hugging Face models.
        
        Args:
            openai_result (Dict[str, Any]): OpenAI model result
            hf_result (Dict[str, Any]): Hugging Face model result
            
        Returns:
            float: Agreement score
        """
        # This is a simple comparison - could be made more sophisticated
        openai_text = openai_result['response'].lower()
        
        if 'predictions' in hf_result:
            if isinstance(hf_result['predictions'], list):
                # For sentiment analysis
                hf_sentiment = hf_result['predictions'][0]['label']
                if ('positive' in openai_text and 'POSITIVE' in hf_sentiment) or \
                   ('negative' in openai_text and 'NEGATIVE' in hf_sentiment):
                    return 1.0
            elif isinstance(hf_result['predictions'], dict):
                # For zero-shot classification
                highest_score_idx = np.argmax(hf_result['predictions']['scores'])
                predicted_label = hf_result['predictions']['labels'][highest_score_idx]
                if ('yes' in openai_text and 'contains' in predicted_label.lower()) or \
                   ('no' in openai_text and 'no' in predicted_label.lower()):
                    return 1.0
                    
        return 0.0
