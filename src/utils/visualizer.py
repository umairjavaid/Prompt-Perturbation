"""
Visualizer module for generating plots and visualizations of model robustness analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelVisualizer:
    """Class for creating visualizations of model analysis results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (Optional[Path]): Directory to save visualizations
        """
        self.output_dir = output_dir or Path.cwd() / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # Updated styling approach compatible with newer matplotlib versions
        try:
            # Try using a versioned seaborn style
            plt.style.use('seaborn-v0_8')
        except (FileNotFoundError, IOError):
            try:
                # Fall back to another version if available
                plt.style.use('seaborn-v0_8-whitegrid')
            except (FileNotFoundError, IOError):
                # If all else fails, use seaborn's native styling
                sns.set(style="whitegrid")
        
    def create_prompt_impact_heatmap(
        self,
        results: List[Dict[str, Any]],
        task: str,
        figsize: tuple = (12, 8)
    ) -> None:
        """
        Create a heatmap showing how different prompt styles affect model outputs.
        
        Args:
            results (List[Dict[str, Any]]): Analysis results
            task (str): Task to visualize
            figsize (tuple): Figure size
        """
        # Extract prompt styles and their impacts
        impact_data = []
        for review in results:
            if task in review['tasks']:
                for variant in review['tasks'][task]:
                    # Check if we have a valid OpenAI response or handle other model results
                    if 'result' in variant:
                        response_length = 0
                        confidence = 0.0
                        
                        # Try to get OpenAI response if available
                        if 'openai' in variant['result'] and 'response' in variant['result']['openai']:
                            response_length = len(variant['result']['openai']['response'])
                            confidence = self._extract_confidence(variant['result']['openai']['response'])
                        # Fallback to HuggingFace if available
                        elif 'huggingface' in variant['result'] and 'predictions' in variant['result']['huggingface']:
                            # Get length of predictions as string
                            response_length = len(str(variant['result']['huggingface']['predictions']))
                            # Try to extract confidence if predictions have scores
                            if isinstance(variant['result']['huggingface']['predictions'], dict) and 'scores' in variant['result']['huggingface']['predictions']:
                                confidence = max(variant['result']['huggingface']['predictions']['scores'])
                            elif isinstance(variant['result']['huggingface']['predictions'], list) and variant['result']['huggingface']['predictions']:
                                if 'score' in variant['result']['huggingface']['predictions'][0][0]:
                                    confidence = variant['result']['huggingface']['predictions'][0][0]['score']
                        
                        impact_data.append({
                            'review_id': review['review_id'],  # Use lowercase key to match the data structure
                            'Prompt Style': variant['style'],
                            'Response Length': response_length,
                            'Confidence': confidence
                        })
                        
        if not impact_data:
            print(f"No valid data found for task: {task}")
            return
            
        df = pd.DataFrame(impact_data)
        pivot_length = df.pivot(
            index='review_id',  # Use lowercase key to match the data structure
            columns='Prompt Style',
            values='Response Length'
        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot_length,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd'
        )
        plt.title(f'Impact of Prompt Style on Response Length - {task.title()}')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(
            self.output_dir / f'prompt_impact_heatmap_{task}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
    def plot_model_agreement(
        self,
        results: List[Dict[str, Any]],
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot agreement between OpenAI and Hugging Face models.
        
        Args:
            results (List[Dict[str, Any]]): Analysis results
            figsize (tuple): Figure size
        """
        agreement_data = []
        for review in results:
            for task, task_results in review['tasks'].items():
                for variant in task_results:
                    if 'openai' in variant['result'] and 'huggingface' in variant['result']:
                        agreement = self._calculate_agreement(
                            variant['result']['openai'],
                            variant['result']['huggingface']
                        )
                        agreement_data.append({
                            'Task': task,
                            'Prompt Style': variant['style'],
                            'Agreement': agreement
                        })
                        
        df = pd.DataFrame(agreement_data)
        plt.figure(figsize=figsize)
        
        sns.barplot(
            data=df,
            x='Task',
            y='Agreement',
            hue='Prompt Style'
        )
        plt.title('Model Agreement by Task and Prompt Style')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(
            self.output_dir / 'model_agreement.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
    def plot_consistency_matrix(
        self,
        evaluation_results: Dict[str, Any],
        figsize: tuple = (8, 8)
    ) -> None:
        """
        Create a matrix showing consistency across tasks and metrics.
        
        Args:
            evaluation_results (Dict[str, Any]): Results from the evaluator
            figsize (tuple): Figure size
        """
        task_metrics = evaluation_results['per_task_metrics']
        metrics = list(next(iter(task_metrics.values())).keys())
        tasks = list(task_metrics.keys())
        
        matrix = np.zeros((len(tasks), len(metrics)))
        for i, task in enumerate(tasks):
            for j, metric in enumerate(metrics):
                matrix[i, j] = task_metrics[task][metric]
                
        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            xticklabels=metrics,
            yticklabels=tasks,
            cmap='viridis'
        )
        plt.title('Consistency Matrix across Tasks and Metrics')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(
            self.output_dir / 'consistency_matrix.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
    def plot_robustness_radar(
        self,
        evaluation_results: Dict[str, Any],
        figsize: tuple = (10, 10)
    ) -> None:
        """
        Create a radar plot showing model robustness across different dimensions.
        
        Args:
            evaluation_results (Dict[str, Any]): Results from the evaluator
            figsize (tuple): Figure size
        """
        metrics = evaluation_results['overall_metrics']
        
        # Prepare the data
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Create the radar plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # complete the circle
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title('Model Robustness Radar Plot')
        
        # Save the plot
        plt.savefig(
            self.output_dir / 'robustness_radar.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
    @staticmethod
    def _extract_confidence(response: str) -> float:
        """
        Extract confidence score from model response if available.
        
        Args:
            response (str): Model response text
            
        Returns:
            float: Confidence score or 0.0 if not found
        """
        # This is a simple implementation - could be made more sophisticated
        confidence_indicators = {
            'definitely': 1.0,
            'likely': 0.8,
            'probably': 0.6,
            'maybe': 0.4,
            'unsure': 0.2
        }
        
        response_lower = response.lower()
        for indicator, score in confidence_indicators.items():
            if indicator in response_lower:
                return score
                
        return 0.0
        
    @staticmethod
    def _calculate_agreement(
        openai_result: Dict[str, Any],
        hf_result: Dict[str, Any]
    ) -> float:
        """
        Calculate agreement score between OpenAI and Hugging Face results.
        
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
