"""
Data loader module for app store review analysis.
Handles loading, preprocessing, and validation of review datasets.
"""

from typing import List, Dict, Union, Optional
import pandas as pd
import json
from pathlib import Path

class AppReviewDataset:
    """Class for loading and managing app store review datasets."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir (Union[str, Path]): Directory containing review data files
        """
        self.data_dir = Path(data_dir)
        self.reviews_df: Optional[pd.DataFrame] = None
        
    def load_csv_data(self, filename: str, required_columns: List[str] = None) -> pd.DataFrame:
        """
        Load review data from a CSV file.
        
        Args:
            filename (str): Name of the CSV file
            required_columns (List[str], optional): List of required column names
            
        Returns:
            pd.DataFrame: Loaded and validated review data
            
        Raises:
            ValueError: If required columns are missing
        """
        if required_columns is None:
            required_columns = ['review_id', 'text', 'rating', 'date']
            
        file_path = self.data_dir / filename
        df = pd.read_csv(file_path)
        
        # Validate required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    def preprocess_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the review dataset.
        
        Args:
            df (pd.DataFrame): Raw review data
            
        Returns:
            pd.DataFrame: Preprocessed review data
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Basic preprocessing steps
        if 'text' in processed_df.columns:
            # Remove empty reviews
            processed_df = processed_df.dropna(subset=['text'])
            # Basic text cleaning
            processed_df['text'] = processed_df['text'].str.strip()
            processed_df = processed_df[processed_df['text'].str.len() > 0]
            
        if 'rating' in processed_df.columns:
            # Ensure ratings are numeric
            processed_df['rating'] = pd.to_numeric(processed_df['rating'], errors='coerce')
            
        if 'date' in processed_df.columns:
            # Convert date strings to datetime objects
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
            
        return processed_df
    
    def load_sample_dataset(self, n_samples: int = 50) -> pd.DataFrame:
        """
        Load a sample dataset of app reviews.
        If no data is available, creates a synthetic dataset for testing.
        
        Args:
            n_samples (int): Number of reviews to include
            
        Returns:
            pd.DataFrame: Sample review dataset
        """
        # Example text reviews to repeat
        sample_texts = [
            "Great app, really helpful for productivity!",
            "The latest update broke the login feature.",
            "Would be better if it had dark mode.",
        ]
        sample_ratings = [5, 2, 4]
        
        # Create lists of exactly the required length by repeating and slicing
        texts = (sample_texts * (n_samples // len(sample_texts) + 1))[:n_samples]
        ratings = (sample_ratings * (n_samples // len(sample_ratings) + 1))[:n_samples]
        
        # Example synthetic data structure with exact lengths
        synthetic_data = {
            'review_id': list(range(1, n_samples + 1)),
            'text': texts,
            'rating': ratings,
            'date': pd.date_range(end=pd.Timestamp.now(), periods=n_samples)
        }
        
        df = pd.DataFrame(synthetic_data)
        return df
    
    def get_reviews(self, filename: Optional[str] = None, n_samples: int = 50) -> pd.DataFrame:
        """
        Main method to get the review dataset.
        
        Args:
            filename (Optional[str]): Name of the data file to load
            n_samples (int): Number of reviews to include if using sample data
            
        Returns:
            pd.DataFrame: Processed review dataset
        """
        if filename:
            df = self.load_csv_data(filename)
        else:
            df = self.load_sample_dataset(n_samples)
            
        self.reviews_df = self.preprocess_reviews(df)
        return self.reviews_df
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Calculate basic statistics about the loaded dataset.
        
        Returns:
            Dict[str, Union[int, float]]: Dictionary containing dataset statistics
        """
        if self.reviews_df is None:
            raise ValueError("No dataset loaded. Call get_reviews() first.")
            
        stats = {
            'total_reviews': len(self.reviews_df),
            'avg_rating': self.reviews_df['rating'].mean(),
            'rating_distribution': self.reviews_df['rating'].value_counts().to_dict(),
            'avg_review_length': self.reviews_df['text'].str.len().mean()
        }
        
        return stats
