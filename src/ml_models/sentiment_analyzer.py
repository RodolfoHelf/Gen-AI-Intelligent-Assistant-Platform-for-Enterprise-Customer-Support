"""
Sentiment Analyzer for analyzing customer sentiment in queries
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

from ..ingestion.preprocessor import DataPreprocessor


class SentimentAnalyzer:
    """
    Sentiment Analyzer for analyzing customer sentiment in queries
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Sentiment Analyzer
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.model_path = model_path or "models/sentiment_model.pkl"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load pre-trained model if available
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Loaded pre-trained sentiment model from {self.model_path}")
            else:
                self.logger.info("No pre-trained model found. Model will be trained when data is provided.")
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {e}")
    
    def _save_model(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Saved sentiment model to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving sentiment model: {e}")
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of the given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label ('positive', 'negative', 'neutral')
        """
        try:
            if self.model is None:
                # Use rule-based sentiment analysis as fallback
                return self._rule_based_sentiment(text)
            
            # Extract features
            features = self._extract_sentiment_features(text)
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            
            # Map prediction to sentiment label
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'neutral')
            
            self.logger.debug(f"Analyzed sentiment for text: {text[:50]}... -> {sentiment}")
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 'neutral'
    
    def _rule_based_sentiment(self, text: str) -> str:
        """Rule-based sentiment analysis as fallback"""
        text_lower = text.lower()
        
        # Positive words
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect',
            'love', 'like', 'happy', 'satisfied', 'pleased', 'thank', 'thanks',
            'awesome', 'fantastic', 'brilliant', 'outstanding', 'superb'
        ]
        
        # Negative words
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated',
            'angry', 'upset', 'annoyed', 'hate', 'dislike', 'problem', 'issue',
            'broken', 'wrong', 'error', 'fail', 'failure', 'poor', 'worst'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_sentiment_features(self, text: str) -> List[float]:
        """Extract features for sentiment analysis"""
        features = []
        
        # Basic text features
        features.append(len(text))  # Text length
        features.append(len(text.split()))  # Word count
        features.append(np.mean([len(word) for word in text.split()]) if text.split() else 0)  # Average word length
        
        # Sentiment indicators
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect']
        features.append(sum(1 for word in positive_words if word in text_lower))
        
        # Negative indicators
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated']
        features.append(sum(1 for word in negative_words if word in text_lower))
        
        # Question indicators
        features.append(1 if '?' in text else 0)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        features.append(sum(1 for word in question_words if word in text_lower))
        
        # Exclamation indicators
        features.append(text.count('!'))
        
        # Capitalization ratio
        features.append(sum(1 for char in text if char.isupper()) / len(text) if text else 0)
        
        # Emoticon indicators
        positive_emoticons = [':)', ':-)', '😊', '😄', '😃', '😀']
        negative_emoticons = [':(', ':-(', '😞', '😢', '😭', '😡']
        
        features.append(sum(1 for emoticon in positive_emoticons if emoticon in text))
        features.append(sum(1 for emoticon in negative_emoticons if emoticon in text))
        
        return features
    
    def train_model(self, training_data: pd.DataFrame, text_column: str = 'text', 
                   label_column: str = 'sentiment') -> Dict[str, Any]:
        """
        Train the sentiment analysis model
        
        Args:
            training_data: DataFrame with training data
            text_column: Name of the text column
            label_column: Name of the sentiment label column
            
        Returns:
            Training results dictionary
        """
        try:
            # Prepare features
            X = []
            y = []
            
            for _, row in training_data.iterrows():
                text = str(row[text_column])
                sentiment = str(row[label_column])
                
                # Extract features
                features = self._extract_sentiment_features(text)
                X.append(features)
                
                # Encode sentiment labels
                sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                y.append(sentiment_map.get(sentiment.lower(), 1))
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            sentiment_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}
            y_test_labels = [sentiment_map_reverse[label] for label in y_test]
            y_pred_labels = [sentiment_map_reverse[label] for label in y_pred]
            
            report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
            
            # Save model
            self._save_model()
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'model_path': self.model_path
            }
            
            self.logger.info(f"Trained sentiment model with accuracy: {accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training sentiment model: {e}")
            raise
    
    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        Get sentiment distribution for a list of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment counts
        """
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            sentiment_counts[sentiment] += 1
        
        return sentiment_counts
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get a sentiment score between -1 and 1
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1, where -1 is very negative, 1 is very positive)
        """
        sentiment = self.analyze_sentiment(text)
        
        if sentiment == 'positive':
            return 1.0
        elif sentiment == 'negative':
            return -1.0
        else:
            return 0.0
    
    def batch_analyze(self, texts: List[str]) -> List[str]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment labels
        """
        return [self.analyze_sentiment(text) for text in texts] 