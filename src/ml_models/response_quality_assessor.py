"""
Response Quality Assessor for evaluating response quality
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

from ..ingestion.preprocessor import DataPreprocessor


class ResponseQualityAssessor:
    """
    Response Quality Assessor for evaluating response quality
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Response Quality Assessor
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.vectorizer = None
        self.preprocessor = DataPreprocessor()
        self.model_path = model_path or "models/quality_model.pkl"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load pre-trained model if available
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.vectorizer = model_data['vectorizer']
                self.logger.info(f"Loaded pre-trained quality model from {self.model_path}")
            else:
                self.logger.info("No pre-trained model found. Model will be trained when data is provided.")
        except Exception as e:
            self.logger.error(f"Error loading quality model: {e}")
    
    def _save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Saved quality model to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving quality model: {e}")
    
    def assess_quality(self, query: str, response: str) -> float:
        """
        Assess the quality of a response to a query
        
        Args:
            query: Original customer query
            response: Generated response
            
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        try:
            if self.model is None:
                # Use rule-based quality assessment as fallback
                return self._rule_based_quality_assessment(query, response)
            
            # Extract features
            features = self._extract_quality_features(query, response)
            
            # Make prediction
            quality_score = self.model.predict([features])[0]
            
            # Ensure score is between 0 and 1
            quality_score = max(0.0, min(1.0, quality_score))
            
            self.logger.debug(f"Assessed quality for query: {query[:50]}... -> {quality_score:.3f}")
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error assessing quality: {e}")
            return 0.5
    
    def _rule_based_quality_assessment(self, query: str, response: str) -> float:
        """Rule-based quality assessment as fallback"""
        score = 0.5  # Base score
        
        # Length score (prefer medium-length responses)
        response_length = len(response)
        if 50 <= response_length <= 500:
            score += 0.2
        elif response_length < 20:
            score -= 0.3
        elif response_length > 1000:
            score -= 0.1
        
        # Relevance score (check if response addresses query keywords)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        keyword_overlap = len(query_words.intersection(response_words))
        if keyword_overlap > 0:
            score += min(0.3, keyword_overlap * 0.1)
        
        # Completeness score (check for common response patterns)
        if any(phrase in response.lower() for phrase in ['i understand', 'let me help', 'here is', 'you can']):
            score += 0.1
        
        # Professionalism score
        if any(word in response.lower() for word in ['please', 'thank you', 'appreciate']):
            score += 0.1
        
        # Penalize generic responses
        generic_phrases = ['i don\'t know', 'cannot help', 'contact support']
        if any(phrase in response.lower() for phrase in generic_phrases):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _extract_quality_features(self, query: str, response: str) -> List[float]:
        """Extract features for quality assessment"""
        features = []
        
        # Query features
        features.append(len(query))  # Query length
        features.append(len(query.split()))  # Query word count
        
        # Response features
        features.append(len(response))  # Response length
        features.append(len(response.split()))  # Response word count
        features.append(np.mean([len(word) for word in response.split()]) if response.split() else 0)  # Average word length
        
        # Relevance features
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        features.append(len(query_words.intersection(response_words)))  # Keyword overlap
        features.append(len(query_words.intersection(response_words)) / max(len(query_words), 1))  # Overlap ratio
        
        # Completeness features
        features.append(1 if '?' in query and any(word in response.lower() for word in ['answer', 'solution', 'help']) else 0)
        features.append(1 if any(word in query.lower() for word in ['how', 'what', 'why']) and len(response) > 50 else 0)
        
        # Professionalism features
        professional_words = ['please', 'thank you', 'appreciate', 'understand', 'help']
        features.append(sum(1 for word in professional_words if word in response.lower()))
        
        # Specificity features
        specific_indicators = ['specifically', 'for example', 'in detail', 'step by step']
        features.append(sum(1 for indicator in specific_indicators if indicator in response.lower()))
        
        # Politeness features
        polite_words = ['please', 'thank you', 'sorry', 'apologize', 'regret']
        features.append(sum(1 for word in polite_words if word in response.lower()))
        
        # Clarity features
        clarity_indicators = ['clearly', 'simply', 'easily', 'straightforward']
        features.append(sum(1 for indicator in clarity_indicators if indicator in response.lower()))
        
        return features
    
    def train_model(self, training_data: pd.DataFrame, query_column: str = 'query', 
                   response_column: str = 'response', quality_column: str = 'quality') -> Dict[str, Any]:
        """
        Train the quality assessment model
        
        Args:
            training_data: DataFrame with training data
            query_column: Name of the query column
            response_column: Name of the response column
            quality_column: Name of the quality score column
            
        Returns:
            Training results dictionary
        """
        try:
            # Prepare features
            X = []
            y = []
            
            for _, row in training_data.iterrows():
                query = str(row[query_column])
                response = str(row[response_column])
                quality = float(row[quality_column])
                
                # Extract features
                features = self._extract_quality_features(query, response)
                X.append(features)
                y.append(quality)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            self._save_model()
            
            results = {
                'mse': mse,
                'r2_score': r2,
                'model_path': self.model_path,
                'feature_count': X.shape[1]
            }
            
            self.logger.info(f"Trained quality model with R² score: {r2:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training quality model: {e}")
            raise
    
    def get_quality_metrics(self, query: str, response: str) -> Dict[str, Any]:
        """
        Get detailed quality metrics for a query-response pair
        
        Args:
            query: Customer query
            response: Generated response
            
        Returns:
            Dictionary with quality metrics
        """
        quality_score = self.assess_quality(query, response)
        
        # Calculate additional metrics
        query_length = len(query)
        response_length = len(response)
        query_words = len(query.split())
        response_words = len(response.split())
        
        # Relevance metrics
        query_keywords = set(query.lower().split())
        response_keywords = set(response.lower().split())
        keyword_overlap = len(query_keywords.intersection(response_keywords))
        relevance_ratio = keyword_overlap / max(len(query_keywords), 1)
        
        # Professionalism metrics
        professional_words = ['please', 'thank you', 'appreciate', 'understand', 'help']
        professional_count = sum(1 for word in professional_words if word in response.lower())
        
        metrics = {
            'overall_quality': quality_score,
            'query_length': query_length,
            'response_length': response_length,
            'query_word_count': query_words,
            'response_word_count': response_words,
            'keyword_overlap': keyword_overlap,
            'relevance_ratio': relevance_ratio,
            'professional_word_count': professional_count,
            'response_to_query_ratio': response_length / max(query_length, 1)
        }
        
        return metrics
    
    def batch_assess(self, query_response_pairs: List[Dict[str, str]]) -> List[float]:
        """
        Assess quality for multiple query-response pairs
        
        Args:
            query_response_pairs: List of dictionaries with 'query' and 'response' keys
            
        Returns:
            List of quality scores
        """
        quality_scores = []
        
        for pair in query_response_pairs:
            query = pair.get('query', '')
            response = pair.get('response', '')
            quality_score = self.assess_quality(query, response)
            quality_scores.append(quality_score)
        
        return quality_scores
    
    def get_quality_distribution(self, query_response_pairs: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Get quality distribution for multiple query-response pairs
        
        Args:
            query_response_pairs: List of dictionaries with 'query' and 'response' keys
            
        Returns:
            Dictionary with quality distribution
        """
        quality_scores = self.batch_assess(query_response_pairs)
        
        # Categorize quality scores
        distribution = {
            'excellent': 0,  # 0.8-1.0
            'good': 0,       # 0.6-0.8
            'fair': 0,       # 0.4-0.6
            'poor': 0        # 0.0-0.4
        }
        
        for score in quality_scores:
            if score >= 0.8:
                distribution['excellent'] += 1
            elif score >= 0.6:
                distribution['good'] += 1
            elif score >= 0.4:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def validate_quality(self, query: str, response: str, expected_quality: float) -> Dict[str, Any]:
        """
        Validate quality assessment against expected quality
        
        Args:
            query: Customer query
            response: Generated response
            expected_quality: Expected quality score
            
        Returns:
            Validation results dictionary
        """
        predicted_quality = self.assess_quality(query, response)
        metrics = self.get_quality_metrics(query, response)
        
        error = abs(predicted_quality - expected_quality)
        
        return {
            'query': query,
            'response': response,
            'predicted_quality': predicted_quality,
            'expected_quality': expected_quality,
            'error': error,
            'metrics': metrics
        } 