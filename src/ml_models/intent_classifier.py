"""
Intent Classifier for categorizing customer queries
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

from ..ingestion.preprocessor import DataPreprocessor


class IntentClassifier:
    """
    Intent Classifier for categorizing customer queries
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Intent Classifier
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.preprocessor = DataPreprocessor()
        self.model_path = model_path or "models/intent_model.pkl"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Define intent categories
        self.intent_categories = [
            'general_inquiry',
            'technical_support',
            'billing_inquiry',
            'product_information',
            'troubleshooting',
            'escalation',
            'complaint',
            'feature_request',
            'account_management',
            'order_status'
        ]
        
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
                    self.label_encoder = model_data['label_encoder']
                self.logger.info(f"Loaded pre-trained intent model from {self.model_path}")
            else:
                self.logger.info("No pre-trained model found. Model will be trained when data is provided.")
        except Exception as e:
            self.logger.error(f"Error loading intent model: {e}")
    
    def _save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Saved intent model to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving intent model: {e}")
    
    def classify_intent(self, text: str) -> str:
        """
        Classify the intent of the given text
        
        Args:
            text: Text to classify
            
        Returns:
            Intent category
        """
        try:
            if self.model is None:
                # Use rule-based intent classification as fallback
                return self._rule_based_intent_classification(text)
            
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Vectorize text
            text_features = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.model.predict(text_features)[0]
            
            # Decode prediction
            intent = self.label_encoder.inverse_transform([prediction])[0]
            
            self.logger.debug(f"Classified intent for text: {text[:50]}... -> {intent}")
            return intent
            
        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}")
            return 'general_inquiry'
    
    def _rule_based_intent_classification(self, text: str) -> str:
        """Rule-based intent classification as fallback"""
        text_lower = text.lower()
        
        # Define keyword patterns for each intent
        intent_patterns = {
            'technical_support': ['error', 'bug', 'crash', 'broken', 'not working', 'problem', 'issue'],
            'billing_inquiry': ['bill', 'payment', 'charge', 'cost', 'price', 'refund', 'money'],
            'product_information': ['feature', 'specification', 'details', 'information', 'what is'],
            'troubleshooting': ['how to', 'fix', 'solve', 'troubleshoot', 'help with'],
            'escalation': ['manager', 'supervisor', 'escalate', 'speak to someone', 'human'],
            'complaint': ['complaint', 'unhappy', 'dissatisfied', 'angry', 'frustrated'],
            'feature_request': ['request', 'suggestion', 'improvement', 'new feature', 'add'],
            'account_management': ['account', 'profile', 'settings', 'password', 'login'],
            'order_status': ['order', 'tracking', 'delivery', 'shipping', 'status']
        }
        
        # Check for patterns
        for intent, patterns in intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        
        return 'general_inquiry'
    
    def train_model(self, training_data: pd.DataFrame, text_column: str = 'text', 
                   label_column: str = 'intent') -> Dict[str, Any]:
        """
        Train the intent classification model
        
        Args:
            training_data: DataFrame with training data
            text_column: Name of the text column
            label_column: Name of the intent label column
            
        Returns:
            Training results dictionary
        """
        try:
            # Prepare data
            texts = training_data[text_column].astype(str).tolist()
            labels = training_data[label_column].astype(str).tolist()
            
            # Clean texts
            cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
            
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = self.vectorizer.fit_transform(cleaned_texts)
            
            # Create and fit label encoder
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            
            report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
            
            # Save model
            self._save_model()
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'model_path': self.model_path,
                'feature_count': X.shape[1],
                'class_count': len(self.label_encoder.classes_)
            }
            
            self.logger.info(f"Trained intent model with accuracy: {accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training intent model: {e}")
            raise
    
    def get_intent_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get probability scores for all intent categories
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with intent probabilities
        """
        try:
            if self.model is None:
                return {intent: 0.0 for intent in self.intent_categories}
            
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Vectorize text
            text_features = self.vectorizer.transform([cleaned_text])
            
            # Get probabilities
            probabilities = self.model.predict_proba(text_features)[0]
            
            # Create probability dictionary
            intent_probs = {}
            for i, intent in enumerate(self.label_encoder.classes_):
                intent_probs[intent] = float(probabilities[i])
            
            return intent_probs
            
        except Exception as e:
            self.logger.error(f"Error getting intent probabilities: {e}")
            return {intent: 0.0 for intent in self.intent_categories}
    
    def get_top_intents(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top-k intent classifications with probabilities
        
        Args:
            text: Text to classify
            top_k: Number of top intents to return
            
        Returns:
            List of dictionaries with intent and probability
        """
        probabilities = self.get_intent_probabilities(text)
        
        # Sort by probability
        sorted_intents = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        top_intents = []
        for intent, probability in sorted_intents[:top_k]:
            top_intents.append({
                'intent': intent,
                'probability': probability
            })
        
        return top_intents
    
    def batch_classify(self, texts: List[str]) -> List[str]:
        """
        Classify intent for multiple texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of intent labels
        """
        return [self.classify_intent(text) for text in texts]
    
    def get_intent_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        Get intent distribution for a list of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with intent counts
        """
        intent_counts = {intent: 0 for intent in self.intent_categories}
        
        for text in texts:
            intent = self.classify_intent(text)
            if intent in intent_counts:
                intent_counts[intent] += 1
        
        return intent_counts
    
    def validate_intent(self, text: str, expected_intent: str) -> Dict[str, Any]:
        """
        Validate intent classification against expected intent
        
        Args:
            text: Text to classify
            expected_intent: Expected intent label
            
        Returns:
            Validation results dictionary
        """
        predicted_intent = self.classify_intent(text)
        probabilities = self.get_intent_probabilities(text)
        
        is_correct = predicted_intent == expected_intent
        confidence = probabilities.get(predicted_intent, 0.0)
        
        return {
            'text': text,
            'predicted_intent': predicted_intent,
            'expected_intent': expected_intent,
            'is_correct': is_correct,
            'confidence': confidence,
            'all_probabilities': probabilities
        } 