"""
Data preprocessor for cleaning and preparing data for ML models
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataPreprocessor:
    """
    Data preprocessor for cleaning and preparing data for ML models
    """
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional - can be modified based on requirements)
        # text = re.sub(r'\d+', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens).strip()
    
    def clean_dataframe(self, df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean a pandas DataFrame
        
        Args:
            df: DataFrame to clean
            text_columns: List of text columns to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Clean text columns
        if text_columns:
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).apply(self.clean_text)
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        self.logger.info(f"Cleaned DataFrame: {df.shape} -> {df_clean.shape}")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in DataFrame
        
        Args:
            df: DataFrame to process
            strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
            
        Returns:
            DataFrame with handled missing values
        """
        df_processed = df.copy()
        
        if strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'fill':
            # Fill numeric columns with median
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Fill categorical columns with mode
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown')
        elif strategy == 'interpolate':
            df_processed = df_processed.interpolate()
        
        self.logger.info(f"Handled missing values using strategy: {strategy}")
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: DataFrame to encode
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle new categories
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        self.logger.info(f"Encoded categorical columns: {categorical_columns}")
        return df_encoded
    
    def scale_numerical(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """
        Scale numerical variables
        
        Args:
            df: DataFrame to scale
            numerical_columns: List of numerical column names
            
        Returns:
            DataFrame with scaled numerical variables
        """
        df_scaled = df.copy()
        
        if numerical_columns:
            # Fit scaler if not already fitted
            if not hasattr(self.scaler, 'mean_'):
                df_scaled[numerical_columns] = self.scaler.fit_transform(df_scaled[numerical_columns])
            else:
                df_scaled[numerical_columns] = self.scaler.transform(df_scaled[numerical_columns])
        
        self.logger.info(f"Scaled numerical columns: {numerical_columns}")
        return df_scaled
    
    def vectorize_text(self, df: pd.DataFrame, text_column: str, max_features: int = 1000) -> pd.DataFrame:
        """
        Vectorize text data using TF-IDF
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the text column
            max_features: Maximum number of features for TF-IDF
            
        Returns:
            DataFrame with vectorized text features
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        
        # Fit and transform the text data
        text_features = vectorizer.fit_transform(df[text_column])
        
        # Convert to DataFrame
        feature_names = vectorizer.get_feature_names_out()
        text_df = pd.DataFrame(text_features.toarray(), columns=feature_names)
        
        # Combine with original DataFrame
        df_vectorized = pd.concat([df.drop(columns=[text_column]), text_df], axis=1)
        
        self.logger.info(f"Vectorized text column '{text_column}' with {len(feature_names)} features")
        return df_vectorized
    
    def prepare_customer_support_data(self, df: pd.DataFrame, text_column: str = 'query', 
                                    label_column: str = 'category') -> Dict[str, Any]:
        """
        Prepare customer support data for ML models
        
        Args:
            df: DataFrame with customer support data
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Dictionary containing prepared data and preprocessing objects
        """
        try:
            # Clean the data
            df_clean = self.clean_dataframe(df, text_columns=[text_column])
            
            # Handle missing values
            df_clean = self.handle_missing_values(df_clean, strategy='fill')
            
            # Encode categorical labels
            if label_column in df_clean.columns:
                df_clean = self.encode_categorical(df_clean, [label_column])
            
            # Vectorize text
            df_vectorized = self.vectorize_text(df_clean, text_column)
            
            # Prepare features and labels
            feature_columns = [col for col in df_vectorized.columns if col != label_column]
            X = df_vectorized[feature_columns]
            y = df_vectorized[label_column] if label_column in df_vectorized.columns else None
            
            prepared_data = {
                'X': X,
                'y': y,
                'feature_columns': feature_columns,
                'label_encoder': self.label_encoders.get(label_column),
                'vectorizer': self.vectorizer,
                'preprocessed_df': df_vectorized
            }
            
            self.logger.info(f"Prepared customer support data: {X.shape}")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Error preparing customer support data: {e}")
            raise
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text for analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Sentiment indicators (simple heuristics)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated']
        
        text_lower = text.lower()
        features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
        features['negative_word_count'] = sum(1 for word in negative_words if word in text_lower)
        
        # Question indicators
        features['is_question'] = 1 if '?' in text else 0
        features['question_words'] = sum(1 for word in ['what', 'how', 'why', 'when', 'where', 'who'] if word in text_lower)
        
        # Urgency indicators
        urgency_words = ['urgent', 'emergency', 'immediately', 'asap', 'critical']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
        
        return features
    
    def get_preprocessing_summary(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of preprocessing steps
        
        Args:
            df_original: Original DataFrame
            df_processed: Processed DataFrame
            
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'rows_removed': df_original.shape[0] - df_processed.shape[0],
            'columns_removed': df_original.shape[1] - df_processed.shape[1],
            'missing_values_original': df_original.isnull().sum().sum(),
            'missing_values_processed': df_processed.isnull().sum().sum(),
            'duplicates_removed': df_original.duplicated().sum()
        }
        
        return summary 