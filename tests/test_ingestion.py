"""
Tests for data ingestion components
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.data_loader import DataLoader
from src.ingestion.preprocessor import DataPreprocessor


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_loader = DataLoader()
        
        # Create temporary test data
        self.test_csv_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['John', 'Jane', 'Bob'],
            'age': [25, 30, 35],
            'city': ['New York', 'Los Angeles', 'Chicago']
        })
        
        self.test_json_data = [
            {'id': 1, 'name': 'John', 'age': 25, 'city': 'New York'},
            {'id': 2, 'name': 'Jane', 'age': 30, 'city': 'Los Angeles'},
            {'id': 3, 'name': 'Bob', 'age': 35, 'city': 'Chicago'}
        ]
    
    def test_load_csv(self):
        """Test loading CSV data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_csv_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            df = self.data_loader.load_csv(csv_path)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 3)
            self.assertEqual(list(df.columns), ['id', 'name', 'age', 'city'])
        finally:
            os.unlink(csv_path)
    
    def test_load_json(self):
        """Test loading JSON data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_json_data, f)
            json_path = f.name
        
        try:
            data = self.data_loader.load_json(json_path)
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]['name'], 'John')
        finally:
            os.unlink(json_path)
    
    def test_load_knowledge_base(self):
        """Test loading knowledge base"""
        # Create temporary knowledge base files
        faq_data = [
            {'id': 'faq_001', 'question': 'Test question', 'answer': 'Test answer'}
        ]
        
        products_data = [
            {'id': 'prod_001', 'name': 'Test Product', 'description': 'Test description'}
        ]
        
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        kb_dir = os.path.join(temp_dir, 'data', 'knowledge_base')
        os.makedirs(kb_dir, exist_ok=True)
        
        try:
            # Create test files
            with open(os.path.join(kb_dir, 'faq.json'), 'w') as f:
                json.dump(faq_data, f)
            
            with open(os.path.join(kb_dir, 'products.json'), 'w') as f:
                json.dump(products_data, f)
            
            # Test loading
            kb = self.data_loader.load_knowledge_base(kb_dir)
            self.assertIsInstance(kb, dict)
            self.assertIn('faq', kb)
            self.assertIn('products', kb)
            self.assertEqual(len(kb['faq']), 1)
            self.assertEqual(len(kb['products']), 1)
        
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_save_data(self):
        """Test saving data"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            # Test saving DataFrame
            success = self.data_loader.save_data(self.test_csv_data, csv_path)
            self.assertTrue(success)
            
            # Verify saved data
            saved_df = pd.read_csv(csv_path)
            self.assertEqual(len(saved_df), 3)
        
        finally:
            os.unlink(csv_path)
    
    def test_get_data_info(self):
        """Test getting data information"""
        info = self.data_loader.get_data_info(self.test_csv_data)
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['type'], 'DataFrame')
        self.assertEqual(info['size'], (3, 4))
        self.assertEqual(info['columns'], ['id', 'name', 'age', 'city'])


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Create test data
        self.test_df = pd.DataFrame({
            'text': ['Hello world!', 'This is a test.', 'Another test case.'],
            'category': ['A', 'B', 'A'],
            'score': [1.5, 2.0, 1.8],
            'missing_col': [1, np.nan, 3]
        })
    
    def test_clean_text(self):
        """Test text cleaning"""
        test_text = "Hello, World! This is a test."
        cleaned_text = self.preprocessor.clean_text(test_text)
        
        self.assertIsInstance(cleaned_text, str)
        self.assertNotIn(',', cleaned_text)
        self.assertNotIn('!', cleaned_text)
    
    def test_clean_dataframe(self):
        """Test DataFrame cleaning"""
        df_clean = self.preprocessor.clean_dataframe(
            self.test_df, 
            text_columns=['text']
        )
        
        self.assertIsInstance(df_clean, pd.DataFrame)
        self.assertEqual(len(df_clean), 3)
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Test drop strategy
        df_drop = self.preprocessor.handle_missing_values(
            self.test_df, 
            strategy='drop'
        )
        self.assertEqual(len(df_drop), 2)  # One row with NaN should be dropped
        
        # Test fill strategy
        df_fill = self.preprocessor.handle_missing_values(
            self.test_df, 
            strategy='fill'
        )
        self.assertEqual(len(df_fill), 3)  # All rows should remain
        self.assertFalse(df_fill['missing_col'].isna().any())
    
    def test_encode_categorical(self):
        """Test categorical encoding"""
        df_encoded = self.preprocessor.encode_categorical(
            self.test_df, 
            ['category']
        )
        
        self.assertIsInstance(df_encoded, pd.DataFrame)
        # Check that categorical column is now numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df_encoded['category']))
    
    def test_scale_numerical(self):
        """Test numerical scaling"""
        df_scaled = self.preprocessor.scale_numerical(
            self.test_df, 
            ['score']
        )
        
        self.assertIsInstance(df_scaled, pd.DataFrame)
        # Check that scaled values are different from original
        self.assertFalse(np.array_equal(
            df_scaled['score'], 
            self.test_df['score']
        ))
    
    def test_vectorize_text(self):
        """Test text vectorization"""
        df_vectorized = self.preprocessor.vectorize_text(
            self.test_df, 
            'text'
        )
        
        self.assertIsInstance(df_vectorized, pd.DataFrame)
        # Check that text column is removed and new features are added
        self.assertNotIn('text', df_vectorized.columns)
        self.assertGreater(len(df_vectorized.columns), 3)  # Should have more features
    
    def test_extract_features(self):
        """Test feature extraction"""
        text = "This is a test message with some positive words like good and great."
        features = self.preprocessor.extract_features(text)
        
        self.assertIsInstance(features, dict)
        self.assertIn('length', features)
        self.assertIn('word_count', features)
        self.assertIn('positive_word_count', features)
        self.assertIn('negative_word_count', features)
    
    def test_prepare_customer_support_data(self):
        """Test customer support data preparation"""
        prepared_data = self.preprocessor.prepare_customer_support_data(
            self.test_df,
            text_column='text',
            label_column='category'
        )
        
        self.assertIsInstance(prepared_data, dict)
        self.assertIn('X', prepared_data)
        self.assertIn('y', prepared_data)
        self.assertIn('feature_columns', prepared_data)
        
        # Check that X and y have correct shapes
        self.assertEqual(len(prepared_data['X']), 3)
        self.assertEqual(len(prepared_data['y']), 3)
    
    def test_get_preprocessing_summary(self):
        """Test preprocessing summary"""
        df_processed = self.preprocessor.clean_dataframe(self.test_df)
        summary = self.preprocessor.get_preprocessing_summary(
            self.test_df, 
            df_processed
        )
        
        self.assertIsInstance(summary, dict)
        self.assertIn('original_shape', summary)
        self.assertIn('processed_shape', summary)
        self.assertIn('rows_removed', summary)
        self.assertIn('columns_removed', summary)


if __name__ == '__main__':
    unittest.main() 