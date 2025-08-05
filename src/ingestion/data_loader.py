"""
Data loader for loading data from various sources
"""

import json
import csv
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
import pymongo
from pymongo import MongoClient


class DataLoader:
    """
    Data loader for loading data from various sources
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.logger = logging.getLogger(__name__)
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"Successfully loaded CSV file: {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    def load_json(self, file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            JSON data as dictionary or list
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    def load_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Load data from database using SQL query
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            
        Returns:
            Pandas DataFrame
        """
        try:
            engine = create_engine(connection_string)
            with engine.connect() as connection:
                df = pd.read_sql(text(query), connection)
            self.logger.info(f"Successfully loaded data from database")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from database: {e}")
            raise
    
    def load_mongodb(self, connection_string: str, database: str, collection: str, 
                    query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Load data from MongoDB
        
        Args:
            connection_string: MongoDB connection string
            database: Database name
            collection: Collection name
            query: Optional query filter
            
        Returns:
            List of documents
        """
        try:
            client = MongoClient(connection_string)
            db = client[database]
            coll = db[collection]
            
            if query:
                documents = list(coll.find(query))
            else:
                documents = list(coll.find())
            
            self.logger.info(f"Successfully loaded {len(documents)} documents from MongoDB")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading data from MongoDB: {e}")
            raise
    
    def load_knowledge_base(self, base_path: str = "data/knowledge_base") -> Dict[str, Any]:
        """
        Load knowledge base from JSON files
        
        Args:
            base_path: Base path to knowledge base directory
            
        Returns:
            Dictionary containing all knowledge base data
        """
        try:
            knowledge_base = {}
            base_path = Path(base_path)
            
            # Load FAQ data
            faq_path = base_path / "faq.json"
            if faq_path.exists():
                knowledge_base["faq"] = self.load_json(str(faq_path))
            
            # Load product data
            products_path = base_path / "products.json"
            if products_path.exists():
                knowledge_base["products"] = self.load_json(str(products_path))
            
            # Load troubleshooting guides
            troubleshooting_path = base_path / "troubleshooting_guides.json"
            if troubleshooting_path.exists():
                knowledge_base["troubleshooting"] = self.load_json(str(troubleshooting_path))
            
            self.logger.info(f"Successfully loaded knowledge base with {len(knowledge_base)} categories")
            return knowledge_base
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            raise
    
    def load_customer_data(self, file_path: str) -> pd.DataFrame:
        """
        Load customer support data
        
        Args:
            file_path: Path to customer data file
            
        Returns:
            Pandas DataFrame with customer data
        """
        try:
            if file_path.endswith('.csv'):
                df = self.load_csv(file_path)
            elif file_path.endswith('.json'):
                data = self.load_json(file_path)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Ensure required columns exist
            required_columns = ['query', 'response', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns in customer data: {missing_columns}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading customer data: {e}")
            raise
    
    def load_conversation_history(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load conversation history data
        
        Args:
            file_path: Path to conversation history file
            
        Returns:
            List of conversation records
        """
        try:
            if file_path.endswith('.json'):
                conversations = self.load_json(file_path)
            else:
                raise ValueError(f"Conversation history must be in JSON format: {file_path}")
            
            # Validate conversation structure
            for conv in conversations:
                if not isinstance(conv, dict) or 'messages' not in conv:
                    raise ValueError("Invalid conversation structure")
            
            self.logger.info(f"Successfully loaded {len(conversations)} conversations")
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error loading conversation history: {e}")
            raise
    
    def save_data(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], 
                  file_path: str, format: str = 'auto') -> bool:
        """
        Save data to file
        
        Args:
            data: Data to save
            file_path: Path to save file
            format: File format ('csv', 'json', 'auto')
            
        Returns:
            Success status
        """
        try:
            if format == 'auto':
                format = Path(file_path).suffix[1:]
            
            if format == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif format == 'json':
                if isinstance(data, pd.DataFrame):
                    data.to_json(file_path, orient='records', indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Successfully saved data to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")
            return False
    
    def get_data_info(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get information about loaded data
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with data information
        """
        info = {
            "type": type(data).__name__,
            "size": None,
            "columns": None,
            "missing_values": None
        }
        
        if isinstance(data, pd.DataFrame):
            info["size"] = data.shape
            info["columns"] = list(data.columns)
            info["missing_values"] = data.isnull().sum().to_dict()
        elif isinstance(data, list):
            info["size"] = len(data)
            if data and isinstance(data[0], dict):
                info["columns"] = list(data[0].keys())
        elif isinstance(data, dict):
            info["size"] = len(data)
            info["columns"] = list(data.keys())
        
        return info 