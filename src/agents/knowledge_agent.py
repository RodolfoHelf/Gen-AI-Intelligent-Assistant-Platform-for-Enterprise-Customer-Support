"""
Knowledge Agent for managing knowledge base operations and product information
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..prompts.templates import KNOWLEDGE_TEMPLATE


class KnowledgeAgent:
    """
    Knowledge Agent for managing knowledge base operations and product information
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.6):
        """
        Initialize the Knowledge Agent
        
        Args:
            api_key: OpenAI API key
            model: Model to use for responses
            temperature: Temperature for response generation
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=temperature
        )
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load all knowledge base components"""
        try:
            knowledge_base = {}
            
            # Load FAQ data
            with open("data/knowledge_base/faq.json", "r") as f:
                knowledge_base["faq"] = json.load(f)
            
            # Load product data
            with open("data/knowledge_base/products.json", "r") as f:
                knowledge_base["products"] = json.load(f)
            
            # Load troubleshooting guides
            with open("data/knowledge_base/troubleshooting_guides.json", "r") as f:
                knowledge_base["troubleshooting"] = json.load(f)
            
            return knowledge_base
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            return {"faq": [], "products": [], "troubleshooting": []}
    
    def get_product_info(self, query: str) -> str:
        """
        Get product information based on customer query
        
        Args:
            query: Customer query about products
            
        Returns:
            Product information response
        """
        try:
            # Find relevant products
            relevant_products = self._find_relevant_products(query)
            
            # Prepare context for LLM
            context = {
                "query": query,
                "relevant_products": relevant_products,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate response
            messages = [
                SystemMessage(content=KNOWLEDGE_TEMPLATE),
                HumanMessage(content=self._format_product_query(query, context))
            ]
            
            response = self.llm(messages)
            
            self.logger.info(f"Generated product info for query: {query[:50]}...")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error getting product info: {e}")
            return "I apologize, but I'm having trouble retrieving product information. Please try again or contact our support team."
    
    def search_knowledge(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information
        
        Args:
            query: Search query
            category: Optional category filter (faq, products, troubleshooting)
            
        Returns:
            List of relevant knowledge entries
        """
        results = []
        query_lower = query.lower()
        
        if category is None or category == "faq":
            for faq in self.knowledge_base.get("faq", []):
                if self._matches_query(faq, query_lower):
                    results.append({
                        "type": "faq",
                        "data": faq,
                        "relevance_score": self._calculate_relevance(faq, query_lower)
                    })
        
        if category is None or category == "products":
            for product in self.knowledge_base.get("products", []):
                if self._matches_query(product, query_lower):
                    results.append({
                        "type": "product",
                        "data": product,
                        "relevance_score": self._calculate_relevance(product, query_lower)
                    })
        
        if category is None or category == "troubleshooting":
            for guide in self.knowledge_base.get("troubleshooting", []):
                if self._matches_query(guide, query_lower):
                    results.append({
                        "type": "troubleshooting",
                        "data": guide,
                        "relevance_score": self._calculate_relevance(guide, query_lower)
                    })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:5]  # Return top 5 results
    
    def _matches_query(self, item: Dict[str, Any], query_lower: str) -> bool:
        """Check if an item matches the query"""
        # Check various fields for matches
        searchable_fields = []
        
        if "question" in item:
            searchable_fields.extend([item["question"], item.get("answer", "")])
        elif "name" in item:
            searchable_fields.extend([item["name"], item.get("description", "")])
        elif "title" in item:
            searchable_fields.extend([item["title"], item.get("category", "")])
        
        # Check tags if available
        if "tags" in item:
            searchable_fields.extend(item["tags"])
        
        # Check if any field contains query words
        for field in searchable_fields:
            if any(word in field.lower() for word in query_lower.split()):
                return True
        
        return False
    
    def _calculate_relevance(self, item: Dict[str, Any], query_lower: str) -> float:
        """Calculate relevance score for an item"""
        score = 0.0
        query_words = set(query_lower.split())
        
        # Check various fields
        if "question" in item:
            question_words = set(item["question"].lower().split())
            answer_words = set(item.get("answer", "").lower().split())
            score += len(query_words.intersection(question_words)) * 2
            score += len(query_words.intersection(answer_words))
        elif "name" in item:
            name_words = set(item["name"].lower().split())
            desc_words = set(item.get("description", "").lower().split())
            score += len(query_words.intersection(name_words)) * 2
            score += len(query_words.intersection(desc_words))
        elif "title" in item:
            title_words = set(item["title"].lower().split())
            category_words = set(item.get("category", "").lower().split())
            score += len(query_words.intersection(title_words)) * 2
            score += len(query_words.intersection(category_words))
        
        # Check tags
        if "tags" in item:
            tag_words = set()
            for tag in item["tags"]:
                tag_words.update(tag.lower().split())
            score += len(query_words.intersection(tag_words)) * 1.5
        
        return score
    
    def _find_relevant_products(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant products for the query"""
        relevant_products = []
        query_lower = query.lower()
        
        for product in self.knowledge_base.get("products", []):
            # Check product name and description
            product_name_lower = product["name"].lower()
            product_desc_lower = product.get("description", "").lower()
            
            # Check if query mentions product name or features
            if any(word in product_name_lower for word in query_lower.split()):
                relevant_products.append(product)
            elif any(word in product_desc_lower for word in query_lower.split()):
                relevant_products.append(product)
            elif any(feature.lower() in query_lower for feature in product.get("features", [])):
                relevant_products.append(product)
        
        return relevant_products[:3]  # Return top 3 most relevant
    
    def _format_product_query(self, query: str, context: Dict[str, Any]) -> str:
        """Format the product query for the LLM"""
        formatted_context = f"""
Customer Query: {query}

Relevant Products:
"""
        
        for product in context.get("relevant_products", []):
            formatted_context += f"""
Product: {product['name']}
Description: {product['description']}
Price: ${product['price']} {product['currency']}
Features: {', '.join(product.get('features', []))}
Specifications: {json.dumps(product.get('specifications', {}), indent=2)}
"""
        
        return formatted_context
    
    def update_knowledge_base(self, entry_type: str, entry_data: Dict[str, Any]) -> bool:
        """
        Update the knowledge base with new information
        
        Args:
            entry_type: Type of entry (faq, product, troubleshooting)
            entry_data: Entry data to add
            
        Returns:
            Success status
        """
        try:
            # Validate entry data
            if not self._validate_entry(entry_type, entry_data):
                return False
            
            # Add timestamp
            entry_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            # Load current data
            file_path = f"data/knowledge_base/{entry_type}.json"
            with open(file_path, "r") as f:
                current_data = json.load(f)
            
            # Add new entry
            if entry_type == "faq":
                entry_data["id"] = f"faq_{len(current_data) + 1:03d}"
            elif entry_type == "products":
                entry_data["id"] = f"prod_{len(current_data) + 1:03d}"
            elif entry_type == "troubleshooting":
                entry_data["id"] = f"trouble_{len(current_data) + 1:03d}"
            
            current_data.append(entry_data)
            
            # Save updated data
            with open(file_path, "w") as f:
                json.dump(current_data, f, indent=2)
            
            # Reload knowledge base
            self.knowledge_base = self._load_knowledge_base()
            
            self.logger.info(f"Updated knowledge base with new {entry_type} entry")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge base: {e}")
            return False
    
    def _validate_entry(self, entry_type: str, entry_data: Dict[str, Any]) -> bool:
        """Validate entry data before adding to knowledge base"""
        required_fields = {
            "faq": ["question", "answer", "category"],
            "products": ["name", "description", "category", "price"],
            "troubleshooting": ["title", "category", "severity", "steps"]
        }
        
        if entry_type not in required_fields:
            return False
        
        for field in required_fields[entry_type]:
            if field not in entry_data:
                return False
        
        return True
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            "total_entries": 0,
            "by_category": {},
            "last_updated": datetime.now().isoformat()
        }
        
        for category, data in self.knowledge_base.items():
            stats["total_entries"] += len(data)
            stats["by_category"][category] = len(data)
        
        return stats 