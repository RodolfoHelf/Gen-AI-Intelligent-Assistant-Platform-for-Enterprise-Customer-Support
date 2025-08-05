"""
Customer Support Agent for handling general customer inquiries
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from ..prompts.templates import CUSTOMER_SUPPORT_TEMPLATE
from ..prompts.utils import PromptUtils


class CustomerSupportAgent:
    """
    Customer Support Agent for handling general customer inquiries
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        """
        Initialize the Customer Support Agent
        
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
        
        self.prompt_utils = PromptUtils()
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load FAQ and product knowledge base"""
        try:
            # Load FAQ data
            with open("data/knowledge_base/faq.json", "r") as f:
                faq_data = json.load(f)
            
            # Load product data
            with open("data/knowledge_base/products.json", "r") as f:
                product_data = json.load(f)
            
            return {
                "faq": faq_data,
                "products": product_data
            }
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            return {"faq": [], "products": []}
    
    def handle_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle a customer support query
        
        Args:
            query: Customer query
            context: Optional context information
            
        Returns:
            Response to the customer
        """
        try:
            # Find relevant knowledge
            relevant_faq = self._find_relevant_faq(query)
            relevant_products = self._find_relevant_products(query)
            
            # Prepare context
            context_info = {
                "query": query,
                "relevant_faq": relevant_faq,
                "relevant_products": relevant_products,
                "timestamp": datetime.now().isoformat()
            }
            
            if context:
                context_info.update(context)
            
            # Generate response using LangChain
            messages = [
                SystemMessage(content=CUSTOMER_SUPPORT_TEMPLATE),
                HumanMessage(content=self._format_query_for_llm(query, context_info))
            ]
            
            response = self.llm(messages)
            
            self.logger.info(f"Generated response for query: {query[:50]}...")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error handling query: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again or contact our support team."
    
    def _find_relevant_faq(self, query: str) -> list:
        """Find relevant FAQ entries for the query"""
        relevant_faq = []
        
        for faq in self.knowledge_base.get("faq", []):
            # Simple keyword matching
            query_lower = query.lower()
            question_lower = faq["question"].lower()
            
            # Check if any keywords match
            if any(keyword in query_lower for keyword in faq.get("tags", [])):
                relevant_faq.append(faq)
            elif any(word in question_lower for word in query_lower.split()):
                relevant_faq.append(faq)
        
        return relevant_faq[:3]  # Return top 3 most relevant
    
    def _find_relevant_products(self, query: str) -> list:
        """Find relevant products for the query"""
        relevant_products = []
        
        for product in self.knowledge_base.get("products", []):
            query_lower = query.lower()
            product_name_lower = product["name"].lower()
            product_desc_lower = product["description"].lower()
            
            # Check if product name or description matches query
            if any(word in product_name_lower for word in query_lower.split()):
                relevant_products.append(product)
            elif any(word in product_desc_lower for word in query_lower.split()):
                relevant_products.append(product)
        
        return relevant_products[:2]  # Return top 2 most relevant
    
    def _format_query_for_llm(self, query: str, context: Dict[str, Any]) -> str:
        """Format the query and context for the LLM"""
        formatted_context = f"""
Query: {query}

Relevant FAQ:
"""
        
        for faq in context.get("relevant_faq", []):
            formatted_context += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
        
        formatted_context += "Relevant Products:\n"
        for product in context.get("relevant_products", []):
            formatted_context += f"- {product['name']}: {product['description']}\n"
        
        return formatted_context
    
    def get_response_quality(self, query: str, response: str) -> float:
        """
        Assess the quality of a response
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            Quality score between 0 and 1
        """
        # Simple quality assessment based on response length and relevance
        if not response or len(response.strip()) < 10:
            return 0.1
        
        # Check if response addresses the query
        query_keywords = set(query.lower().split())
        response_keywords = set(response.lower().split())
        
        # Calculate keyword overlap
        overlap = len(query_keywords.intersection(response_keywords))
        keyword_score = min(overlap / max(len(query_keywords), 1), 1.0)
        
        # Length score (prefer medium-length responses)
        length_score = min(len(response) / 200, 1.0)
        
        # Combine scores
        quality_score = (keyword_score * 0.7) + (length_score * 0.3)
        
        return min(quality_score, 1.0) 