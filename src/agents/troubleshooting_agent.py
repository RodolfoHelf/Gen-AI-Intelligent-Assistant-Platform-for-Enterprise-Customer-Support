"""
Troubleshooting Agent for handling technical issues and diagnostics
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..prompts.templates import TROUBLESHOOTING_TEMPLATE


class TroubleshootingAgent:
    """
    Troubleshooting Agent for handling technical issues and diagnostics
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.5):
        """
        Initialize the Troubleshooting Agent
        
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
        
        # Load troubleshooting guides
        self.troubleshooting_guides = self._load_troubleshooting_guides()
        
    def _load_troubleshooting_guides(self) -> List[Dict[str, Any]]:
        """Load troubleshooting guides from knowledge base"""
        try:
            with open("data/knowledge_base/troubleshooting_guides.json", "r") as f:
                guides = json.load(f)
            return guides
        except Exception as e:
            self.logger.error(f"Error loading troubleshooting guides: {e}")
            return []
    
    def diagnose_issue(self, query: str, device_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Diagnose a technical issue and provide troubleshooting steps
        
        Args:
            query: Customer's issue description
            device_info: Optional device information
            
        Returns:
            Diagnostic response with troubleshooting steps
        """
        try:
            # Find relevant troubleshooting guides
            relevant_guides = self._find_relevant_guides(query)
            
            # Prepare context for LLM
            context = {
                "query": query,
                "relevant_guides": relevant_guides,
                "device_info": device_info or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate diagnostic response
            messages = [
                SystemMessage(content=TROUBLESHOOTING_TEMPLATE),
                HumanMessage(content=self._format_diagnostic_query(query, context))
            ]
            
            response = self.llm(messages)
            
            self.logger.info(f"Generated diagnostic for issue: {query[:50]}...")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error diagnosing issue: {e}")
            return "I apologize, but I'm having trouble diagnosing your issue. Please try again or contact our technical support team."
    
    def _find_relevant_guides(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant troubleshooting guides for the issue"""
        relevant_guides = []
        query_lower = query.lower()
        
        for guide in self.troubleshooting_guides:
            # Check title relevance
            title_lower = guide["title"].lower()
            if any(word in title_lower for word in query_lower.split()):
                relevant_guides.append(guide)
                continue
            
            # Check category relevance
            category_lower = guide["category"].lower()
            if any(word in category_lower for word in query_lower.split()):
                relevant_guides.append(guide)
                continue
            
            # Check common causes
            for cause in guide.get("common_causes", []):
                if any(word in cause.lower() for word in query_lower.split()):
                    relevant_guides.append(guide)
                    break
        
        # Sort by severity (high severity first)
        severity_order = {"high": 3, "medium": 2, "low": 1}
        relevant_guides.sort(key=lambda x: severity_order.get(x.get("severity", "medium"), 1), reverse=True)
        
        return relevant_guides[:3]  # Return top 3 most relevant
    
    def _format_diagnostic_query(self, query: str, context: Dict[str, Any]) -> str:
        """Format the diagnostic query for the LLM"""
        formatted_context = f"""
Customer Issue: {query}

Relevant Troubleshooting Guides:
"""
        
        for guide in context.get("relevant_guides", []):
            formatted_context += f"""
Title: {guide['title']}
Category: {guide['category']}
Severity: {guide['severity']}
Estimated Time: {guide['estimated_time']}
Common Causes: {', '.join(guide.get('common_causes', []))}
Steps: {', '.join(guide.get('steps', []))}
Precautions: {', '.join(guide.get('precautions', []))}
"""
        
        if context.get("device_info"):
            formatted_context += f"\nDevice Information: {context['device_info']}"
        
        return formatted_context
    
    def get_escalation_recommendation(self, query: str, attempted_steps: List[str]) -> Dict[str, Any]:
        """
        Determine if an issue should be escalated to human support
        
        Args:
            query: Original issue description
            attempted_steps: Steps that have been attempted
            
        Returns:
            Escalation recommendation
        """
        # Simple escalation logic based on keywords and attempted steps
        escalation_keywords = [
            "hardware", "broken", "damaged", "warranty", "replacement",
            "complex", "advanced", "technical", "specialist", "expert"
        ]
        
        query_lower = query.lower()
        has_escalation_keywords = any(keyword in query_lower for keyword in escalation_keywords)
        
        # Check if many steps have been attempted
        many_attempts = len(attempted_steps) >= 3
        
        should_escalate = has_escalation_keywords or many_attempts
        
        return {
            "should_escalate": should_escalate,
            "reason": "Hardware issue or complex technical problem" if has_escalation_keywords else "Multiple troubleshooting steps attempted",
            "priority": "high" if has_escalation_keywords else "medium",
            "recommended_action": "Escalate to technical specialist" if should_escalate else "Continue with automated troubleshooting"
        }
    
    def get_issue_severity(self, query: str) -> str:
        """
        Assess the severity of an issue
        
        Args:
            query: Issue description
            
        Returns:
            Severity level (low, medium, high, critical)
        """
        critical_keywords = ["data loss", "security breach", "system down", "emergency"]
        high_keywords = ["not working", "broken", "error", "crash", "freeze"]
        medium_keywords = ["slow", "performance", "issue", "problem"]
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in critical_keywords):
            return "critical"
        elif any(keyword in query_lower for keyword in high_keywords):
            return "high"
        elif any(keyword in query_lower for keyword in medium_keywords):
            return "medium"
        else:
            return "low" 