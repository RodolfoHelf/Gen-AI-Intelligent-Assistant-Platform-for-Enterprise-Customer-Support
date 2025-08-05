"""
Escalation Agent for handling complex cases requiring human intervention
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..prompts.templates import ESCALATION_TEMPLATE


class EscalationAgent:
    """
    Escalation Agent for handling complex cases requiring human intervention
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.3):
        """
        Initialize the Escalation Agent
        
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
    
    def prepare_escalation(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare an escalation response and summary for human agents
        
        Args:
            query: Customer's original query
            context: Additional context about the interaction
            
        Returns:
            Escalation response for the customer
        """
        try:
            # Analyze the issue for escalation
            escalation_summary = self._create_escalation_summary(query, context)
            
            # Prepare context for LLM
            escalation_context = {
                "query": query,
                "summary": escalation_summary,
                "timestamp": datetime.now().isoformat()
            }
            
            if context:
                escalation_context.update(context)
            
            # Generate escalation response
            messages = [
                SystemMessage(content=ESCALATION_TEMPLATE),
                HumanMessage(content=self._format_escalation_query(query, escalation_context))
            ]
            
            response = self.llm(messages)
            
            self.logger.info(f"Prepared escalation for query: {query[:50]}...")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error preparing escalation: {e}")
            return "I understand this is a complex issue. I'm transferring you to our specialized support team who will be able to assist you better. Please hold while I connect you."
    
    def _create_escalation_summary(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a summary of the issue for human agents"""
        summary = {
            "customer_issue": query,
            "escalation_reason": self._determine_escalation_reason(query),
            "priority_level": self._determine_priority_level(query),
            "recommended_actions": self._get_recommended_actions(query),
            "customer_context": context or {},
            "escalation_timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def _determine_escalation_reason(self, query: str) -> str:
        """Determine the reason for escalation"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["hardware", "broken", "damaged", "physical"]):
            return "Hardware issue requiring physical inspection"
        elif any(word in query_lower for word in ["billing", "payment", "refund", "charge"]):
            return "Billing or payment issue requiring account access"
        elif any(word in query_lower for word in ["security", "breach", "hack", "unauthorized"]):
            return "Security concern requiring immediate attention"
        elif any(word in query_lower for word in ["legal", "compliance", "regulatory"]):
            return "Legal or compliance issue"
        elif any(word in query_lower for word in ["complex", "advanced", "specialist", "expert"]):
            return "Complex technical issue requiring specialist knowledge"
        else:
            return "Multiple troubleshooting steps attempted without resolution"
    
    def _determine_priority_level(self, query: str) -> str:
        """Determine the priority level of the escalation"""
        query_lower = query.lower()
        
        # Critical keywords
        critical_keywords = ["security breach", "data loss", "system down", "emergency", "urgent"]
        if any(keyword in query_lower for keyword in critical_keywords):
            return "critical"
        
        # High priority keywords
        high_keywords = ["not working", "broken", "error", "crash", "freeze", "hardware"]
        if any(keyword in query_lower for keyword in high_keywords):
            return "high"
        
        # Medium priority keywords
        medium_keywords = ["slow", "performance", "issue", "problem", "billing"]
        if any(keyword in query_lower for keyword in medium_keywords):
            return "medium"
        
        return "low"
    
    def _get_recommended_actions(self, query: str) -> List[str]:
        """Get recommended actions for human agents"""
        query_lower = query.lower()
        actions = []
        
        if any(word in query_lower for word in ["hardware", "broken", "damaged"]):
            actions.extend([
                "Verify warranty status",
                "Check repair options",
                "Assess replacement costs",
                "Schedule technician visit if applicable"
            ])
        
        if any(word in query_lower for word in ["billing", "payment", "refund"]):
            actions.extend([
                "Review account history",
                "Check payment methods",
                "Process refund if applicable",
                "Update billing information"
            ])
        
        if any(word in query_lower for word in ["security", "breach", "hack"]):
            actions.extend([
                "Immediately assess security impact",
                "Check for unauthorized access",
                "Implement security measures",
                "Notify relevant authorities if necessary"
            ])
        
        if not actions:
            actions = [
                "Review customer interaction history",
                "Assess technical complexity",
                "Determine appropriate specialist",
                "Schedule follow-up if needed"
            ]
        
        return actions
    
    def _format_escalation_query(self, query: str, context: Dict[str, Any]) -> str:
        """Format the escalation query for the LLM"""
        summary = context.get("summary", {})
        
        formatted_context = f"""
Customer Query: {query}

Escalation Summary:
- Reason: {summary.get('escalation_reason', 'Unknown')}
- Priority: {summary.get('priority_level', 'medium')}
- Recommended Actions: {', '.join(summary.get('recommended_actions', []))}

Customer Context: {summary.get('customer_context', {})}
"""
        
        return formatted_context
    
    def create_escalation_ticket(self, query: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an escalation ticket for the support team
        
        Args:
            query: Customer's original query
            summary: Escalation summary
            
        Returns:
            Escalation ticket information
        """
        ticket = {
            "ticket_id": f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "customer_query": query,
            "escalation_reason": summary.get("escalation_reason"),
            "priority_level": summary.get("priority_level"),
            "recommended_actions": summary.get("recommended_actions", []),
            "customer_context": summary.get("customer_context", {}),
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "assigned_to": None,
            "estimated_resolution_time": self._estimate_resolution_time(summary.get("priority_level"))
        }
        
        return ticket
    
    def _estimate_resolution_time(self, priority_level: str) -> str:
        """Estimate resolution time based on priority level"""
        time_estimates = {
            "critical": "2-4 hours",
            "high": "4-8 hours",
            "medium": "24-48 hours",
            "low": "3-5 business days"
        }
        
        return time_estimates.get(priority_level, "24-48 hours")
    
    def get_escalation_metrics(self) -> Dict[str, Any]:
        """Get escalation metrics for monitoring"""
        # This would typically pull from a database
        return {
            "total_escalations_today": 0,
            "average_resolution_time": "4.5 hours",
            "escalation_rate": "5.2%",
            "priority_distribution": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            }
        } 