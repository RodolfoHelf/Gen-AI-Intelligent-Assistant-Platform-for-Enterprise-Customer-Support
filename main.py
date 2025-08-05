#!/usr/bin/env python3
"""
Gen-AI Assistant Platform for Enterprise Customer Support
Main application entry point
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from src.ingestion.data_loader import DataLoader
from src.agents.customer_support_agent import CustomerSupportAgent
from src.agents.troubleshooting_agent import TroubleshootingAgent
from src.agents.escalation_agent import EscalationAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.ml_models.sentiment_analyzer import SentimentAnalyzer
from src.ml_models.intent_classifier import IntentClassifier
from src.ml_models.response_quality_assessor import ResponseQualityAssessor


class GenAIAssistant:
    """
    Main Gen-AI Assistant class that orchestrates all components
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Gen-AI Assistant with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_components()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {config_path}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('file', 'logs/assistant.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Gen-AI Assistant initialized")
    
    def _setup_components(self):
        """Initialize all components of the assistant"""
        try:
            # Initialize data loader
            self.data_loader = DataLoader()
            
            # Initialize agents
            self.customer_support_agent = CustomerSupportAgent(
                api_key=self.config.get('openai_api_key'),
                model=self.config['agents']['customer_support']['model'],
                temperature=self.config['agents']['customer_support']['temperature']
            )
            
            self.troubleshooting_agent = TroubleshootingAgent(
                api_key=self.config.get('openai_api_key'),
                model=self.config['agents']['troubleshooting']['model'],
                temperature=self.config['agents']['troubleshooting']['temperature']
            )
            
            self.escalation_agent = EscalationAgent(
                api_key=self.config.get('openai_api_key'),
                model=self.config['agents']['escalation']['model'],
                temperature=self.config['agents']['escalation']['temperature']
            )
            
            self.knowledge_agent = KnowledgeAgent(
                api_key=self.config.get('openai_api_key'),
                model=self.config['agents']['knowledge']['model'],
                temperature=self.config['agents']['knowledge']['temperature']
            )
            
            # Initialize ML models
            self.sentiment_analyzer = SentimentAnalyzer()
            self.intent_classifier = IntentClassifier()
            self.response_quality_assessor = ResponseQualityAssessor()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a customer query and return a response
        
        Args:
            query: Customer query text
            user_id: Optional user identifier
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        
        try:
            # Analyze query
            sentiment = self.sentiment_analyzer.analyze_sentiment(query)
            intent = self.intent_classifier.classify_intent(query)
            
            self.logger.info(f"Query analysis - Sentiment: {sentiment}, Intent: {intent}")
            
            # Route query to appropriate agent
            if intent in ['troubleshooting', 'technical_issue']:
                response = self.troubleshooting_agent.diagnose_issue(query)
                agent_used = 'troubleshooting'
            elif intent in ['escalation', 'complex_issue']:
                response = self.escalation_agent.prepare_escalation(query)
                agent_used = 'escalation'
            elif intent in ['product_info', 'pricing']:
                response = self.knowledge_agent.get_product_info(query)
                agent_used = 'knowledge'
            else:
                response = self.customer_support_agent.handle_query(query)
                agent_used = 'customer_support'
            
            # Assess response quality
            quality_score = self.response_quality_assessor.assess_quality(query, response)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'response': response,
                'agent_used': agent_used,
                'sentiment': sentiment,
                'intent': intent,
                'quality_score': quality_score,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id
            }
            
            self.logger.info(f"Query processed successfully in {response_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again later or contact our support team.",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        return {
            'status': 'operational',
            'components': {
                'customer_support_agent': 'active',
                'troubleshooting_agent': 'active',
                'escalation_agent': 'active',
                'knowledge_agent': 'active',
                'sentiment_analyzer': 'active',
                'intent_classifier': 'active',
                'response_quality_assessor': 'active'
            },
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main entry point for the application"""
    try:
        # Initialize the assistant
        assistant = GenAIAssistant()
        
        print("🤖 Gen-AI Assistant Platform")
        print("=" * 40)
        print("Type 'quit' to exit")
        print()
        
        # Interactive demo
        while True:
            query = input("Customer: ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not query.strip():
                continue
            
            # Process the query
            result = assistant.process_query(query)
            
            print(f"\nAssistant: {result['response']}")
            print(f"Agent: {result['agent_used']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Intent: {result['intent']}")
            print(f"Quality Score: {result['quality_score']:.2f}")
            print(f"Response Time: {result['response_time']:.2f}s")
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Application error: {e}")


if __name__ == "__main__":
    main() 