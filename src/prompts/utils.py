"""
Prompt utilities for the Gen-AI Assistant
"""

import re
import logging
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher


class PromptUtils:
    """
    Utility class for managing and optimizing prompts
    """
    
    def __init__(self):
        """Initialize the prompt utilities"""
        self.logger = logging.getLogger(__name__)
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Remove common stop words and extract meaningful words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Clean text and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_relevant_knowledge(self, query: str, knowledge_base: List[Dict[str, Any]], 
                               threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find relevant knowledge base entries for a query
        
        Args:
            query: Customer query
            knowledge_base: List of knowledge base entries
            threshold: Similarity threshold
            
        Returns:
            List of relevant knowledge entries
        """
        relevant_entries = []
        query_keywords = set(self.extract_keywords(query))
        
        for entry in knowledge_base:
            # Extract text to compare (could be question, answer, title, etc.)
            entry_text = ""
            if 'question' in entry:
                entry_text += entry['question'] + " "
            if 'answer' in entry:
                entry_text += entry['answer'] + " "
            if 'title' in entry:
                entry_text += entry['title'] + " "
            if 'description' in entry:
                entry_text += entry['description'] + " "
            
            # Calculate similarity
            similarity = self.calculate_similarity(query, entry_text)
            
            # Check keyword overlap
            entry_keywords = set(self.extract_keywords(entry_text))
            keyword_overlap = len(query_keywords.intersection(entry_keywords))
            
            # If similarity or keyword overlap is above threshold, include entry
            if similarity > threshold or keyword_overlap > 0:
                entry['relevance_score'] = similarity
                entry['keyword_overlap'] = keyword_overlap
                relevant_entries.append(entry)
        
        # Sort by relevance score
        relevant_entries.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_entries[:5]  # Return top 5 most relevant
    
    def format_knowledge_for_prompt(self, knowledge_entries: List[Dict[str, Any]]) -> str:
        """
        Format knowledge base entries for inclusion in prompts
        
        Args:
            knowledge_entries: List of knowledge base entries
            
        Returns:
            Formatted knowledge string
        """
        if not knowledge_entries:
            return ""
        
        formatted_knowledge = "Relevant Information:\n\n"
        
        for i, entry in enumerate(knowledge_entries, 1):
            if 'question' in entry and 'answer' in entry:
                formatted_knowledge += f"{i}. Q: {entry['question']}\n   A: {entry['answer']}\n\n"
            elif 'title' in entry and 'description' in entry:
                formatted_knowledge += f"{i}. {entry['title']}: {entry['description']}\n\n"
            elif 'name' in entry and 'description' in entry:
                formatted_knowledge += f"{i}. {entry['name']}: {entry['description']}\n\n"
        
        return formatted_knowledge.strip()
    
    def optimize_prompt_length(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Optimize prompt length to fit within token limits
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum token limit
            
        Returns:
            Optimized prompt
        """
        # Simple token estimation (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # If prompt is too long, truncate while preserving important parts
        lines = prompt.split('\n')
        important_lines = []
        
        # Keep system instructions and important context
        for line in lines:
            if any(keyword in line.lower() for keyword in ['guidelines', 'context', 'you are']):
                important_lines.append(line)
        
        # Add truncated content
        remaining_space = max_tokens - (len(important_lines) * 20)  # Rough estimate
        if remaining_space > 0:
            # Add some content back, prioritizing the beginning
            content_lines = [line for line in lines if line not in important_lines]
            for line in content_lines[:remaining_space//10]:  # Rough estimate
                important_lines.append(line)
        
        return '\n'.join(important_lines)
    
    def add_context_to_prompt(self, base_prompt: str, context: Dict[str, Any]) -> str:
        """
        Add context information to a base prompt
        
        Args:
            base_prompt: Base prompt template
            context: Context information to add
            
        Returns:
            Prompt with context added
        """
        context_section = "Additional Context:\n"
        
        for key, value in context.items():
            if isinstance(value, str):
                context_section += f"- {key}: {value}\n"
            elif isinstance(value, list):
                context_section += f"- {key}: {', '.join(map(str, value))}\n"
            else:
                context_section += f"- {key}: {value}\n"
        
        # Insert context after the main prompt
        if "Context:" in base_prompt:
            # Replace existing context section
            prompt_parts = base_prompt.split("Context:")
            if len(prompt_parts) > 1:
                remaining = prompt_parts[1].split("\n", 1)
                if len(remaining) > 1:
                    return prompt_parts[0] + "Context:" + context_section + remaining[1]
        
        # Add context at the end if no existing context section
        return base_prompt + "\n\n" + context_section
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt for common issues
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check for common issues
        if len(prompt) < 50:
            issues.append("Prompt is too short")
        
        if len(prompt) > 10000:
            warnings.append("Prompt is very long, may hit token limits")
        
        if not any(keyword in prompt.lower() for keyword in ['guidelines', 'instructions', 'context']):
            warnings.append("Prompt lacks clear guidelines or context")
        
        if prompt.count('{') != prompt.count('}'):
            issues.append("Unmatched curly braces in prompt")
        
        # Check for potential issues
        if 'password' in prompt.lower() or 'api_key' in prompt.lower():
            warnings.append("Prompt may contain sensitive information")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'length': len(prompt),
            'estimated_tokens': len(prompt.split()) * 1.3
        }
    
    def create_dynamic_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Create a dynamic prompt by substituting variables
        
        Args:
            template: Prompt template with placeholders
            variables: Variables to substitute
            
        Returns:
            Prompt with variables substituted
        """
        prompt = template
        
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                if isinstance(value, str):
                    prompt = prompt.replace(placeholder, value)
                else:
                    prompt = prompt.replace(placeholder, str(value))
        
        return prompt
    
    def extract_prompt_variables(self, prompt: str) -> List[str]:
        """
        Extract variable placeholders from a prompt template
        
        Args:
            prompt: Prompt template
            
        Returns:
            List of variable names
        """
        variables = re.findall(r'\{(\w+)\}', prompt)
        return list(set(variables))  # Remove duplicates
    
    def sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize prompt to remove potentially sensitive information
        
        Args:
            prompt: Prompt to sanitize
            
        Returns:
            Sanitized prompt
        """
        # Remove common sensitive patterns
        sensitive_patterns = [
            r'api_key["\']?\s*[:=]\s*["\'][^"\']+["\']',
            r'password["\']?\s*[:=]\s*["\'][^"\']+["\']',
            r'token["\']?\s*[:=]\s*["\'][^"\']+["\']',
            r'secret["\']?\s*[:=]\s*["\'][^"\']+["\']'
        ]
        
        sanitized = prompt
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized 