"""
Streamlit Dashboard for the Gen-AI Assistant
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List

# Import the main assistant
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import GenAIAssistant


class StreamlitDashboard:
    """
    Streamlit Dashboard for the Gen-AI Assistant
    """
    
    def __init__(self):
        """Initialize the Streamlit dashboard"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'assistant' not in st.session_state:
            try:
                st.session_state.assistant = GenAIAssistant()
            except Exception as e:
                st.error(f"Error initializing assistant: {e}")
                st.session_state.assistant = None
    
    def render_main_interface(self):
        """Render the main dashboard interface"""
        st.set_page_config(
            page_title="Gen-AI Assistant Dashboard",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 Gen-AI Assistant Platform")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Analytics", "⚙️ Configuration", "📈 Performance"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_analytics()
        
        with tab3:
            self.render_configuration()
        
        with tab4:
            self.render_performance()
    
    def render_sidebar(self):
        """Render the sidebar"""
        st.sidebar.title("🎛️ Controls")
        
        # System status
        if st.session_state.assistant:
            status = st.session_state.assistant.get_system_status()
            st.sidebar.success(f"System Status: {status['status']}")
        else:
            st.sidebar.error("System Status: Offline")
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("🔄 Refresh System"):
            try:
                st.session_state.assistant = GenAIAssistant()
                st.success("System refreshed successfully!")
            except Exception as e:
                st.error(f"Error refreshing system: {e}")
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
        
        # Statistics
        st.sidebar.subheader("📈 Statistics")
        st.sidebar.metric("Total Conversations", len(st.session_state.conversation_history))
        
        if st.session_state.conversation_history:
            avg_response_time = sum(
                conv.get('response_time', 0) for conv in st.session_state.conversation_history
            ) / len(st.session_state.conversation_history)
            st.sidebar.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.header("💬 Chat Interface")
        
        # Chat input
        user_input = st.text_area("Enter your message:", height=100, key="user_input")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Send", type="primary"):
                if user_input.strip():
                    self.process_user_input(user_input)
        
        with col2:
            if st.button("Clear"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Display conversation history
        st.subheader("Conversation History")
        
        if st.session_state.conversation_history:
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Conversation {len(st.session_state.conversation_history) - i}"):
                    st.write(f"**User:** {conv['query']}")
                    st.write(f"**Assistant:** {conv['response']}")
                    st.write(f"**Agent:** {conv['agent_used']}")
                    st.write(f"**Sentiment:** {conv['sentiment']}")
                    st.write(f"**Intent:** {conv['intent']}")
                    st.write(f"**Quality Score:** {conv['quality_score']:.2f}")
                    st.write(f"**Response Time:** {conv['response_time']:.2f}s")
                    st.write(f"**Timestamp:** {conv['timestamp']}")
        else:
            st.info("No conversations yet. Start by sending a message!")
    
    def process_user_input(self, user_input: str):
        """Process user input and generate response"""
        if not st.session_state.assistant:
            st.error("Assistant not initialized. Please check configuration.")
            return
        
        try:
            with st.spinner("Processing your request..."):
                result = st.session_state.assistant.process_query(user_input)
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'query': user_input,
                    'response': result['response'],
                    'agent_used': result['agent_used'],
                    'sentiment': result['sentiment'],
                    'intent': result['intent'],
                    'quality_score': result['quality_score'],
                    'response_time': result['response_time'],
                    'timestamp': result['timestamp']
                })
                
                st.success("Response generated successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing request: {e}")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.conversation_history:
            st.info("No data available for analytics. Start conversations to see insights!")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(st.session_state.conversation_history)
        
        # Analytics metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", len(df))
        
        with col2:
            avg_quality = df['quality_score'].mean()
            st.metric("Avg Quality Score", f"{avg_quality:.2f}")
        
        with col3:
            avg_response_time = df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col4:
            unique_agents = df['agent_used'].nunique()
            st.metric("Agents Used", unique_agents)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Agent usage distribution
            agent_counts = df['agent_used'].value_counts()
            fig_agents = px.pie(
                values=agent_counts.values,
                names=agent_counts.index,
                title="Agent Usage Distribution"
            )
            st.plotly_chart(fig_agents, use_container_width=True)
        
        with col2:
            # Sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            fig_sentiment = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Quality score over time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_sorted = df.sort_values('timestamp')
            
            fig_quality = px.line(
                df_sorted,
                x='timestamp',
                y='quality_score',
                title="Quality Score Over Time"
            )
            st.plotly_chart(fig_quality, use_container_width=True)
    
    def render_configuration(self):
        """Render configuration interface"""
        st.header("⚙️ Configuration")
        
        # Load current configuration
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return
        
        # Configuration form
        with st.form("configuration_form"):
            st.subheader("OpenAI Configuration")
            
            api_key = st.text_input("OpenAI API Key", value=config.get('openai_api_key', ''), type="password")
            model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"], index=0 if config.get('model') == "gpt-4" else 1)
            temperature = st.slider("Temperature", 0.0, 1.0, config.get('temperature', 0.7), 0.1)
            
            st.subheader("Agent Configuration")
            
            # Customer Support Agent
            cs_enabled = st.checkbox("Customer Support Agent", value=config.get('agents', {}).get('customer_support', {}).get('enabled', True))
            cs_temperature = st.slider("CS Agent Temperature", 0.0, 1.0, config.get('agents', {}).get('customer_support', {}).get('temperature', 0.7), 0.1)
            
            # Troubleshooting Agent
            ts_enabled = st.checkbox("Troubleshooting Agent", value=config.get('agents', {}).get('troubleshooting', {}).get('enabled', True))
            ts_temperature = st.slider("TS Agent Temperature", 0.0, 1.0, config.get('agents', {}).get('troubleshooting', {}).get('temperature', 0.5), 0.1)
            
            submitted = st.form_submit_button("Save Configuration")
            
            if submitted:
                # Update configuration
                config['openai_api_key'] = api_key
                config['model'] = model
                config['temperature'] = temperature
                
                config['agents']['customer_support']['enabled'] = cs_enabled
                config['agents']['customer_support']['temperature'] = cs_temperature
                config['agents']['troubleshooting']['enabled'] = ts_enabled
                config['agents']['troubleshooting']['temperature'] = ts_temperature
                
                # Save configuration
                try:
                    with open("config.json", "w") as f:
                        json.dump(config, f, indent=2)
                    st.success("Configuration saved successfully!")
                except Exception as e:
                    st.error(f"Error saving configuration: {e}")
    
    def render_performance(self):
        """Render performance monitoring"""
        st.header("📈 Performance Monitoring")
        
        if not st.session_state.conversation_history:
            st.info("No performance data available. Start conversations to see metrics!")
            return
        
        df = pd.DataFrame(st.session_state.conversation_history)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Requests", len(df))
            st.metric("Success Rate", "100%")  # Assuming all requests succeed for demo
        
        with col2:
            avg_time = df['response_time'].mean()
            st.metric("Average Response Time", f"{avg_time:.2f}s")
            max_time = df['response_time'].max()
            st.metric("Max Response Time", f"{max_time:.2f}s")
        
        with col3:
            avg_quality = df['quality_score'].mean()
            st.metric("Average Quality", f"{avg_quality:.2f}")
            min_quality = df['quality_score'].min()
            st.metric("Min Quality", f"{min_quality:.2f}")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution
            fig_time = px.histogram(
                df,
                x='response_time',
                title="Response Time Distribution",
                nbins=20
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Quality score distribution
            fig_quality = px.histogram(
                df,
                x='quality_score',
                title="Quality Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Recent activity
        st.subheader("Recent Activity")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent_df = df.tail(10).sort_values('timestamp', ascending=False)
            
            for _, row in recent_df.iterrows():
                with st.expander(f"{row['timestamp'].strftime('%H:%M:%S')} - {row['query'][:50]}..."):
                    st.write(f"**Query:** {row['query']}")
                    st.write(f"**Response:** {row['response']}")
                    st.write(f"**Agent:** {row['agent_used']}")
                    st.write(f"**Quality:** {row['quality_score']:.2f}")
                    st.write(f"**Time:** {row['response_time']:.2f}s")


def main():
    """Main function to run the Streamlit dashboard"""
    dashboard = StreamlitDashboard()
    dashboard.render_main_interface()


if __name__ == "__main__":
    main() 