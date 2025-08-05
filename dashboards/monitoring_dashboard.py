"""
Monitoring Dashboard for the Gen-AI Assistant
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List
import psutil
import os

# Import the main assistant
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import GenAIAssistant


class MonitoringDashboard:
    """
    Monitoring Dashboard for system health and performance
    """
    
    def __init__(self):
        """Initialize the monitoring dashboard"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if 'monitoring_data' not in st.session_state:
            st.session_state.monitoring_data = []
    
    def render_main_interface(self):
        """Render the main monitoring interface"""
        st.set_page_config(
            page_title="Gen-AI Assistant Monitoring",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 Gen-AI Assistant Monitoring Dashboard")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["🖥️ System Health", "📈 Performance", "🔍 Logs", "⚙️ Alerts"])
        
        with tab1:
            self.render_system_health()
        
        with tab2:
            self.render_performance()
        
        with tab3:
            self.render_logs()
        
        with tab4:
            self.render_alerts()
    
    def render_sidebar(self):
        """Render the sidebar"""
        st.sidebar.title("🎛️ Monitoring Controls")
        
        # System status
        system_status = self.get_system_status()
        if system_status['status'] == 'operational':
            st.sidebar.success("System Status: Operational")
        else:
            st.sidebar.error("System Status: Issues Detected")
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("🔄 Refresh Data"):
            self.collect_monitoring_data()
            st.success("Monitoring data refreshed!")
        
        if st.sidebar.button("📊 Export Data"):
            self.export_monitoring_data()
        
        # System metrics
        st.sidebar.subheader("System Metrics")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        st.sidebar.metric("CPU Usage", f"{cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        st.sidebar.metric("Memory Usage", f"{memory.percent}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        st.sidebar.metric("Disk Usage", f"{disk.percent}%")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            # Check if assistant is available
            assistant = GenAIAssistant()
            status = assistant.get_system_status()
            
            # Add system metrics
            status['cpu_usage'] = psutil.cpu_percent()
            status['memory_usage'] = psutil.virtual_memory().percent
            status['disk_usage'] = psutil.disk_usage('/').percent
            
            # Determine overall status
            if (status['cpu_usage'] > 90 or 
                status['memory_usage'] > 90 or 
                status['disk_usage'] > 90):
                status['status'] = 'warning'
            
            return status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0
            }
    
    def collect_monitoring_data(self):
        """Collect current monitoring data"""
        try:
            system_status = self.get_system_status()
            
            monitoring_data = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': system_status.get('cpu_usage', 0),
                'memory_usage': system_status.get('memory_usage', 0),
                'disk_usage': system_status.get('disk_usage', 0),
                'system_status': system_status.get('status', 'unknown'),
                'components': system_status.get('components', {})
            }
            
            st.session_state.monitoring_data.append(monitoring_data)
            
            # Keep only last 100 data points
            if len(st.session_state.monitoring_data) > 100:
                st.session_state.monitoring_data = st.session_state.monitoring_data[-100:]
                
        except Exception as e:
            st.error(f"Error collecting monitoring data: {e}")
    
    def render_system_health(self):
        """Render system health monitoring"""
        st.header("🖥️ System Health")
        
        # Collect current data
        self.collect_monitoring_data()
        
        if not st.session_state.monitoring_data:
            st.info("No monitoring data available. Click 'Refresh Data' to collect metrics.")
            return
        
        # Current metrics
        latest_data = st.session_state.monitoring_data[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_color = "red" if latest_data['cpu_usage'] > 80 else "orange" if latest_data['cpu_usage'] > 60 else "green"
            st.metric("CPU Usage", f"{latest_data['cpu_usage']}%", delta=None, delta_color="normal")
            st.markdown(f"<div style='background-color: {cpu_color}; height: 4px; border-radius: 2px;'></div>", unsafe_allow_html=True)
        
        with col2:
            memory_color = "red" if latest_data['memory_usage'] > 80 else "orange" if latest_data['memory_usage'] > 60 else "green"
            st.metric("Memory Usage", f"{latest_data['memory_usage']}%", delta=None, delta_color="normal")
            st.markdown(f"<div style='background-color: {memory_color}; height: 4px; border-radius: 2px;'></div>", unsafe_allow_html=True)
        
        with col3:
            disk_color = "red" if latest_data['disk_usage'] > 80 else "orange" if latest_data['disk_usage'] > 60 else "green"
            st.metric("Disk Usage", f"{latest_data['disk_usage']}%", delta=None, delta_color="normal")
            st.markdown(f"<div style='background-color: {disk_color}; height: 4px; border-radius: 2px;'></div>", unsafe_allow_html=True)
        
        with col4:
            status_color = "green" if latest_data['system_status'] == 'operational' else "red"
            st.metric("System Status", latest_data['system_status'].title(), delta=None, delta_color="normal")
            st.markdown(f"<div style='background-color: {status_color}; height: 4px; border-radius: 2px;'></div>", unsafe_allow_html=True)
        
        # System health charts
        df = pd.DataFrame(st.session_state.monitoring_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU usage over time
            fig_cpu = px.line(
                df,
                x='timestamp',
                y='cpu_usage',
                title="CPU Usage Over Time",
                color_discrete_sequence=['#FF6B6B']
            )
            fig_cpu.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Warning Threshold")
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            # Memory usage over time
            fig_memory = px.line(
                df,
                x='timestamp',
                y='memory_usage',
                title="Memory Usage Over Time",
                color_discrete_sequence=['#4ECDC4']
            )
            fig_memory.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Warning Threshold")
            st.plotly_chart(fig_memory, use_container_width=True)
        
        # Component status
        st.subheader("Component Status")
        
        if 'components' in latest_data:
            components = latest_data['components']
            
            col1, col2 = st.columns(2)
            
            with col1:
                for component, status in list(components.items())[:len(components)//2]:
                    status_color = "green" if status == 'active' else "red"
                    st.markdown(f"**{component.replace('_', ' ').title()}:** <span style='color: {status_color}'>{status}</span>", unsafe_allow_html=True)
            
            with col2:
                for component, status in list(components.items())[len(components)//2:]:
                    status_color = "green" if status == 'active' else "red"
                    st.markdown(f"**{component.replace('_', ' ').title()}:** <span style='color: {status_color}'>{status}</span>", unsafe_allow_html=True)
    
    def render_performance(self):
        """Render performance monitoring"""
        st.header("📈 Performance Monitoring")
        
        if not st.session_state.monitoring_data:
            st.info("No performance data available.")
            return
        
        df = pd.DataFrame(st.session_state.monitoring_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_cpu = df['cpu_usage'].mean()
            st.metric("Average CPU Usage", f"{avg_cpu:.1f}%")
            
            max_cpu = df['cpu_usage'].max()
            st.metric("Peak CPU Usage", f"{max_cpu:.1f}%")
        
        with col2:
            avg_memory = df['memory_usage'].mean()
            st.metric("Average Memory Usage", f"{avg_memory:.1f}%")
            
            max_memory = df['memory_usage'].max()
            st.metric("Peak Memory Usage", f"{max_memory:.1f}%")
        
        with col3:
            avg_disk = df['disk_usage'].mean()
            st.metric("Average Disk Usage", f"{avg_disk:.1f}%")
            
            max_disk = df['disk_usage'].max()
            st.metric("Peak Disk Usage", f"{max_disk:.1f}%")
        
        # Performance trends
        st.subheader("Performance Trends")
        
        # Create a combined chart
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cpu_usage'],
            mode='lines',
            name='CPU Usage',
            line=dict(color='#FF6B6B')
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['memory_usage'],
            mode='lines',
            name='Memory Usage',
            line=dict(color='#4ECDC4')
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['disk_usage'],
            mode='lines',
            name='Disk Usage',
            line=dict(color='#45B7D1')
        ))
        
        fig_combined.update_layout(
            title="System Resource Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Performance alerts
        st.subheader("Performance Alerts")
        
        alerts = []
        
        if df['cpu_usage'].iloc[-1] > 80:
            alerts.append("⚠️ High CPU usage detected")
        
        if df['memory_usage'].iloc[-1] > 80:
            alerts.append("⚠️ High memory usage detected")
        
        if df['disk_usage'].iloc[-1] > 80:
            alerts.append("⚠️ High disk usage detected")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ All systems operating normally")
    
    def render_logs(self):
        """Render log monitoring"""
        st.header("🔍 Log Monitoring")
        
        # Log file monitoring
        log_file = "logs/assistant.log"
        
        if os.path.exists(log_file):
            # Read recent logs
            try:
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                
                # Get last 50 lines
                recent_logs = logs[-50:] if len(logs) > 50 else logs
                
                st.subheader("Recent Log Entries")
                
                # Filter options
                log_level = st.selectbox("Filter by log level:", ["ALL", "INFO", "WARNING", "ERROR"])
                
                filtered_logs = []
                for log in recent_logs:
                    if log_level == "ALL" or log_level in log:
                        filtered_logs.append(log)
                
                # Display logs
                log_text = "".join(filtered_logs)
                st.text_area("Logs", log_text, height=400)
                
                # Log statistics
                st.subheader("Log Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Log Entries", len(logs))
                
                with col2:
                    error_count = sum(1 for log in logs if "ERROR" in log)
                    st.metric("Error Count", error_count)
                
                with col3:
                    warning_count = sum(1 for log in logs if "WARNING" in log)
                    st.metric("Warning Count", warning_count)
                
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.info("No log file found. Logs will appear here once the system starts generating them.")
    
    def render_alerts(self):
        """Render alert management"""
        st.header("⚙️ Alert Management")
        
        # Alert configuration
        st.subheader("Alert Configuration")
        
        with st.form("alert_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_threshold = st.slider("CPU Usage Threshold (%)", 50, 100, 80)
                memory_threshold = st.slider("Memory Usage Threshold (%)", 50, 100, 80)
            
            with col2:
                disk_threshold = st.slider("Disk Usage Threshold (%)", 50, 100, 80)
                alert_email = st.text_input("Alert Email", "admin@company.com")
            
            enable_alerts = st.checkbox("Enable Email Alerts", value=True)
            
            if st.form_submit_button("Save Alert Configuration"):
                # Save alert configuration
                alert_config = {
                    'cpu_threshold': cpu_threshold,
                    'memory_threshold': memory_threshold,
                    'disk_threshold': disk_threshold,
                    'alert_email': alert_email,
                    'enable_alerts': enable_alerts
                }
                
                try:
                    with open("alert_config.json", "w") as f:
                        json.dump(alert_config, f, indent=2)
                    st.success("Alert configuration saved!")
                except Exception as e:
                    st.error(f"Error saving alert configuration: {e}")
        
        # Current alerts
        st.subheader("Current Alerts")
        
        if not st.session_state.monitoring_data:
            st.info("No monitoring data available for alerts.")
            return
        
        latest_data = st.session_state.monitoring_data[-1]
        
        alerts = []
        
        if latest_data['cpu_usage'] > 80:
            alerts.append({
                'type': 'CPU Usage',
                'severity': 'High',
                'message': f"CPU usage is {latest_data['cpu_usage']}%",
                'timestamp': latest_data['timestamp']
            })
        
        if latest_data['memory_usage'] > 80:
            alerts.append({
                'type': 'Memory Usage',
                'severity': 'High',
                'message': f"Memory usage is {latest_data['memory_usage']}%",
                'timestamp': latest_data['timestamp']
            })
        
        if latest_data['disk_usage'] > 80:
            alerts.append({
                'type': 'Disk Usage',
                'severity': 'High',
                'message': f"Disk usage is {latest_data['disk_usage']}%",
                'timestamp': latest_data['timestamp']
            })
        
        if alerts:
            for alert in alerts:
                with st.expander(f"{alert['type']} - {alert['severity']}"):
                    st.write(f"**Message:** {alert['message']}")
                    st.write(f"**Timestamp:** {alert['timestamp']}")
                    st.write(f"**Severity:** {alert['severity']}")
        else:
            st.success("✅ No active alerts")
    
    def export_monitoring_data(self):
        """Export monitoring data"""
        if not st.session_state.monitoring_data:
            st.warning("No monitoring data to export.")
            return
        
        df = pd.DataFrame(st.session_state.monitoring_data)
        
        # Create CSV
        csv = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="📥 Download Monitoring Data (CSV)",
            data=csv,
            file_name=f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    """Main function to run the monitoring dashboard"""
    dashboard = MonitoringDashboard()
    dashboard.render_main_interface()


if __name__ == "__main__":
    main() 