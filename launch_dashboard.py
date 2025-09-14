#!/usr/bin/env python3
"""
Launch script for the Agent-Based Marketing Simulation Dashboard
"""

import sys
import os
import subprocess

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Agent-Based Marketing Simulation Dashboard...")
    print("=" * 60)
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        # Test imports first
        from src.visualization.dashboard import MarketingDashboard
        from src.environment.marketing_simulation import MarketingSimulation
        print("✅ All imports successful")
        
        # Launch Streamlit dashboard
        dashboard_path = os.path.join(current_dir, 'src', 'visualization', 'dashboard.py')
        print(f"📊 Starting dashboard at: {dashboard_path}")
        print("💡 Dashboard will open in your web browser")
        print("⚠️  Press Ctrl+C to stop the dashboard")
        print("-" * 60)
        
        # Run streamlit
        subprocess.run(['streamlit', 'run', dashboard_path, '--server.port=8501'])
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   uv pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    launch_dashboard()