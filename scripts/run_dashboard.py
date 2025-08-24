"""
Run Streamlit Dashboard
Streamlit 대시보드 실행 스크립트
"""

import subprocess
import sys
import os

def main():
    """대시보드 실행"""
    print("\n" + "="*60)
    print("  KIMCHI PREMIUM TRADING DASHBOARD")
    print("="*60)
    print("\nStarting Streamlit dashboard...")
    print("Dashboard will open in your browser automatically")
    print("\nPress Ctrl+C to stop the dashboard")
    print("-"*60 + "\n")
    
    # Streamlit 실행
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend",
        "dashboard",
        "streamlit_dashboard.py"
    )
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print("\nMake sure Streamlit is installed:")
        print("pip install streamlit plotly")


if __name__ == "__main__":
    main()