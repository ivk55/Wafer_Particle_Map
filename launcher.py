#!/usr/bin/env python3
"""
웨이퍼 매핑 도구 런처
실행 파일로 배포하기 위한 런처 스크립트
"""
import os
import sys
import subprocess
import webbrowser
import time
import threading
import signal
from pathlib import Path

def find_free_port():
    """사용 가능한 포트 찾기"""
    import socket
    for port in range(8501, 8520):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return 8501

def run_streamlit(port):
    """Streamlit 앱 실행"""
    script_dir = Path(__file__).parent
    app_file = script_dir / "wafer_mapping_app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_file),
        "--server.port", str(port),
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    print("🚀 웨이퍼 매핑 도구를 시작합니다...")
    
    # 사용 가능한 포트 찾기
    port = find_free_port()
    print(f"포트 {port}에서 실행 중...")
    
    # Streamlit 앱 실행
    process = run_streamlit(port)
    
    # 잠시 대기 후 브라우저 열기
    def open_browser():
        time.sleep(3)
        url = f"http://localhost:{port}"
        print(f"브라우저에서 {url} 을 여는 중...")
        webbrowser.open(url)
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        print("웨이퍼 매핑 도구가 실행 중입니다.")
        print("브라우저가 자동으로 열립니다.")
        print("종료하려면 Ctrl+C를 누르세요.")
        
        # 프로세스 대기
        process.wait()
        
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다...")
        process.terminate()
        process.wait()
    
    print("웨이퍼 매핑 도구가 종료되었습니다.")

if __name__ == "__main__":
    main()