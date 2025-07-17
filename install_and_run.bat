@echo off
echo 🚀 웨이퍼 매핑 도구 설치 및 실행

REM Python 설치 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    echo https://www.python.org/downloads/ 에서 Python을 설치하세요.
    pause
    exit /b
)

REM 필요한 패키지 설치
echo 📦 필요한 패키지 설치 중...
pip install streamlit numpy matplotlib scipy pandas plotly

REM 앱 실행
echo 🚀 웨이퍼 매핑 도구 실행 중...
streamlit run wafer_mapping_app.py --server.port 8501

pause