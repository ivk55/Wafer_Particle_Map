@echo off
echo 🚀 웨이퍼 매핑 도구 Windows 실행 파일 생성 중...

REM 필요한 패키지 설치
echo 📦 필요한 패키지 설치 중...
pip install -r requirements.txt

REM 실행 파일 생성
echo 🔨 실행 파일 빌드 중...
pyinstaller --onefile --console --name="웨이퍼매핑도구" launcher.py

REM 사용법 파일 생성
echo 📖 사용법 파일 생성 중...
echo 웨이퍼 매핑 도구 사용법 > dist\사용법.txt
echo. >> dist\사용법.txt
echo 1. 웨이퍼매핑도구.exe 를 더블클릭하세요 >> dist\사용법.txt
echo 2. 자동으로 브라우저가 열립니다 >> dist\사용법.txt
echo 3. 웨이퍼 매핑 도구를 사용하세요 >> dist\사용법.txt
echo. >> dist\사용법.txt
echo 문제가 있으면 http://localhost:8501 로 접속하세요 >> dist\사용법.txt

echo ✅ 완료! dist 폴더의 파일들을 배포하세요.
pause