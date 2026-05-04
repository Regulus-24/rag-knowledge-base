@echo off
chcp 65001 >nul
cd /d C:\Users\23050\Desktop\rag_project
echo ========================================
echo   RAG 知识库问答系统
echo   http://localhost:8501
echo   LAN: http://<你的IP>:8501
echo ========================================
echo.
C:\Users\23050\Desktop\Python\3.10.11\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
