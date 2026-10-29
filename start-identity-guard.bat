@echo off
REM Xekinaei to DeepSafe (Docker) kai tin efarmogi Identity Guard.
REM Douleuei apo opoiondhpote fakelo, arkei to bat na einai mesa sto repo.

REM Pame ston fakelo tou repo (ekei pou vrisketai afto to arxeio)
cd /d "%~dp0"

echo ============================================
echo   Ekkinisi DeepSafe containers...
echo ============================================
REM Ta varh tou retrained montelou mpainoun panta prin to start, giati an
REM to container ftiaxtei apo thn arxh xanontai kai to katwfli 0.24 den isxyei.
REM To docker cp douleuei kai se stamatimeno container.
docker cp models\fc_weights_faceswap.pth deepsafe-universalfakedetect:/app/model/universalfakedetect/pretrained_weights/fc_weights.pth
if errorlevel 1 (
    echo.
    echo To container deepsafe-universalfakedetect den vrethike.
    echo Einai anoixto to Docker Desktop? An nai, ftiakse to DeepSafe opws
    echo leei to README: docker compose up -d --build api universalfakedetect
    pause
    exit /b 1
)
docker start deepsafe-api deepsafe-universalfakedetect >nul
docker restart deepsafe-universalfakedetect >nul

echo.
echo Perimeno 5 defterolepta gia na stathertopoiithoun ta containers...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   Ekkinisi tis efarmogis (Identity Guard)...
echo ============================================
REM To venv mporei na legetai venv311 h venv (opws sto README)
if exist venv311\Scripts\activate.bat (
    call venv311\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Den vrethike venv mesa sto repo. Ftiakse to opws leei to README:
    echo   py -3.11 -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Anoigei ton browser meta apo 4 defterolepta, oso o server akoma xekinaei
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8080"

python -m uvicorn app.main:app --port 8080

REM Otan kleiseis ton server (Ctrl+C), afto to mynima emfanizetai:
echo.
echo O server stamatise. Ta Docker containers sinexizoun na trexoun sto parasenio.
echo An theleis na ta stamatiseis kai afta, grapse:
echo   docker stop deepsafe-api deepsafe-universalfakedetect
pause
