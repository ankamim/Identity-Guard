@echo off
REM Xekinaei to DeepSafe (Docker) kai meta tin diki mas efarmogi.
REM An allakses ta paths sou, allakse tis diadromes edw katw.

echo ============================================
echo   Ekkinisi DeepSafe containers...
echo ============================================
cd /d C:\Users\kamil\Desktop\DeepSafe
docker compose up -d api npr_deepfakedetection

echo.
echo Perimeno 5 defterolepta gia na stathertopoiithoun ta containers...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   Ekkinisi tis efarmogis (Identity Guard)...
echo ============================================
cd /d C:\Users\kamil\Desktop\ptyxiakh

REM Anoigei ton browser meta apo 4 defterolepta, oso o serverakoma xekinaei
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8080"

call venv311\Scripts\activate
uvicorn app.main:app --reload --port 8080

REM Otan kleiseis ton server (Ctrl+C), afto to mynima emfanizetai:
echo.
echo O server stamatise. Ta Docker containers sinexizoun na trexoun sto parasenio.
echo An theleis na ta stamatiseis kai afta, grapse: docker compose down
echo (mesa ston fakelo DeepSafe)
pause
