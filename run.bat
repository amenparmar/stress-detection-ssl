@echo off
setlocal enabledelayedexpansion

REM Save the script directory
set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe
set PARENT_DIR=%SCRIPT_DIR%..

echo SELECT MODE:
echo 1. Test Run (Mock Data)
echo 2. Pre-train (Real WESAD Data - 500 epochs)
echo 3. Calculate Model Accuracy (Evaluate - Standard Encoder)
echo 4. Train Ensemble (5 models for better accuracy)
echo 5. Train Multi-Modal Fusion (Separate encoders per modality)
echo 6. FULL PIPELINE - All Improvements Combined
echo 7. SMOTE Oversampling (Fix Class Imbalance)
echo 8. Leave-One-Subject-Out CV (Gold Standard Evaluation)
echo.
set /p choice="Enter choice (1-8): "

if "%choice%"=="1" goto option1
if "%choice%"=="2" goto option2
if "%choice%"=="3" goto option3
if "%choice%"=="4" goto option4
if "%choice%"=="5" goto option5
if "%choice%"=="6" goto option6
if "%choice%"=="7" goto option7
if "%choice%"=="8" goto option8
goto invalid

:option1
echo Running Test Run...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode test_run
goto end

:option2
echo Running Pre-training (500 epochs)...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode pretrain --epochs 500 --batch_size 32
goto end

:option3
echo Calculating Model Accuracy...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode evaluate --epochs 100 --batch_size 32
goto end

:option4
echo Training Ensemble (5 models)...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode ensemble --epochs 100 --batch_size 32
goto end

:option5
echo Training Multi-Modal Fusion...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode multimodal --epochs 100 --batch_size 32
goto end

:option6
echo.
echo ========================================
echo   FULL PIPELINE - Maximum Accuracy
echo ========================================
echo This will run:
echo 1. Pre-training with 500 epochs
echo 2. Multi-Modal Ensemble with 5 fusion models
echo Expected accuracy: 85-88 percent
echo Estimated time: 3-4 hours on CPU or 20-30 minutes on GPU
echo.
pause
echo.
echo [Step 1/2] Pre-training encoder with all optimizations...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode pretrain --epochs 500 --batch_size 32
echo.
echo [Step 2/2] Training multi-modal ensemble...
"%VENV_PYTHON%" -m stress_detection.main --mode multimodal_ensemble --epochs 100 --batch_size 32
echo.
echo ========================================
echo   FULL PIPELINE COMPLETE
echo ========================================
echo ========================================
goto end

:option7
echo Training with SMOTE Oversampling...
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode smote --epochs 100 --batch_size 32
goto end

:option8
echo.
echo ========================================
echo   LEAVE-ONE-SUBJECT-OUT CV
echo ========================================
echo This will train and test on EACH subject
echo Estimated time: 3-6 hours (15 subjects)
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode loso --epochs 100 --batch_size 32
goto end

:invalid
echo Invalid choice. Please run again and select 1-8.
goto end

:end
cd /d %SCRIPT_DIR%
pause
