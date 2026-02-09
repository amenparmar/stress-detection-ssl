@echo off
setlocal enabledelayedexpansion

REM Save the script directory
set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe
set PARENT_DIR=%SCRIPT_DIR%..

REM Force PyTorch to use only NVIDIA GPU (hide Intel integrated graphics)
set CUDA_VISIBLE_DEVICES=1


echo SELECT MODE:
echo 1. Test Run (Mock Data)
echo 2. Pre-train (Real WESAD Data - 500 epochs)
echo 3. Calculate Model Accuracy (Evaluate - Standard Encoder)
echo 4. Train Ensemble (5 models for better accuracy)
echo 5. Train Multi-Modal Fusion (Separate encoders per modality)
echo 6. FULL PIPELINE - All Improvements Combined
echo 7. SMOTE Oversampling (Fix Class Imbalance)
echo 8. Leave-One-Subject-Out CV (Gold Standard Evaluation)
echo 9. Domain Adversarial Training (DANN - Subject-Invariant)
echo 10. Latent Trajectory Analysis (Continuous Monitoring)
echo 11. Subject-Invariant Loss Training (MMD + CORAL + Contrastive)
echo 12. COMBINED ADVANCED - MAXIMUM PERFORMANCE (DANN + Multi-Modal)
echo 13. 🏆 ULTIMATE PERFORMANCE - ALL TECHNIQUES + ENSEMBLE (85-88%% Expected)
echo 14. 📊 BENCHMARK ALL MODELS - Run and rank all configurations
echo 15. 🚀 ADVANCED BENCHMARK - Test SMOTE, DANN, Invariant, Ultimate
echo.
set /p choice="Enter choice (1-15): "

if "%choice%"=="1" goto option1
if "%choice%"=="2" goto option2
if "%choice%"=="3" goto option3
if "%choice%"=="4" goto option4
if "%choice%"=="5" goto option5
if "%choice%"=="6" goto option6
if "%choice%"=="7" goto option7
if "%choice%"=="8" goto option8
if "%choice%"=="9" goto option9
if "%choice%"=="10" goto option10
if "%choice%"=="11" goto option11
if "%choice%"=="12" goto option12
if "%choice%"=="13" goto option13
if "%choice%"=="14" goto option14
if "%choice%"=="15" goto option15
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

:option9
echo.
echo ========================================
echo   DOMAIN ADVERSARIAL TRAINING (DANN)
echo ========================================
echo Subject-invariant feature learning
echo Expected improvement: 74%% -^> 78-82%% LOSO accuracy
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode dann --epochs 100 --batch_size 32
goto end

:option10
echo.
echo ========================================
echo   LATENT TRAJECTORY ANALYSIS
echo ========================================
echo Continuous stress monitoring
echo Personalized baselines per subject
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode trajectory --epochs 100 --batch_size 32
goto end

:option11
echo.
echo ========================================
echo   SUBJECT-INVARIANT LOSS TRAINING
echo ========================================
echo Using MMD + CORAL + Contrastive losses
echo Expected improvement: 3-7%% accuracy gain
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode invariant --epochs 100 --batch_size 32
goto end

:option12
echo.
echo ========================================
echo   COMBINED ADVANCED - MAXIMUM PERFORMANCE
echo ========================================
echo Combines:
echo - Domain Adversarial Training
echo - Multi-Modal Fusion
echo - Subject-Invariant Losses
echo Expected: 82-86%% LOSO accuracy
echo Estimated time: 2-3 hours on GPU
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode combined --epochs 100 --batch_size 32
goto end

:option13
echo.
echo ========================================
echo   🏆 ULTIMATE PERFORMANCE PIPELINE 🏆
echo ========================================
echo.
echo This is the MOST POWERFUL configuration:
echo.
echo Stage 1: SSL Pre-training (500 epochs)
echo Stage 2: Ensemble of 5 Ultimate Models
echo          - Multi-Modal Fusion
echo          - Domain Adversarial (DANN)
echo          - Subject-Invariant Losses
echo          - Trajectory Analysis  
echo          - Temporal Consistency
echo Stage 3: Ensemble Evaluation
echo.
echo Expected: 85-88%% LOSO accuracy
echo Current Baseline: 74.35%%
echo Improvement: +11-14%% absolute gain!
echo.
echo Estimated time: 6-8 hours on RTX 5070 Ti GPU
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode ultimate --epochs 100 --batch_size 32
goto end

:option14
echo.
echo ========================================
echo   📊 BENCHMARK ALL MODELS
echo ========================================
echo.
echo This will run and rank ALL configurations:
echo   1. Baseline (SSL + Classifier)
echo   2. Multi-Modal Fusion
echo   3. Multi-Modal Ensemble (5 models)
echo   4. SMOTE Oversampling
echo   5. DANN (Domain Adversarial)
echo   6. Trajectory Analysis
echo   7. Subject-Invariant Losses
echo   8. Combined (DANN + Multi-Modal)
echo   9. Ultimate (All Techniques)
echo.
echo Estimated Time: 15-20 hours (full) or 3-4 hours (quick mode)
echo ========================================
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode benchmark --batch_size 32
goto end

:option15
echo.
echo ========================================
echo   🚀 ADVANCED BENCHMARK
echo ========================================
echo.
echo This will run and rank advanced techniques:
echo   1. SMOTE Oversampling
echo   2. DANN (Domain Adversarial)
echo   3. Subject-Invariant Loss
echo   4. Ultimate Performance
echo.
echo Estimated Time: 20-25 hours (full) or 4-6 hours (quick mode)
echo ========================================
echo.
pause
cd /d %PARENT_DIR%
"%VENV_PYTHON%" -m stress_detection.main --mode advanced_benchmark --batch_size 32
goto end

:invalid
echo Invalid choice. Please run again and select 1-15.
goto end

:end
cd /d %SCRIPT_DIR%
pause
