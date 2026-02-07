@echo off
REM Auto-sync script for GitHub
REM This script commits and pushes all changes to GitHub

SET GIT_PATH="C:\Program Files\Git\cmd\git.exe"

echo ========================================
echo Auto-syncing to GitHub...
echo ========================================
echo.

REM Check if there are any changes
%GIT_PATH% status --short
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Adding all changes...
    %GIT_PATH% add .
    
    echo.
    echo Committing changes...
    %GIT_PATH% commit -m "Auto-sync: %date% %time%"
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Pushing to GitHub...
        %GIT_PATH% push
        
        if %ERRORLEVEL% EQU 0 (
            echo.
            echo ========================================
            echo Successfully synced to GitHub!
            echo ========================================
        ) else (
            echo.
            echo ========================================
            echo ERROR: Failed to push to GitHub
            echo Please check your internet connection and credentials
            echo ========================================
        )
    ) else (
        echo.
        echo No changes to commit
    )
) else (
    echo.
    echo ========================================
    echo ERROR: Git command failed
    echo ========================================
)

echo.
pause
