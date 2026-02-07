# Simple Auto-Sync Setup (No Admin Required)
# This creates a startup script that runs auto-sync when you log in

$scriptPath = Join-Path $PSScriptRoot "auto_sync.bat"
$startupFolder = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupFolder "StressDetection-AutoSync.lnk"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Auto-Sync (Simple Mode)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Choose your preferred method:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Add to Windows Startup (runs once at login)" -ForegroundColor White
Write-Host "2. Manual sync only (you run auto_sync.bat when needed)" -ForegroundColor White
Write-Host "3. Create desktop shortcut for easy access" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        # Create shortcut in startup folder
        $WScriptShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WScriptShell.CreateShortcut($shortcutPath)
        $Shortcut.TargetPath = $scriptPath
        $Shortcut.WorkingDirectory = $PSScriptRoot
        $Shortcut.WindowStyle = 7  # Minimized
        $Shortcut.Save()
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Success! Auto-sync will run at startup." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "To remove, delete: $shortcutPath" -ForegroundColor Yellow
    }
    "2" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Manual mode selected." -ForegroundColor Green
        Write-Host "Run 'auto_sync.bat' whenever you want to sync." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    }
    "3" {
        $desktopPath = [Environment]::GetFolderPath('Desktop')
        $desktopShortcut = Join-Path $desktopPath "Sync to GitHub.lnk"
        
        $WScriptShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WScriptShell.CreateShortcut($desktopShortcut)
        $Shortcut.TargetPath = $scriptPath
        $Shortcut.WorkingDirectory = $PSScriptRoot
        $Shortcut.Description = "Sync Stress Detection project to GitHub"
        $Shortcut.Save()
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Desktop shortcut created!" -ForegroundColor Green
        Write-Host "Double-click 'Sync to GitHub' on your desktop anytime." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "TIP: You can also add auto_sync.bat to your run.bat file" -ForegroundColor Cyan
Write-Host "to sync after training completes." -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
