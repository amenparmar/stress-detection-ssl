# Setup script to create a scheduled task for auto-syncing to GitHub
# Run this script with: powershell -ExecutionPolicy Bypass -File setup_auto_sync.ps1

$scriptPath = Join-Path $PSScriptRoot "auto_sync.bat"
$taskName = "StressDetection-AutoSync"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Auto-Sync to GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "Task already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Write-Host "Creating scheduled task: $taskName" -ForegroundColor Green
Write-Host ""
Write-Host "Choose auto-sync interval:" -ForegroundColor Cyan
Write-Host "1. Every 30 minutes" -ForegroundColor White
Write-Host "2. Every 1 hour" -ForegroundColor White
Write-Host "3. Every 2 hours" -ForegroundColor White
Write-Host "4. Every 6 hours" -ForegroundColor White
Write-Host "5. Daily at 11 PM" -ForegroundColor White
Write-Host "6. Manual only (no scheduled task)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-6)"

switch ($choice) {
    "1" { 
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration ([TimeSpan]::MaxValue)
        $intervalText = "every 30 minutes"
    }
    "2" { 
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration ([TimeSpan]::MaxValue)
        $intervalText = "every 1 hour"
    }
    "3" { 
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration ([TimeSpan]::MaxValue)
        $intervalText = "every 2 hours"
    }
    "4" { 
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration ([TimeSpan]::MaxValue)
        $intervalText = "every 6 hours"
    }
    "5" { 
        $trigger = New-ScheduledTaskTrigger -Daily -At 11PM
        $intervalText = "daily at 11 PM"
    }
    "6" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "No scheduled task created." -ForegroundColor Green
        Write-Host "You can manually run 'auto_sync.bat' anytime to sync." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit
    }
    default {
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        exit
    }
}

# Create the action
$action = New-ScheduledTaskAction -Execute $scriptPath -WorkingDirectory $PSScriptRoot

# Create the task settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Auto-sync Stress Detection project to GitHub $intervalText" -ErrorAction Stop
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Auto-sync successfully configured!" -ForegroundColor Green
    Write-Host "Your code will sync to GitHub $intervalText" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can also manually run 'auto_sync.bat' anytime." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To disable auto-sync, run:" -ForegroundColor Yellow
    Write-Host "Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false" -ForegroundColor Yellow
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "You may need to run this script as Administrator." -ForegroundColor Yellow
    Write-Host "Alternatively, you can manually run 'auto_sync.bat' anytime." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
