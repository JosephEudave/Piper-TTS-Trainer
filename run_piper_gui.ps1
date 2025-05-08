# PowerShell script to run Piper TTS Trainer GUI in WSL

Write-Host "Piper TTS Trainer - GUI Launcher" -ForegroundColor Cyan

# Check if WSL is available
$wslCheck = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL does not appear to be installed or working correctly." -ForegroundColor Red
    Write-Host "Please install WSL by running: wsl --install" -ForegroundColor Yellow
    Exit 1
}

# Get project directory path
$projectDir = $PWD.Path.Replace('\', '/')
$wslPath = "/mnt/c" + $projectDir.Substring(2)
Write-Host "Project directory: $wslPath" -ForegroundColor Cyan

# Make sure the script is executable
Write-Host "Setting execute permissions on scripts..." -ForegroundColor Cyan
wsl -e chmod +x run_gui.sh

# Run the GUI script in WSL
Write-Host "Starting Piper TTS Trainer GUI in WSL..." -ForegroundColor Green
Write-Host "The interface will be available at: http://localhost:7860" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
wsl -e bash -c "cd '$wslPath' && ./run_gui.sh"
