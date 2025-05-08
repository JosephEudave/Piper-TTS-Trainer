# PowerShell script to run the installer in WSL

Write-Host "Running piper_phonemize installer in WSL..." -ForegroundColor Cyan

# Check if WSL is available
$wslCheck = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL does not appear to be installed or working correctly." -ForegroundColor Red
    Write-Host "Please install WSL by running: wsl --install" -ForegroundColor Yellow
    Exit 1
}

# Make sure the scripts are executable in WSL
Write-Host "Setting execute permissions on scripts..." -ForegroundColor Cyan
wsl -e chmod +x install_precompiled_phonemize.sh fix_phonemize_poetry.sh rebuild_phonemize.sh

# Change directory to the project directory in WSL
$projectDir = $PWD.Path.Replace('\', '/')
$wslPath = "/mnt/c" + $projectDir.Substring(2)
Write-Host "Project directory: $wslPath" -ForegroundColor Cyan

# Run the installer script in WSL
Write-Host "Running installer in WSL..." -ForegroundColor Cyan
wsl -e bash -c "cd '$wslPath' && ./install_precompiled_phonemize.sh"

# Check the result
if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "You can now run the GUI with: ./run_gui.sh" -ForegroundColor Green
} else {
    Write-Host "Installation failed with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Please check the logs above for errors." -ForegroundColor Yellow
    Write-Host "You may need to try the fix script: ./fix_phonemize_poetry.sh" -ForegroundColor Yellow
} 