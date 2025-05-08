# PowerShell script to install WSL and Ubuntu

Write-Host "Installing Windows Subsystem for Linux (WSL)..." -ForegroundColor Yellow

# Enable WSL feature
Write-Host "Enabling WSL feature..." -ForegroundColor Yellow
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
Write-Host "Enabling Virtual Machine Platform..." -ForegroundColor Yellow
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "Please restart your computer now, then run this script again to continue installation." -ForegroundColor Green
Write-Host "After restart, press any key to continue installation..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Set WSL 2 as default
Write-Host "Setting WSL 2 as default..." -ForegroundColor Yellow
wsl --set-default-version 2

# Install Ubuntu
Write-Host "Installing Ubuntu... This might take a while." -ForegroundColor Yellow
Write-Host "If the Microsoft Store opens, please follow the prompts to install Ubuntu." -ForegroundColor Yellow
wsl --install -d Ubuntu

Write-Host "WSL and Ubuntu installation completed!" -ForegroundColor Green
Write-Host "You can now launch Ubuntu from the Start menu" -ForegroundColor Green
Write-Host "After launching Ubuntu, navigate to your project directory using:" -ForegroundColor Yellow
Write-Host "cd /mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer" -ForegroundColor Cyan
Write-Host "Then run the setup script:" -ForegroundColor Yellow
Write-Host "bash setup_poetry.sh" -ForegroundColor Cyan 