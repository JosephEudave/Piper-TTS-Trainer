# Piper TTS Trainer - Windows Setup Guide

## Quick Start for Windows Users

1. **Run `install_wsl.bat`** (double-click it) to set up WSL with Ubuntu
   - This will install WSL if needed
   - This will install Ubuntu 22.04 if needed
   - You may need to restart your computer after WSL installation

2. **Follow the on-screen instructions**
   - After WSL is set up, Ubuntu will open automatically
   - Navigate to this project directory in Ubuntu
   - Run the Linux setup script

## Why WSL?

Piper TTS works best in Linux environments. WSL allows you to run Linux inside your Windows system without dual-booting or using a virtual machine.

## Troubleshooting

If you encounter issues with the WSL installation:

1. Make sure you're running Windows 10 version 2004 or higher
2. Try running PowerShell as Administrator and use:
   ```
   wsl --install -d Ubuntu-22.04
   ```
3. After installation, run `wsl --set-default-version 2`

## Manual Setup

If the automatic setup doesn't work, you can:

1. Install WSL 2 manually following [Microsoft's guide](https://learn.microsoft.com/en-us/windows/wsl/install)
2. Install Ubuntu 22.04 from the Microsoft Store
3. Open Ubuntu and navigate to this project:
   ```
   cd /mnt/c/path/to/Piper-TTS-Trainer
   ```
4. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ``` 