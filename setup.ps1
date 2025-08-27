#Requires -Version 5.0
$ErrorActionPreference = 'Stop'

if (-not $IsWindows) {
    Write-Error 'setup.ps1 should be run on Windows.'
    exit 1
}

function Install-Python {
    Write-Host 'Python not found. Attempting installation...'
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host 'Installing Python 3.11 via winget.'
        try {
            winget install --id Python.Python.3.11 -e --source winget
        } catch {
            Write-Error 'Failed to install Python via winget. Please install it manually from https://www.python.org/downloads/'
            exit 1
        }
    } else {
        Write-Error 'winget is not available. Install Python manually from https://www.python.org/downloads/'
        exit 1
    }
}

function Install-Poetry {
    Write-Host 'Poetry not found. Attempting installation...'
    try {
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    } catch {
        Write-Error 'Failed to install Poetry. See https://python-poetry.org/docs/#installation for help.'
        exit 1
    }
}

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Install-Python
} else {
    $pyVersion = & python --version
    Write-Host "Python found: $pyVersion"
}

# Check Poetry
$poetry = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetry) {
    Install-Poetry
} else {
    $poetryVersion = & poetry --version
    Write-Host "Poetry found: $poetryVersion"
}

Write-Host 'Environment setup complete.'
