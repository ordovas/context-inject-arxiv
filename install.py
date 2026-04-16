#!/usr/bin/env python3
"""
Cross-platform installation script for the ArXiv Paper Retrieval Agent.
Sets up a virtual environment and installs all required dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(cmd, check=True, shell=isinstance(cmd, str))
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error during {description}: {e}")
        return False


def get_venv_path():
    """Get the virtual environment path."""
    return Path(".venv")


def get_python_executable():
    """Get the path to the Python interpreter in the venv."""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def activate_venv_command():
    """Get the command to activate the venv."""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "activate.bat")
    else:
        return f"source {venv_path}/bin/activate"


def main():
    """Main installation flow."""
    print("\n" + "="*60)
    print("🚀 ArXiv Paper Retrieval Agent - Installation Script")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if Python 3.9+ is available
    if sys.version_info < (3, 9):
        print(f"\n✗ Python 3.9+ is required (found {sys.version_info.major}.{sys.version_info.minor})")
        return False
    
    venv_path = get_venv_path()
    
    # Step 1: Create virtual environment
    if venv_path.exists():
        print(f"\n✓ Virtual environment already exists at {venv_path}")
    else:
        if not run_command([sys.executable, "-m", "venv", str(venv_path)], 
                          "Creating virtual environment"):
            return False
    
    # Step 2: Upgrade pip, setuptools, wheel
    python_exe = str(get_python_executable())
    if not run_command([python_exe, "-m", "pip", "install", "--upgrade", 
                       "pip", "setuptools", "wheel"],
                      "Upgrading pip, setuptools, and wheel"):
        return False
    
    # Step 3: Install dependencies
    if not run_command([python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
                      "Installing dependencies from requirements.txt"):
        return False
    
    # Step 4: Success message
    print("\n" + "="*60)
    print("✓ Installation completed successfully!")
    print("="*60)
    print(f"\nTo activate the virtual environment, run:")
    print(f"\n  {activate_venv_command()}\n")
    print("Then you can run the agent:")
    print(f"\n  python -m src.adapters.inbound.rest_api.app  # Start REST API")
    print(f"  jupyter notebook run.ipynb                   # Run notebooks")
    print(f"  python -c 'from src.application.usecases.arxiv import run_arxiv_agent'  # Use Python API\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
