"""Setup script for FinSight AI development environment."""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("✗ Python 3.9 or higher is required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def setup_environment():
    """Set up the development environment."""
    
    print("="*60)
    print("FINSIGHT AI SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\nCreating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✓ .env file created")
        print("  ⚠️  Please edit .env and add your OpenAI API key")
    elif env_file.exists():
        print("✓ .env file already exists")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create necessary directories
    directories = [
        "data/chroma_db",
        "data/documents", 
        "data/feedback",
        "logs"
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")
    
    # Download required NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK download failed: {e}")
    
    # Run basic tests
    print("\nRunning basic validation tests...")
    if run_command("python -m pytest tests/test_basic_functionality.py -v", "Running tests"):
        print("✓ All tests passed")
    else:
        print("⚠️  Some tests failed - check configuration")
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    
    print("\n" + "="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    
    print("\nNext Steps:")
    print("1. Edit .env file and add your OpenAI API key:")
    print("   OPENAI_API_KEY=your_api_key_here")
    
    print("\n2. Add sample financial documents to data/sample_reports/:")
    print("   - data/sample_reports/castrol/castrol_annual_report_2023.pdf")
    print("   - data/sample_reports/veedol/veedol_annual_report_2023.pdf")
    print("   - data/sample_reports/valvoline/valvoline_annual_report_2023.pdf")
    
    print("\n3. Ingest the documents:")
    print("   python scripts/ingest_reports.py --directory ./data/sample_reports")
    
    print("\n4. Test the system:")
    print("   python scripts/test_system.py")
    
    print("\n5. Start the API server:")
    print("   uvicorn src.api.main:app --reload")
    
    print("\n6. Access the API documentation:")
    print("   http://localhost:8000/docs")
    
    print("\nFor detailed documentation, see:")
    print("- docs/GETTING_STARTED.md")
    print("- docs/API_REFERENCE.md")
    print("- README.md")


def main():
    """Main setup function."""
    
    try:
        if setup_environment():
            print_next_steps()
        else:
            print("\n✗ Setup failed. Please check the errors above and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Setup failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
