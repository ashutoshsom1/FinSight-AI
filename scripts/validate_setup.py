"""Comprehensive validation script for FinSight AI setup."""

import sys
import os
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version compatibility."""
    print("1. Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 9:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Required: Python 3.9 or higher")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n2. Checking dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "openai", "langchain",
        "pypdf2", "pandas", "chromadb", "tiktoken", "structlog",
        "numpy", "requests", "aiofiles"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True


def check_environment_config():
    """Check environment configuration."""
    print("\n3. Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("   ✗ .env file not found")
        print("   Copy .env.example to .env and configure it")
        return False
    
    print("   ✓ .env file exists")
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                print(f"   ✗ {var} - NOT SET")
            else:
                print(f"   ✓ {var} - SET")
        
        if missing_vars:
            print(f"\n   Missing variables: {', '.join(missing_vars)}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error loading .env file: {e}")
        return False
    
    return True


def check_directory_structure():
    """Check if all required directories exist."""
    print("\n4. Checking directory structure...")
    
    required_dirs = [
        "src/data/ingestion",
        "src/data/storage", 
        "src/data/models",
        "src/preprocessing",
        "src/embeddings",
        "src/retrieval",
        "src/llm",
        "src/api",
        "src/evaluation",
        "tests",
        "data",
        "config",
        "scripts",
        "docs"
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"   ✓ {directory}")
        else:
            print(f"   ✗ {directory} - MISSING")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n   Missing directories: {', '.join(missing_dirs)}")
        return False
    
    return True


def check_configuration_files():
    """Check if configuration files are present and valid."""
    print("\n5. Checking configuration files...")
    
    config_files = [
        "config/settings.py",
        "requirements.txt",
        "pyproject.toml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"   ✓ {config_file}")
        else:
            print(f"   ✗ {config_file} - MISSING")
            return False
    
    # Test configuration loading
    try:
        sys.path.append(".")
        from config.settings import settings
        print("   ✓ Configuration loads successfully")
        print(f"   ✓ Target companies: {settings.get_target_companies_list()}")
        print(f"   ✓ Vector DB type: {settings.vector_db_type}")
    except Exception as e:
        print(f"   ✗ Configuration loading failed: {e}")
        return False
    
    return True


def check_openai_connection():
    """Test OpenAI API connection."""
    print("\n6. Testing OpenAI API connection...")
    
    try:
        import openai
        from config.settings import settings
        
        client = openai.OpenAI(api_key=settings.openai_api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("   ✓ OpenAI API connection successful")
        print(f"   ✓ Model: {response.model}")
        return True
        
    except Exception as e:
        print(f"   ✗ OpenAI API connection failed: {e}")
        print("   Check your API key and internet connection")
        return False


def check_sample_data():
    """Check for sample data availability."""
    print("\n7. Checking sample data...")
    
    sample_dir = Path("data/sample_reports")
    if not sample_dir.exists():
        print("   ✗ Sample reports directory not found")
        return False
    
    companies = ["castrol", "veedol", "valvoline"]
    found_files = 0
    
    for company in companies:
        company_dir = sample_dir / company
        if company_dir.exists():
            files = list(company_dir.glob("*.pdf")) + list(company_dir.glob("*.csv"))
            if files:
                print(f"   ✓ {company}: {len(files)} files found")
                found_files += len(files)
            else:
                print(f"   ⚠️  {company}: No files found")
        else:
            print(f"   ⚠️  {company}: Directory not found")
    
    if found_files == 0:
        print("   ⚠️  No sample data found")
        print("   Add PDF/CSV files to data/sample_reports/{company}/ directories")
        return False
    
    print(f"   ✓ Total sample files: {found_files}")
    return True


def run_basic_import_test():
    """Test basic imports of core modules."""
    print("\n8. Testing core module imports...")
    
    test_imports = [
        ("src.data.models.document", "DocumentMetadata"),
        ("src.preprocessing.text_cleaner", "FinancialTextCleaner"),
        ("src.embeddings.embedding_service", "EmbeddingService"),
        ("src.data.storage.vector_store", "ChromaVectorStore"),
    ]
    
    sys.path.append("src")
    
    for module_name, class_name in test_imports:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"   ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ✗ {module_name}.{class_name} - {e}")
            return False
    
    return True


def main():
    """Run all validation checks."""
    
    print("="*60)
    print("FINSIGHT AI SETUP VALIDATION")
    print("="*60)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_environment_config,
        check_directory_structure,
        check_configuration_files,
        check_openai_connection,
        check_sample_data,
        run_basic_import_test
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check in checks:
        try:
            if check():
                passed_checks += 1
        except Exception as e:
            print(f"   ✗ Check failed with error: {e}")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! FinSight AI is ready to use.")
        print("\nNext steps:")
        print("1. Run: make ingest (to ingest sample documents)")
        print("2. Run: make run (to start the API server)")
        print("3. Visit: http://localhost:8000/docs")
        
    elif passed_checks >= total_checks - 2:
        print("\n⚠️  Most checks passed. Minor issues detected.")
        print("The system should work, but consider fixing the issues above.")
        
    else:
        print("\n❌ Multiple issues detected. Please fix the errors above before proceeding.")
        print("\nCommon solutions:")
        print("- Run: pip install -r requirements.txt")
        print("- Copy .env.example to .env and configure it")
        print("- Add sample documents to data/sample_reports/")
        
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
