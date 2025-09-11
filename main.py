"""Main entry point for the LangGraph Data Analysis Agent."""
import sys
import warnings
from pathlib import Path

# Suppress BigQuery Storage warnings for cleaner user experience
warnings.filterwarnings("ignore", message="BigQuery Storage module not found")
warnings.filterwarnings("ignore", category=UserWarning, module="google.cloud.bigquery")

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli.main import main as cli_main  # noqa: E402
from config import config  # noqa: E402
from logging_config import get_logger


logger = get_logger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    from logging_config import setup_logging as setup_centralized_logging
    
    # Use centralized logging configuration
    log_file = config.logging_settings.file_path or "logs/agent.log"
    setup_centralized_logging(debug=debug, log_file=log_file)


def check_environment() -> None:
    """Check that required environment variables and dependencies are available."""
    missing_vars = []
    
    # Check for required API keys
    if not config.api_configurations.gemini_api_key:
        missing_vars.append("GEMINI_API_KEY")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\\n📝 Setup Instructions:")
        print("1. Create a .env file in the project root")
        print("2. Add your Gemini API key:")
        for var in missing_vars:
            print(f"   {var}=your_api_key_here")
        print("\\n🔗 Get your Gemini API key from: https://makersuite.google.com/app/apikey")
        print("\\n🚀 Run 'python3 setup_environment.py' for guided setup")
        sys.exit(1)
    
    # Check BigQuery configuration
    if not config.api_configurations.bigquery_project_id:
        print("⚠️  Warning: No BigQuery project ID set. Using default credentials.")
    
    print("✅ Environment check passed")


def main() -> None:
    """Main entry point."""
    try:
        # Setup logging first
        setup_logging(debug=False)
        
        # Check environment
        check_environment()
        
        # Start CLI (pass debug flag if needed)
        debug_mode = '--debug' in sys.argv
        if debug_mode:
            setup_logging(debug=True)
        
        cli_main()
        
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
