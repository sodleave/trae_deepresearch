import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

def setup_logging():
    """Configures the logging system."""
    logger = logging.getLogger('DeepResearch')
    logger.setLevel(logging.DEBUG)  # Record all levels

    if not logger.handlers:
        # Create file handler
        file_handler = logging.FileHandler('deepresearch.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Set format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()
