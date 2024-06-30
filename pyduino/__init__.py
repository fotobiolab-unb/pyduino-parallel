import logging
import os
from dotenv import load_dotenv
from .paths import PATHS

load_dotenv()
load_dotenv(dotenv_path=".env.local")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

log_file_path = os.environ.get('PYDUINO_LOG','/var/log/pyduino/pyduino.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=PATHS.SYSTEM_PARAMETERS.get('log_level', logging.INFO),
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
