import logging
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=".env.local")

log_file_path = os.environ.get('PYDUINO_SLAVE_LOG','/var/log/pyduino/slave.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)