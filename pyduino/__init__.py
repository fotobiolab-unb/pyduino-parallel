import logging
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=".env.local")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.environ.get('PYDUINO_SLAVE_LOG','/var/log/pyduino/slave.log')),
        logging.StreamHandler()
    ]
)