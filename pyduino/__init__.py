import logging
import os
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.environ.get('PYDUINO_SLAVE_LOG','/var/log/pyduino/slave.log')),
        logging.StreamHandler()
    ]
)