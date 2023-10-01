import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/var/log/pyduino/slave.log'),
        logging.StreamHandler()
    ]
)