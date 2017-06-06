import logging

def default_logging(name):
    # prints log to stdout and also saves to specified log file
    logger = logging.getLogger(name)
    fh = logging.FileHandler(name + '.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger
