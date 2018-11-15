import logging


def set_logger(model_id):
    logger = logging.getLogger(model_id)
    logger.setLevel(logging.INFO)
    if model_id:
        formatter = logging.Formatter(
            '%(levelname)-.1s:' + model_id + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
            '%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(
            '%(levelname)-.1s:[%(filename)s:%(lineno)d]:%(message)s', datefmt=
            '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger
