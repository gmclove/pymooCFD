import logging


def log():
    logger = logging.getLogger('mod1.mod2')
    print(logger.hasHandlers())
    print(logger.handlers)
    logger.propogate = False

    # FILE
    fileHandler = logging.FileHandler('mod2.log')
    # fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # FILTER
    # filt = DispNameFilter(self.optName)
    # logger.addFilter(filt)

    # STREAM
    streamHandler = logging.StreamHandler()  # sys.stdout)
    # streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    logger.error('MOD2 TEST')
