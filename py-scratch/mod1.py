import mod2
import logging


logger = logging.getLogger('mod1')

# STREAM
if not logger.hasHandlers():
    streamHandler = logging.StreamHandler()  # sys.stdout)
    # streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

# FILE
fileHandler = logging.FileHandler('mod1.log')
# fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# FILTER
# filt = DispNameFilter(self.optName)
# logger.addFilter(filt)

logger.error("TEST")

mod2.log()
