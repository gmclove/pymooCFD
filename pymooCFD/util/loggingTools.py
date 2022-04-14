
import logging


class MultiLineFormatter(logging.Formatter):

    def __init__(self, fmt, datefmt=None):
        """
        Init given the log line format and date format
        """
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        """
        Override format function
        """
        if record.levelno == logging.WARNING:
            record.msg = '\033[93m%s\033[0m' % record.msg
        elif record.levelno == logging.ERROR:
            record.msg = '\033[91m%s\033[0m' % record.msg
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace('\n', '\n' + parts[0])

        return msg
#
#
# class ColoredFormatter(logging.Formatter):
#     def format(self, record):
#         if record.levelno == logging.WARNING:
#             record.msg = '\033[93m%s\033[0m' % record.msg
#         elif record.levelno == logging.ERROR:
#             record.msg = '\033[91m%s\033[0m' % record.msg
#         return logging.Formatter.format(self, record)
#
# class CustomFormatter(logging.Formatter):
#
#     grey = "\x1b[38;20m"
#     yellow = "\x1b[33;20m"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
#
#     FORMATS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: grey + format + reset,
#         logging.WARNING: yellow + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset
#     }
#
#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)

class DispNameFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.

    Rather than use actual contextual information, we just use random
    data in this demo.
    """
    def __init__(self, dispName, name=''):
        self.dispName = dispName
        super().__init__(name=name)

    def filter(self, record):
        record.msg = f'{self.dispName} :: {record.msg}'
        return True



# from colorlog import ColoredFormatter
#
# formatter = ColoredFormatter(
#         "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
#         datefmt=None,
#         reset=True,
#         log_colors={
#                 'DEBUG':    'cyan',
#                 'INFO':     'green',
#                 'WARNING':  'yellow',
#                 'ERROR':    'red',
#                 'CRITICAL': 'red',
#         }
# )
