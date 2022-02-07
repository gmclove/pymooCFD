
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
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace('\n', '\n' + parts[0])

        return msg


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
