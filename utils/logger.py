import os
import logging


class LoggerHelper:
    def __init__(self, log_name="example"):
        # create logger
        logger_name = log_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.__addStreamHanlder()

    def __setFormatter(self):
        # create formatter
        fmt = "%(asctime)-15s %(levelname)s %(filename)s line:%(lineno)d pid:%(process)d %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter(fmt, datefmt)

    def __addStreamHanlder(self):
        # add std console handler and formatter to logger
        sh = logging.StreamHandler(stream=None)
        sh.setLevel(logging.DEBUG)
        fmt = "%(asctime)-15s %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def addFileHanlder(self, log_path, log_name):
        os.makedirs(log_path, exist_ok=True)
        # create file handler
        log_file_path = os.path.join(log_path, log_name)
        fh = logging.FileHandler(log_file_path)
        print("create FileHandler in {}".format(log_file_path))
        fh.setLevel(logging.INFO)

        fmt = "%(asctime)-15s  %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)


def get_log(log_path, save_log_name):
    logHelper = LoggerHelper(log_name=save_log_name)
    logHelper.addFileHanlder(log_path=log_path, log_name=save_log_name + ".log")
    logger = logHelper.logger
    return logger
