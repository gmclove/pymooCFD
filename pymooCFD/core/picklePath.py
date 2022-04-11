import logging
import os
import numpy as np
import shutil

from pymooCFD.util.loggingTools import MultiLineFormatter, DispNameFilter
import pymooCFD.config as config
from pymooCFD.util.sysTools import yes_or_no


class PicklePath:
    def __init__(self, dir_path=None, sub_dirs=[]):
        if dir_path is None:
            dir_path = self.__class__.__name__
        self.abs_path = os.path.abspath(dir_path)
        # make the following into properties
        # self.rel_path = os.path.relpath(self.abs_path)
        # _, self.data_folder = os.path.split(self.abs_path)
        # self.cp_path = os.path.join(self.abs_path, self.data_folder+'.checkpoint.npy')
        # self.cp_rel_path = os.path.relpath(self.cp_path)
        # self.par_path = os.path.join(self.abs_path, os.pardir)
        # self.log_path = os.path.join(self.par_path, self.data_folder+'.log')
        self.init_logger()
        #########################
        #    Checkpoint Load    #
        #########################
        if os.path.isdir(self.abs_path):
            try:
                self.update_self()
                self.logger.info('RESTARTED FROM CHECKPOINT')
                return
            except FileNotFoundError:
                question = f'\n{self.cp_rel_path} does not exist.\n\
                                EMPTY {self.rel_path} DIRECTORY?'
                overwrite = yes_or_no(question)
                if overwrite:
                    shutil.rmtree(self.abs_path)
                    os.mkdir(self.abs_path)
                    self.logger.info(f'EMPTIED - {self.rel_path}')
                else:
                    self.logger.info(f'KEEPING FILES - {self.rel_path}')
                # self.logger.info('RE-INITIALIZING OPTIMIZATION ALGORITHM')
        else:
            os.makedirs(self.abs_path)
            self.init_logger()
            self.logger.info(
                f'NEW - {self.rel_path} did not exist')
        for i, sub_dir in enumerate(sub_dirs):
            sub_dirs[i] = os.path.join(self.abs_path, sub_dir)
            # os.makedirs(sub_dirs[i]) #, exist_ok=True)
        self.sub_dirs = sub_dirs

        os.makedirs(self.abs_path, exist_ok=True)
        # for path in sub_dirs:
        #     os.makedirs(path, exist_ok=True)

    def init_logger(self):
        # log_level = config.OPT_STUDY_LOGGER_LEVEL
        name = '.'.join(os.path.normpath(self.rel_path).split(os.path.sep))
        logger = logging.getLogger(name)
        logger.setLevel(config.OPT_STUDY_LOGGER_LEVEL)
        # define handlers
        # if not logger.handlers:
        # file handler
        fileHandler = logging.FileHandler(self.log_path)
        logger.addHandler(fileHandler)
        # stream handler
        streamHandler = logging.StreamHandler()  # sys.stdout)
        streamHandler.setLevel(config.OPT_STUDY_LOGGER_LEVEL)
        if streamHandler not in logger.handlers:
            logger.addHandler(streamHandler)
        # define filter
        # filt = DispNameFilter(self.optName)
        # logger.addFilter(filt)
        # define formatter
        formatter = MultiLineFormatter(
            '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s',
            "%m-%d %H:%M:%S")
        #     f'%(asctime)s :: %(levelname)-8s :: {self.optName} :: %(message)s')
        #     f'%(asctime)s.%(msecs)03d :: %(levelname)-8s :: {self.optName} :: %(message)s')
        #     '%(name)s :: %(levelname)-8s :: %(message)s')
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)
        intro_str = 'INITIALIZED: logger'
        logger.info('~' * len(intro_str))
        logger.info(intro_str)
        self.logger = logger

    def save_self(self):
        self.saveNumpyFile(self.cp_path, self)

    def load_self(self):
        loaded_self = self.loadNumpyFile(self.cp_path)
        # logging
        self.logger.info(f'\tCHECKPOINT LOADED: {self.cp_rel_path}')
        self.logger.debug('\tRESTART DICTONARY')
        for key, val in self.__dict__.items():
            self.logger.debug(f'\t\t{key}: {val}')
        self.logger.debug('\tCHECKPOINT DICTONARY')
        for key, val in loaded_self.__dict__.items():
            self.logger.debug(f'\t\t{key}: {val}')
        return loaded_self

    def update_self(self, loaded_self=None):
        if loaded_self is None:
            loaded_self = self.load_self()
        # UPDATE
        self.__dict__.update(loaded_self.__dict__)

    def check_saves(self, print_info=False):
        if print_info:
            disp = print
        else:
            disp = self.logger.info
        disp(f'SAVED CHECKPOINTS CHECK - {self.cp_rel_path}')
        if os.path.exists(self.cp_path):
            for cp in self.loadNumpyFile(self.cp_path):
                disp(cp.__dict__)
        else:
            disp(f'\t{self.cp_path} does not exist')

    @staticmethod
    def loadNumpyFile(path):
        if not path.endswith('.npy'):
            path = path + '.npy'
        old_path = path.replace('.npy', '.old.npy')
        if os.path.exists(old_path):
            os.rename(old_path, path)
        files = np.load(path, allow_pickle=True).flatten()
        latest_file = files[0]
        return latest_file

    @staticmethod
    def saveNumpyFile(path, data):
        if not path.endswith('.npy'):
            path = path + '.npy'
        temp_path = path.replace('.npy', '.temp.npy')
        old_path = path.replace('.npy', '.old.npy')
        np.save(temp_path, data)
        if os.path.exists(path):
            os.rename(path, old_path)
        os.rename(temp_path, path)
        if os.path.exists(old_path):
            os.remove(old_path)
        return path

    @property
    def rel_path(self):
        return os.path.relpath(self.abs_path)

    @property
    def data_folder(self):
        _, data_folder = os.path.split(self.abs_path)
        return data_folder

    @property
    def cp_path(self):
        return os.path.join(self.abs_path, self.data_folder+'.checkpoint.npy')

    @property
    def cp_rel_path(self):
        return os.path.relpath(self.cp_path)

    @property
    def par_path(self):
        return os.path.join(self.abs_path, os.pardir)

    @property
    def log_path(self):
        return os.path.join(self.par_path, self.data_folder+'.log')
