import logging
import os
import numpy as np

from pymooCFD.util.loggingTools import MultiLineFormatter, DispNameFilter
import pymooCFD.config as config
from pymooCFD.util.sysTools import yes_or_no


class PicklePath:
    def __init__(self, dir_path=None, sub_dirs=[]):
        if dir_path is None:
            dir_path = self.__class__.__name__
        self.abs_path = os.path.abspath(dir_path)
        self.rel_path = os.path.relpath(self.abs_path)
        _, self.data_folder = os.path.join(self.abs_path)
        self.cp_path = os.path.join(self.abs_path, self.data_folder+'.checkpoint.npy')
        #########################
        #    Checkpoint Load    #
        #########################
        if os.path.isdir(self.abs_path):
            try:
                self.update_self()
                self.logger.info('RESTARTED FROM CHECKPOINT')
                return
            except FileNotFoundError:
                question = f'\n{self.CP_path} does not exist.\nEMPTY {self.optDatDir} DIRECTORY?'
                overwrite = yes_or_no(question)
                if overwrite:
                    shutil.rmtree(self.optDatDir)
                    os.mkdir(self.optDatDir)
                    self.logger.info(f'EMPTIED - {self.optDatDir}')
                else:
                    self.logger.info(f'KEEPING FILES - {self.optDatDir}')
                self.logger.info('RE-INITIALIZING OPTIMIZATION ALGORITHM')
        else:
            os.makedirs(self.optDatDir)
            self.logger.info(
                f'NEW OPTIMIZATION STUDY - {self.optDatDir} did not exist')
        for i, sub_dir in enumerate(sub_dirs):
            sub_dirs[i] = os.path.join(self.abs_path, sub_dir)
            # os.makedirs(sub_dirs[i]) #, exist_ok=True)
        self.sub_dirs = sub_dirs

        os.makedirs(self.abs_path, exist_ok=True)
        # for path in sub_dirs:
        #     os.makedirs(path, exist_ok=True)
        def getLogger():
            log_level = config.OPT_STUDY_LOGGER_LEVEL
            # create root logger
            logger = logging.getLogger(__name__+'.'+self.rel_path)
            logger.setLevel(config.OPT_STUDY_LOGGER_LEVEL)
            # define handlers
            # if not logger.handlers:
            # file handler
            fileHandler = logging.FileHandler(f'{self.data_folder}.log')
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
                '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s')
            #     f'%(asctime)s :: %(levelname)-8s :: {self.optName} :: %(message)s')
            #     f'%(asctime)s.%(msecs)03d :: %(levelname)-8s :: {self.optName} :: %(message)s')
            #     '%(name)s :: %(levelname)-8s :: %(message)s')
            fileHandler.setFormatter(formatter)
            streamHandler.setFormatter(formatter)
            intro_str = 'INITIALIZED: logger'
            logger.info('~' * len(intro_str))
            logger.info(intro_str)
            return logger
        self.logger = getLogger()

    # def makePaths(self):
    #     pass

    def save_self(self):
        self.saveNumpyFile(self.cp_path, self)

    def get_self(self):
        return self.loadNumpyFile(self.cp_path)

    def update_self(self, loaded_self=None):
        if loaded_self is None:
            loaded_self = self.get_self
        self.__dict__.update(loaded_self.__dict__)
        # logging
        self.logger.info(f'\tCHECKPOINT LOADED: {self.CP_path}.npy')
        self.logger.debug('\tRESTART DICTONARY')
        for key in self.__dict__:
            self.logger.debug(f'\t\t{key}: {self.__dict__[key]}')
        self.logger.debug('\tCHECKPOINT DICTONARY')
        for key in cp.__dict__:
            self.logger.debug(f'\t\t{key}: {cp.__dict__[key]}')



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
