import pprint
import logging
import os
import sys
import numpy as np
import shutil
from deepdiff import DeepDiff

from pymooCFD.util.loggingTools import MultiLineFormatter, DispNameFilter
import pymooCFD.config as config
from pymooCFD.util.sysTools import yes_or_no


class PicklePath:
    def __init__(self, dir_path=None, sub_dirs=[], log_level=logging.DEBUG):
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
        self.log_level = log_level
        self.logger = self.get_logger()
        # INTRODUCTION MESSAGE
        intro_str = 'INITIALIZING - Pickle Path'
        self.logger.info('~' * len(intro_str))
        self.logger.info(intro_str)
        #########################
        #    Checkpoint Load    #
        #########################
        self.cp_init = False
        if os.path.isdir(self.abs_path):
            # try:
            if os.path.exists(self.cp_path):
                self.update_self()
                # self.load_sub_pickle_paths()
                self.logger.info('RESTARTED FROM CHECKPOINT')
                self.cp_init = True
                return
            # except FileNotFoundError:
            else:
                question = f'\n{self.rel_path} exists but {self.cp_rel_path} does not.\n\tEMPTY {self.rel_path} DIRECTORY?'
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
            # self.logger = self.get_logger()
            self.logger.info(f'NEW - {self.rel_path} did not exist')

        self.load_sub_pickle_paths()

        # for i, sub_dir in enumerate(sub_dirs):
        #     sub_dirs[i] = os.path.join(self.abs_path, sub_dir)
        #     # os.makedirs(sub_dirs[i]) #, exist_ok=True)
        # self.sub_dirs = sub_dirs

        # os.makedirs(self.abs_path, exist_ok=True)

        # for path in sub_dirs:
        #     os.makedirs(path, exist_ok=True)
    def load_sub_pickle_paths(self):
        # Check for other PicklePath child class and load if found
        for attr_key, attr_val in self.__dict__.items():
            # try:
            # if isinstance(attr_val, list): # hasattr(attr_val, "__getitem__")
            if self.is_iterable(attr_val):
                # if (isinstance(attr_val, list) or
                #     isinstance(attr_val, set) or
                #     isinstance(attr_val, tuple)):
                #     for item in attr_val:
                #         self.loadIfPP(item)
                if isinstance(attr_val, dict):
                    for key, val in attr_val.items():
                        self.loadIfPP(val)
                else:
                    for item in attr_val:
                        self.loadIfPP(item)
                    # self.logger.warning('Unknown Iterable Object')
                    # self.logger.warning(f'ITEMS INSIDE ITERABLE ATTRIBUTE NOT LOADED - {attr_key} : {attr_val}')

            # type(attr_val).mro(): #
            self.loadIfPP(attr_val)
            # if __class__ in attr_val.__class__.mro():
            #     self.logger.info(
            #         f'LOADING: {attr_key} FROM {attr_val.cp_path}')
            #     attr_val.update_self()
            # except AttributeError as err:
            #     self.logger.error(err)

    def save_sub_pickle_paths(self):
        # Check for other PicklePath child class and load if found
        for attr_key, attr_val in self.__dict__.items():
            # try:
            # if isinstance(attr_val, list): # hasattr(attr_val, "__getitem__")
            # print(attr_key, ':', attr_val)
            # print(self.is_iterable(attr_val))
            if self.is_iterable(attr_val):
                # if (isinstance(attr_val, list) or
                #     isinstance(attr_val, set) or
                #     isinstance(attr_val, tuple)):
                #     for item in attr_val:
                #         self.saveIfPP(item)
                if isinstance(attr_val, dict):
                    for key, val in attr_val.items():
                        self.saveIfPP(val)
                else:
                    for item in attr_val:
                        self.saveIfPP(item)
                    # self.logger.warning('Unknown Iterable Object')
                    # self.logger.warning(f'ITEMS INSIDE ITERABLE ATTRIBUTE NOT SAVED - {attr_key} : {attr_val}')
                    # if __class__ in item.__class__.mro():  # type(attr_val).mro(): #
                    #     self.logger.info(
                    #         f'LOADING: {attr_key}[{i}] FROM {item.cp_path}')
                    #     item.update_self()
            # type(attr_val).mro(): #
            self.saveIfPP(attr_val)
            # if __class__ in attr_val.__class__.mro():
            #     self.logger.info(
            #         f'LOADING: {attr_key} FROM {attr_val.cp_path}')
            #     attr_val.update_self()
            # except AttributeError as err:
            #     self.logger.error(err)

    def loadIfPP(self, inst):
        if self.is_pickle_path(inst):
            self.logger.info(
                f'LOADING: {inst} FROM {inst.cp_path}')
            inst.update_self()

    def saveIfPP(self, inst):
        # try:
        #     inst.__class__.mro()
        # except TypeError as err:
        #     self.logger.error(err)
        #     return
        if self.is_pickle_path(inst):
            self.logger.info(
                f'SAVING: {inst} TO {os.path.relpath(inst.cp_path)}')
            inst.save_self()

    def get_logger(self):
        name = '.'.join(os.path.normpath(self.rel_path).split(os.path.sep))
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        # logger.propagate = False
        # if logger.handlers:
        #     logger.handlers.clear()

        # FORMATTER
        formatter = MultiLineFormatter(
            '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s',
            "%m-%d %H:%M")

        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logger.critical("Uncaught exception", exc_info=(exc_type, exc_value,
                                                            exc_traceback))

        sys.excepthook = handle_exception

        # STREAM
        if not logger.hasHandlers():
            streamHandler = logging.StreamHandler()  # sys.stdout)
            streamHandler.setFormatter(formatter)
            logger.addHandler(streamHandler)

        # FILE
        fileHandler = logging.FileHandler(self.log_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        # FILTER
        # filt = DispNameFilter(self.optName)
        # logger.addFilter(filt)

        return logger

    def save_self(self):
        if os.path.isdir(self.abs_path):
            self.saveNumpyFile(self.cp_path, self)
            self.logger.info('CHECKPOINT SAVED')
        else:
            self.logger.warning('FAILED: SAVING CHECKPOINT - directory does not exist')
        self.save_sub_pickle_paths()

    def load_self(self):
        if os.path.exists(self.cp_path):
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
        else:
            self.logger.error(f'CHECKPOINT NOT FOUND: {self.cp_path}')

    def update_self(self, loaded_self=None):
        if loaded_self is None:
            loaded_self = self.load_self()
        if loaded_self is None:
            self.logger.error('FAILED UPDATE SELF: LOADED CHECKPOINT is None')
            return
        self.update_warnings(loaded_self)
        loaded_self = self._update_filter(loaded_self)
        # UPDATE: only instance dictonary
        # class dictionary/code changes between checkpoints will be reflected
        self.__dict__.update(loaded_self.__dict__)
        self.load_sub_pickle_paths()
        # log
        self.logger.debug('\tUPDATED DICTONARY')
        for key, val in self.__dict__.items():
            self.logger.debug(f'\t\t{key}: {val}')

    def update_warnings(self, loaded_self=None):
        if loaded_self is None:
            loaded_self = self.load_self()
        try:
            diff = DeepDiff(self, loaded_self)
        except TypeError:
            diff = ''
        pp = pprint.PrettyPrinter(width=41, indent=4)  # ,compact=True)
        if diff:
            self.logger.warning(pp.pformat(diff))
        # for key, val in self.__dict__.items():
        #     if key in loaded_self.__dict__:
        #         loaded_val = loaded_self.__dict__[key]
        #         if not all(val) == all(loaded_val):
        #             self.logger.warning(f'UPDATED {key}: {val} -> {loaded_val}')
        # if self.abs_path != loaded_self.abs_path:
        #     self.logger.warning('PATH CHANGED BETWEEN CHECKPOINTS')
        #     self.logger.debug(str(loaded_self.rel_path) + ' -> ' + str(self.rel_path))
        # if loaded_self.cp_path != self.cp_path:
        #     self.logger.warning('CHECKPOINT PATH CHANGED BETWEEN CHECKPOINTS')
        #     self.logger.warning(f'{loaded_self.cp_path} != {self.cp_path}')

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

    def _update_filter(self, loaded_self):
        return loaded_self

    @staticmethod
    def is_iterable(obj):
        try:
            iter(obj)
        except TypeError:
            return False
        else:
            return True

    @staticmethod
    def is_pickle_path(inst):
        if inst.__class__ is not type and __class__ in inst.__class__.mro():
            return True

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
        return os.path.join(self.abs_path, self.__class__.__name__ +
                            '.checkpoint.npy')

    @property
    def cp_rel_path(self):
        return os.path.relpath(self.cp_path)

    @property
    def par_path(self):
        return os.path.dirname(self.abs_path)

    @property
    def log_path(self):
        os.makedirs(self.par_path, exist_ok=True)
        return os.path.join(self.par_path, self.data_folder + '.log')
