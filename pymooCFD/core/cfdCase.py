# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:33:00-05:00
# from pymooCFD.util.handleData import findKeywordLine
from pymoo.visualization.scatter import Scatter
import re
from pymooCFD.core.mesh_study import mesh_study
from pymooCFD.core.picklePath import PicklePath
from pymooCFD.util.sysTools import saveTxt, yes_or_no
import pymooCFD.config as config
import multiprocessing.pool
import multiprocessing as mp
import numpy as np
from glob import glob
import os
import subprocess
import time
import shutil
import matplotlib.pyplot as plt
plt.set_loglevel('info')
# import matplotlib.pyplot as plt
# plt.set_loglevel("info")

# import matplotlib.dates as mdates
# from matplotlib.ticker import AutoMinorLocator


class CFDCase(PicklePath):  # (PreProcCase, PostProcCase)
    '''
    Notes:
        - CFD cases with external solvers are launched using the subprocess module.
            The execution directory is set as the self.abs_path.
    '''
    base_case_path = None
    datFile = None
    inputFile = None
    meshFile = None
    ####### Define Design Space #########
    n_var = None
    var_labels = None
    var_type = None  # OPTIONS: 'int' or 'real'
    # xl = None  # lower limits of parameters/variables
    # xu = None  # upper limits of variables
    # if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
    #     raise ExceptionDesign Space Definition Incorrect")
    ####### Define Objective Space ########
    obj_labels = None
    n_obj = None
    ####### Define Constraints #########
    n_constr = None

    ##### External Solver #######
    externalSolver = False
    onlyParallelizeSolve = False
    procLim = None
    nProc = None
    nTasks = None
    solverExecCmd = None


    def __init__(self, case_path, x,
                 validated=False,
                 mesh_study=None,
                 # externalSolver=False,
                 var_labels=None, obj_labels=None,
                 meshFile=None,  # meshLines = None,
                 jobFile=None,  # jobLines = None,
                 inputFile=None,  # inputLines = None,
                 datFile=None,
                 # restart=False,
                 # solverExecCmd=None,
                 # *args,
                 **kwargs
                 ):
        self.meshSF = kwargs.get('meshSF', 1)

        ###########################
        #    RESTART VARIABLES    #
        ###########################
        # self.complete = False
        self.restart = False
        self.parallelizeInit(self.externalSolver)
        self._x = x

        #####################################
        #    CHECKPOINT/PICKLE PATH INIT    #
        #####################################
        super().__init__(dir_path=case_path)
        self.copy_base_case()
        saveTxt(self.abs_path, 'var.txt', self.x)
        #############################
        #    Optional Attributes    #
        #############################
        self.meshFile = meshFile
        self.jobFile = jobFile
        self.datFile = datFile
        self.inputFile = inputFile
        # self.jobFile = kwargs.get('jobFile')
        # self.inputFile = kwargs.get('inputFile')
        # self.datFile = kwargs.get('datFile')
        # self.meshFile = kwargs.get('meshFile')

        ####################
        #    Attributes    #
        ####################
        self.validated = validated
        # Default Attributes
        if mesh_study is None:
            self.mesh_study = mesh_study(self)
        # os.makedirs(self.basecase_path, exist_ok=True)
        # Using kwargs (not an option with labels as class variables)
        # self.var_labels = kwargs.get('var_labels')
        # self.obj_labels = kwargs.get('obj_labels')

        # Design and Objective Space Labels
        if self.var_labels is None:
            self.var_labels = [f'var{x_i}' for x_i in range(self.n_var)]
        else:
            self.var_labels = var_labels
        if self.obj_labels is None:
            self.obj_labels = [f'obj{x_i}' for x_i in range(self.n_obj)]
        else:
            self.obj_labels = obj_labels
        # if not len(self.xl) == len(self.xu) and len(self.xu) == len(self.var_labels) and len(self.var_labels) == self.n_var:
        #     raise Exception("Design Space Definition Incorrect")
        ###################################################
        #    Attributes To Be Set Later During Each Run   #
        ###################################################
        self.f = None  # if not None then case complete
        self.g = None
        # meshing attributes
        self.msCases = None
        self.numElem = None

        # case file lines
        self.inputLines = None
        self.jobLines = None

        self.solnTime = None

        self.logger.info('CASE INTITIALIZED')
        self.logger.debug('INITIAL CASE DICTONARY')
        for key, val in self.__dict__.items():
            self.logger.debug(f'\t{key}: {val}')
        # self.genMesh()
        ### Save Checkpoint ###
        self.save_self()

    # def run(self):
    #     self.preProc()
    #     proc = self.solve()
    #     proc.wait()
    #     obj = self.postProc()
    #     return obj

    # def solve(self):
    #     proc = self._solve()
    #     return proc
    # def slurmSolve(self):
    #     cmd = ['sbatch', '--wait', self.jobFile]
    #     proc = subprocess.Popen(cmd, cwd=self.case_path,
    #                             stdout=subprocess.DEVNULL)
    #     return proc
    ###  Parallel Processing  ###
    @classmethod
    def parallelizeInit(cls, externalSolver):
        # if externalSolver is None:
        #     externalSolver = cls.externalSolver
        if cls.nTasks is None:
            if cls.nProc is not None and cls.nProc is not None:
                cls.nTasks = int(cls.procLim / cls.nProc)
            else:
                cls.nTasks = config.MP_POOL_NTASKS_MAX
        else:
            nTasks = cls.nTasks
        if externalSolver:
            assert cls.solverExecCmd is not None
            assert cls.nTasks is not None
            cls._solve = cls.solveExternal
            cls.pool = mp.pool.ThreadPool(cls.nTasks)
        else:
            cls._solve = cls._solve
            cls.pool = mp.Pool(cls.nTasks)

    @classmethod
    def parallelize(cls, cases):
        # cls.parallelizeInit()
        #cls.logger.info('PARALLELIZING . . .')
        if cls.onlyParallelizeSolve:
            # print('\tParallelizing Only Solve')
            for case in cases:
                case.preProc()
            print('PARALLELIZING . . .')
            for case in cases:
                cls.pool.apply_async(case.solve, ())
            cls.pool.close()
            cls.pool.join()
            for case in cases:
                case.postProc()
        else:
            print('PARALLELIZING . . .')
            for case in cases:
                cls.pool.apply_async(case.run, ())
            cls.pool.close()
            cls.pool.join()

    # def parallelizeCleanUp(self):
    #     self.pool.terminate()
    # def solve(self):
    #     # if not self.complete:
    #     if self.f is None:
    #         self._solve()

    def solve(self):
        if self.f is None or np.isfinite(np.sum(self.f)):
            # try to prevent re-run if execution done
            if self._execDone() and self.restart:
                self.logger.debug(
                    'self.solve() called but self._execDone() and self.restart are both True')
                self.logger.warning('TRYING: POST-PROCESS BEFORE SOLVE')
                try:
                    self.postProc()
                except FileNotFoundError as err:
                    self.logger.error(err)
        if self.f is None or np.isfinite(np.sum(self.f)):
            self.restart = True
            start = time.time()
            self._solve()
            end = time.time()
            dt = end - start
            if dt >= 3600:
                hrs = int(dt / 3600)
                mins = int(dt % 3600 / 60)
                secs = dt % 60
                t_str = '%i hrs | %i mins | %i secs' % (hrs, mins, secs)
            elif dt >= 60:
                mins = int(dt / 60)
                secs = dt % 60
                t_str = '%i mins | %i secs' % (mins, secs)
            else:
                t_str = '%i secs' % dt
            self.logger.info(f'Solve Time: {t_str}')
            self.solnTime = dt
            self.save_self()
        else:
            self.logger.warning('SKIPPED: SOLVE')

    def solveExternal(self):
        #self.restart = True
        #start = time.time()
        self.logger.info('SOLVING AS SUBPROCESS...')
        self.logger.info(f'\tcommand: {self.solverExecCmd}')
        subprocess.run(self.solverExecCmd, cwd=self.abs_path,
                       stdout=subprocess.DEVNULL)
        #end = time.time()
        #self.logger.info(f'Solve Time: {start-end}')

    def run(self, max_reruns=3, n_reruns=0):
        # print('RUNNING')
        if self.f is None or np.isfinite(np.sum(self.f)):
            self.preProc()
            self.solve()
            self.postProc()
            if self._execDone():
                self.logger.info('COMPLETE: SOLVE')
            else:
                if n_reruns < max_reruns:
                    self.logger.warning('RUN FAILED TO EXECUTE')
                    self.logger.info('RE-RUNNING')
                    n_reruns += 1
                    self.run(max_reruns=max_reruns, n_reruns=n_reruns)
                else:
                    self.logger.warning(
                        f'MAX NUMBER OF RE-RUNS ({max_reruns}) REACHED')
            # self.postProc()
        else:
            self.logger.warning(
                'SKIPPED: RUN - self.run() called but case already complete')

    # def execDone(self):
    #     pass

    def preProc(self):
        if self.f is None or np.isfinite(np.sum(self.f)):
            if self.restart:
                # self.cp_rel_path = os.path.join
                self.logger.info(
                    'PRE-PROCESS RESTART - Using self._preProc_restart()')
                self._preProc_restart()
            else:
                self._preProc()
            # save variables in case directory as text file after completing pre-processing
            # saveTxt(self.abs_path, 'var.txt', self.x)
            self.logger.info('COMPLETE: PRE-PROCESS')
            # self.restart = True  # ??????????????????
            self.save_self()
        else:
            self.logger.warning('SKIPPED: PRE-PROCESS')
    # def pySolve(self):
    #     self.logger.info('SOLVING . . . ')
    #     self._pySolve()
    #     self.logger.info('SOLVED')
    # if self.solverExecCmd is None:
    #     self.logger.error('No external solver execution command give. \
    #                        Please override solve() method with python CFD \
    #                        solver or add solverExecCmd to CFDCase object.')
    #     raise Exception('No external solver execution command give. Please \
    #                     override solve() method with python CFD solver or \
    #                     add solverExecCmd to CFDCase object.')
    # else:
    #     subprocess.run(self.solverExecCmd, cwd=self.abs_path,
    #                    stdout=subprocess.DEVNULL)

    def postProc(self):
        if self.f is None or np.isfinite(np.sum(self.f)):
            self._postProc()
        else:
            self.logger.warning('SKIPPED: POST-PROCESSING')
            self.logger.debug(
                'self.postProc() called but self.f is not None or NaN so no action was taken')
        # Check Completion
        if self.f is None or np.isfinite(np.sum(self.f)):
            self.logger.error('INCOMPLETE: POST-PROCESS')
        else:
            self.logger.info('COMPLETE: POST-PROCESS')
        self.save_self()
        self.logger.info(f'\tObjectives: {self.f}')
        return self.f

    def genMesh(self):
        if self.meshPath is None:
            self.logger.warning(
                'self.genMesh() called but self.meshPath is None')
        if self.meshSF is None:
            self.logger.warning(
                'self.genMesh() called but self.meshPath is None')
        else:
            self._genMesh()
            self.logger.info('MESH GENERATED - Using self._genMesh()')

    ##########################
    #    CLASS PROPERTIES    #
    ##########################
    ### Case File Paths ###
    @property
    def inputPath(self):
        if self.inputFile is None:
            return None
        else:
            return os.path.join(self.abs_path, self.inputFile)

    @property
    def datPath(self):
        if self.datFile is None:
            return None
        else:
            return os.path.join(self.abs_path, self.datFile)

    @property
    def meshPath(self):
        if self.meshFile is None:
            return None
        else:
            return os.path.join(self.abs_path, self.meshFile)

    @property
    def jobPath(self):
        if self.jobFile is None:
            return None
        else:
            return os.path.join(self.abs_path, self.jobFile)

    @property
    def mesh_studyDir(self):
        return os.path.join(self.abs_path, 'mesh_study')

    # @property
    # def cp_path(self):
    #     return os.path.join(self.abs_path, 'case.npy')

    ### Job Lines ###
    @property
    def jobLines(self):
        if self.jobPath is None:
            self.logger.warning(
                'self.jobPath is None: self.jobLines is empty list')
            return []
        with open(self.jobPath, 'r') as f:
            jobLines = f.readlines()
        return jobLines

    @jobLines.setter
    def jobLines(self, lines):
        if self.jobPath is None:
            self.logger.warning(
                'self.jobPath is None: self.jobLines not written')
        elif lines is None:
            pass
        else:
            for i, line in enumerate(lines):
                if not line.endswith('\n'):
                    lines[i] += '\n'
            with open(self.jobPath, 'w+') as f:
                f.writelines(lines)

    ### Input Lines ###
    @property
    def inputLines(self):
        if self.inputPath is None:
            self.logger.warning(
                'self.inputPath is None: self.inputLines is empty list')
            return []
        with open(self.inputPath, 'r') as f:
            inputLines = f.readlines()
        return inputLines

    @inputLines.setter
    def inputLines(self, lines):
        if self.inputPath is None:
            self.logger.info(
                'self.inputPath is None: self.inputLines not written')
        elif lines is None:
            pass
        else:
            for i, line in enumerate(lines):
                if not line.endswith('\n'):
                    lines[i] += '\n'
            with open(self.inputPath, 'w+') as f:
                f.writelines(lines)
    # @jobLines.deleter
    # def jobLines(self):
    #     self.jobLines = None

    ### Data File Lines ###
    @property
    def datLines(self):
        with open(self.datPath, 'r') as f:
            lines = f.readlines()
        return lines

    ### Variables ###
    @property
    def x(self): return self._x

    @x.setter
    def x(self, x):
        x = self._getVar(x)
        x = np.array(x)
        if not x.shape:  # for single parameter studies
            x = np.array([x])
        self._x = x
        if os.path.exists(self.abs_path):
            saveTxt(self.abs_path, 'var.txt', x)
            # path = os.path.join(self.abs_path, 'var.txt')
            # np.savetxt(path, x)

    @property
    def g(self): return self._g

    @g.setter
    def g(self, g):
        # g = self._getObj(g)
        self._g = g
        if g is not None:
            g = np.array(g)
            if not g.shape:  # for single objective studies
                g = np.array([g])
            # path = os.path.join(self.abs_path, 'const.txt')
            # np.savetxt(path, g)
            saveTxt(self.abs_path, 'const.txt', g)
            if np.isnan(np.sum(g)):
                self.logger.warning(f'CONSTRAINT(S) CONTAINS NaN VALUE - {g}')
                for const_i, const in enumerate(g):
                    if np.isnan(const):
                        g[const_i] = np.inf
                        self.logger.warning(
                            f'\t Constraint {const_i}: {const} -> {np.inf}')
                        self._g = g
    ### Objectives ###
    @property
    def f(self): return self._f

    @f.setter
    def f(self, f):
        f = self._getObj(f)
        self._f = f
        if f is not None:
            f = np.array(f)
            if not f.shape:  # for single objective studies
                f = np.array([f])
            path = os.path.join(self.abs_path, 'obj.txt')
            np.savetxt(path, f)
            if np.isnan(np.sum(f)):
                self.logger.warning(f'OBJECTIVE CONTAINS NaN VALUE - {f}')
                for obj_i, obj in enumerate(f):
                    if np.isnan(obj):
                        f[obj_i] = np.inf
                        self.logger.warning(
                            f'\t Objective {obj_i}: {obj} -> {np.inf}')
                        self._f = f

    ################
    #    LOGGER    #
    ################
    # def getLogger(self):
    #     _, tail = os.path.split(self.abs_path)
    #     # Get Child Logger using hierarchical "dot" convention
    #     logger = logging.getLogger(__name__ + '.' + self.abs_path)
    #     logger.setLevel(config.CFD_CASE_LOGGER_LEVEL)
    #     # Filters
    #     # Filters added to logger do not propogate up logger hierarchy
    #     # Filters added to handlers do propogate
    #     # filt = DispNameFilter(self.abs_path)
    #     # logger.addFilter(filt)
    #     # File Handle
    #     logFile = os.path.join(self.abs_path, f'{tail}.log')
    #     fileHandler = logging.FileHandler(logFile)
    #     logger.addHandler(fileHandler)
    #     # Stream Handler
    #     # parent root logger takes care of stream display
    #     # Formatter
    #     formatter = MultiLineFormatter(
    #         '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s')
    #     fileHandler.setFormatter(formatter)
    #     # Initial Message
    #     logger.info('-' * 30)
    #     logger.info('LOGGER INITIALIZED')
    #     # Plot logger
    #     plot_logger = logging.getLogger(Scatter.__name__)
    #     plot_logger.setLevel(config.PLOT_LOGGER_LEVEL)
    #     return logger

    #######################
    #    CHECKPOINTING    #
    #######################
    # def save_self(self):
    #     cp_path = self.cp_path.replace('.npy', '')
    #     try:
    #         np.save(cp_path + '.temp.npy', self)
    #         if os.path.exists(cp_path + '.npy'):
    #             os.rename(cp_path + '.npy', cp_path + '.old.npy')
    #         os.rename(cp_path + '.temp.npy', cp_path + '.npy')
    #         if os.path.exists(cp_path + '.old.npy'):
    #             os.remove(cp_path + '.old.npy')
    #     except FileNotFoundError as err:
    #         self.logger.error(str(err))
    def _update_filter(self, loaded_self):
        cp = self.load_self()
        # print(cp._x)
        if np.array_equal(self._x, cp._x):
            if self.abs_path != cp.abs_path:
                self.logger.warning(
                    'CASE DIRECTORY CHANGED BETWEEN CHECKPOINTS')
                self.logger.debug(str(cp.rel_path) + ' -> ' + str(self.rel_path))
            if cp.cp_path != self.cp_path:
                self.logger.warning(f'{cp.cp_path} != {self.cp_path}')
            if cp.meshSF != self.meshSF:
                self.logger.warning(f'{cp.meshSF} != {self.meshSF}')
                self.logger.warning(
                    'Running genMesh() to reflect change in mesh size factor')
                self.numElem = None
                self.genMesh()
                # cp.meshSF = self.meshSF
                self.logger.info(
                    'Mesh size factor changed, mesh generated, self.f and self.numElem set to None')
                cp.f = None
            cp.abs_path = self.abs_path
            # cp.cp_path = self.cp_path
            # cp.mesh_studyDir = self.mesh_studyDir
            # cp.meshSFs = self.meshSFs
            cp.base_case_path = self.base_case_path
            return cp
        else:
            self.logger.info(f'Given Parameters: {self._x}')
            self.logger.info(f'Checkpoint Parameters: {cp._x}')
            question = f'\nCASE PARAMETERS DO NOT MATCH.\nOVERWRITE {self.rel_path}?'
            overwrite = yes_or_no(question)
            if overwrite:
                shutil.rmtree(self.abs_path)
                self.__init__(self.abs_path, self.x)
            else:
                self.logger.exception(
                    'GIVEN PARAMETERS DO NOT MATCH CHECKPOINT PARAMETERS.')

    def update_self(self):
        cp = self.load_self()
        # print(cp._x)
        if np.array_equal(self._x, cp._x):
            if self.abs_path != cp.abs_path:
                self.logger.warning(
                    'CASE DIRECTORY CHANGED BETWEEN CHECKPOINTS')
                self.logger.debug(str(cp.rel_path) + ' -> ' + str(self.rel_path))
            if cp.cp_path != self.cp_path:
                self.logger.warning(f'{cp.cp_path} != {self.cp_path}')
            if cp.meshSF != self.meshSF:
                self.logger.warning(f'{cp.meshSF} != {self.meshSF}')
                self.logger.warning(
                    'Running genMesh() to reflect change in mesh size factor')
                self.numElem = None
                self.genMesh()
                # cp.meshSF = self.meshSF
                self.logger.info(
                    'Mesh size factor changed, mesh generated, self.f and self.numElem set to None')
                cp.f = None
            cp.abs_path = self.abs_path
            # cp.cp_path = self.cp_path
            # cp.mesh_studyDir = self.mesh_studyDir
            # cp.meshSFs = self.meshSFs
            cp.base_case_path = self.base_case_path
            # cp.logger = self.getLogger()
            self.update_self(loaded_self=cp)
            self.logger.info(f'CHECKPOINT LOADED - {self.cp_rel_path}')
        else:
            self.logger.info(f'Given Parameters: {self._x}')
            self.logger.info(f'Checkpoint Parameters: {cp._x}')
            question = f'\nCASE PARAMETERS DO NOT MATCH.\nOVERWRITE {self.rel_path}?'
            overwrite = yes_or_no(question)
            if overwrite:
                shutil.rmtree(self.abs_path)
                self.__init__(self.abs_path, self.x)
            else:
                self.logger.exception(
                    'GIVEN PARAMETERS DO NOT MATCH CHECKPOINT PARAMETERS.')
        # del_keys = []
        # for cp_key in cp.__dict__:
        #     if cp_key in self.__dict__:
        #         print(cp_key, 'in self.__dict__')
        #         print(self.__dict__[cp_key])
        #         print(cp.__dict__[cp_key])
        #         print(f'Updating - {cp_key}: {cp.__dict__[cp_key]} -> {self.__dict__[cp_key]}')
        #         del_keys.append(cp_key)
        #         del cp.__dict__[cp_key]
        # for key in del_keys:
        #     del cp.__dict__[key]
        # print('self.__dict__', self.__dict__)
        # print()
        # print('cp.__dict__', cp.__dict__)
        # cp.__dict__.update(self.__dict__)
        # Directory Name Changes

    ########################
    #    HELPER METHODS    #
    ########################
    @staticmethod
    def tail(f, lines=1, _buffer=4098):
        """
        Tail a file and get X lines from the end
        AUTHOR: https://stackoverflow.com/users/1889809/glenbot
        """
        # place holder for the lines found
        lines_found = []
        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1
        # loop until we find X lines
        while len(lines_found) < lines:
            try:
                f.seek(block_counter * _buffer, os.SEEK_END)
            except IOError:  # either file is too small, or too many lines requested
                f.seek(0)
                lines_found = f.readlines()
                break
            lines_found = f.readlines()
            # decrement the block counter to get the
            # next X bytes
            block_counter -= 1
        lines_found = [line.decode() for line in lines_found]
        return lines_found[-lines:]

    def copy_base_case(self):
        shutil.copytree(self.base_case_path, self.abs_path, dirs_exist_ok=True)
        self.logger.info(f'COPIED FROM: {self.base_case_path}')

    # @staticmethod
    def findKeywordLines(self, kw, file_lines, exact=False, stripKW=True):
        kw_lines = []
        if stripKW:
            kw = kw.rstrip().lstrip()
        for line_i, line in enumerate(file_lines):
            if exact and kw == line:
                kw_lines.append([line_i, line])
            elif line.find(kw) >= 0:
                kw_lines.append([line_i, line])
        return kw_lines

    def findAndReplaceKeywordLines(self, file_lines, newLine, kws, insertIndex=0, replaceOnce=False, exact=False, stripKW=True):
        '''
        Finds and replaces any file_lines with newLine that match keywords (kws) give.
        If no keyword lines are found the newLine is inserted at the beginning of the file_lines.
        '''
        kw_lines_array = []
        for kw in kws:
            kw_lines_array.append(self.findKeywordLines(
                kw, file_lines, exact=exact, stripKW=stripKW))
        # print(kw_lines_array)
        if sum([len(kw_lines) for kw_lines in kw_lines_array]) > 0:
            def replace():
                for kw_lines in kw_lines_array:
                    for line_i, line in kw_lines:
                        file_lines[line_i] = newLine
                        if replaceOnce:
                            return
            replace()
        else:
            file_lines.insert(insertIndex, newLine)
        return file_lines

    def commentKeywordLines(self, kw, file_lines, marker='#', exact=False):
        kw_lines = self.findKeywordLines(kw, file_lines, exact=exact)
        for kw_line_i, kw_line in kw_lines:
            if kw_line[0] != marker:
                file_lines[kw_line_i] = marker + kw_line
        return file_lines

        # @run_once# import functools
    # def run_once(f):
    #     """Runs a function (successfully) only once.
    #     The running can be reset by setting the `has_run` attribute to False
    #     """
    #     @functools.wraps(f)
    #     def wrapper(*args, **kwargs):
    #         if not wrapper.complete:
    #             result = f(*args, **kwargs)
    #             wrapper.complete = True
    #             return result
    #     wrapper.complete = False
    #     return wrapper
    #
    # def calltracker(func):
    #     @functools.wraps(func)
    #     def wrapper(*args, **kwargs):
    #         result = func(*args, **kwargs)
    #         wrapper.complete = True
    #         return result
    #     wrapper.complete = False
    #     return wrapper

    ########################
    #    DUNDER METHODS    #
    ########################
    def __str__(self):
        s = f'Directory: {self.rel_path} | Parameters: {self.x}'
        if self._f is not None:
            s += f' | Objectives: {self._f}'
        return s

    # __repr__ = __str__

    # def __deepcopy__(self, memo):
    #     # shutil.copytree(self.base_case_path, self.abs_path, dirs_exist_ok=True)
    #     # print('COPIED:', self.base_case_path, '->', self.abs_path)
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for k, v in self.__dict__.items():
    #         setattr(result, k, copy.deepcopy(v, memo))
    #     return result

    # Calling destructor
    # def __del__(self):
    #     # self.save_self()
    #     # shutil.rmtree(caseDir)
    #     self.logger.info('EXITED')
    #     print('EXITED:', self.abs_path)

    # ==========================================================================
    # TO BE OVERWRITTEN
    # ==========================================================================
    def _preProc(self):
        pass

    def _preProc_restart(self):
        self._preProc()
        # pass

    # def _pySolve(self):
    #     pass

    def _solve(self):
        self.logger.error(
            'OVERRIDE _solve(self) method to execute internal python solver OR use CFDCase.externalSolver=True')

    def _execDone(self):
        if (self.f is not None and np.isfinite(np.sum(self.f))):
            return True

    def _postProc(self):
        pass

    def _genMesh(self):
        pass

    def _getObj(self, f):
        return f

    def _getVar(self, x):
        return x


class YALES2Case(CFDCase):
    # def __init__(self, caseDir, x, *args, **kwargs):
    #     super().__init__(caseDir, x, *args, **kwargs)
    # add random line to get to git push
    # def preProc(self):
    #     # ensure dump is in directory called 'dump'
    #     in_lines = self.inputLines
    #     if in_lines:
    #         kw_lines = self.findKeywordLines('DUMP_PREFIX', in_lines)
    #         for kw_line_i, kw_line in kw_lines:
    #             if kw_line[-1] != '#':
    #                 print(kw_line)
    #                 print(kw_line.split("'", 2))
    #                 start, mid, end = kw_line.split("'", 2)
    #                 _, tail = os.path.split(mid)
    #                 path = os.path.join('dump', tail)
    #                 in_lines[kw_line_i] = 'DUMP_PREFIX = ' + path
    #         self.inputLines = in_lines
    #     else:
    #         self.logger.error('YALES2 input file (.in) can not be read')
    #     super().preProc()
    def _execDone(self):
        # print('EXECUTION DONE?')
        searchPath = os.path.join(self.abs_path, 'solver01_rank*.log')
        resPaths = glob(searchPath)
        for fPath in resPaths:
            with open(fPath, 'rb') as f:
                try:  # catch OSError in case of a one line file
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
            if 'in destroy_mpi' in last_line:
                return True

    def _preProc_restart(self):
        pass
        # self._preProc()
        # XMF RESTARTS DO NOT WORK
        # read input lines
        # in_lines = self.inputLines
        # in_lines = self.commentKeywordLines('RESTART', in_lines)
        # # re-read lines
        # in_lines = self.inputLines
        # # delete all 'XMF' and "RESTART" lines
        # kw_lines = self.findKeywordLines('XMF', in_lines)
        # del_markers = [line_i for line_i, line in kw_lines
        #                if 'RESTART' in line]
        # in_lines = [line for line_i, line in enumerate(in_lines)
        #             if line_i in del_markers]
        # # append restart lines with lastest xmf file from dump directory
        # latestXMF = self.getLatestXMF()
        # in_lines.append('RESTART_TYPE = XMF')
        # path = os.path.join('dump', latestXMF)
        # in_lines.append('RESTART_XMF_SOLUTION = ' + path)
        # # write input lines
        # self.inputLines = in_lines

    def solve(self):
        super().solve()
        self.wallTime = self.getWallTime()
        self.save_self()

    def getWallTime(self):
        search_str = os.path.join(self.abs_path, 'solver01_rank*.log')
        fPaths = glob(search_str)
        for fPath in fPaths:
            with open(fPath, 'rb') as f:
                final_lines = self.tail(f, lines=10)
            for line in final_lines:
                if 'WALL CLOCK TIME' in line:
                    wall_time = int(float(line[-13:]))
                    self.logger.info(
                        f'YALES2 Wall Clock Time: {wall_time} seconds')
                    return wall_time

    def getLatestXMF(self):
        latestXMF = None
        ents = os.listdir(self.dumpDir)
        ents.sort()
        for ent in ents:
            if ent.endswith('.xmf') and not re.search('.sol.+_.+\\.xmf', ent):
                latestXMF = ent
        return latestXMF

    def getLatestMesh(self):
        latestMesh = None
        ents = os.listdir(self.dumpDir)
        ents.sort()
        for ent in ents:
            if ent.endswith('.mesh.h5'):
                latestMesh = ent
        return latestMesh

    def getLatestSoln(self):
        latestSoln = None
        ents = os.listdir(self.dumpDir)
        ents.sort()
        for ent in ents:
            if ent.endswith('.sol.h5'):
                latestSoln = ent
        return latestSoln

    def getLatestDataFiles(self):
        latestMesh = self.getLatestMesh()
        latestSoln = self.getLatestSoln()
        return latestMesh, latestSoln

    # def setRestart(self):
    #     # latestMesh, latestSoln = self.getLatestDataFiles()
    #     latestMesh = self.getLatestMesh()
    #     latestSoln = self.getLatestSoln()
    #     # with open(self.inputPath, 'r') as f:
    #     #     in_lines = f.readlines()
    #     in_lines = self.inputLines
    #     kw = 'RESTART_TYPE = GMSH'
    #     kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    #     in_lines[kw_line_i] = '#' + kw
    #     kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
    #     kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    #     in_lines[kw_line_i] = '#' + kw
    #     kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
    #     kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    #     in_lines[kw_line_i] = '#' + kw
    #     in_lines.append('RESTART_TYPE = XMF')
    #     in_lines.append('RESTART_XMF_SOLUTION = dump/' + latestXMF)
    #     # with open(self.inputPath, 'w') as f:
    #     #     f.writelines(in_lines)
    #     self.inputLines = in_lines

    @property
    def dumpDir(self):
        return os.path.join(self.abs_path, 'dump')


class FluentCase(CFDCase):
    pass


# class YALES2Case(CFDCase):
#     # def __init__(self, caseDir, x, *args, **kwargs):
#     #     super().__init__(caseDir, x, *args, **kwargs)
#
#     def solve(self):
#         super().solve()
#         self.wallTime = self.getWallTime()
#
#     def getWallTime(self):
#         search_str = os.path.join(self.abs_path, 'solver01_rank*.log')
#         fPaths = glob(search_str)
#         for fPath in fPaths:
#             with open(fPath, 'rb') as f:
#                 try:  # catch OSError in case of a one line file
#                     f.seek(-1020, os.SEEK_END)
#                 except OSError:
#                     f.seek(0)
#                 clock_line = f.readline().decode()
#             if 'WALL CLOCK TIME' in clock_line:
#                 wall_time = int(float(clock_line[-13:]))
#                 self.logger.info(f'YALES2 Wall Clock Time: {wall_time} seconds')
#             else:
#                 self.logger.warning('no wall clock time found')
#                 wall_time = None
#             return wall_time
#
#     def getLatestXMF(self):
#         ents = os.listdir(self.dumpDir)
#         ents.sort()
#         for ent in ents:
#             if ent.endswith('.xmf') and not re.search('.sol.+_.+\\.xmf', ent):
#                 latestXMF = ent
#         return latestXMF
#
#     def getLatestMesh(self):
#         ents = os.listdir(self.dumpDir)
#         ents.sort()
#         for ent in ents:
#             if ent.endswith('.mesh.h5'):
#                 latestMesh = ent
#         return latestMesh
#
#     def getLatestSoln(self):
#         ents = os.listdir(self.dumpDir)
#         ents.sort()
#         for ent in ents:
#             if ent.endswith('.sol.h5'):
#                 latestSoln = ent
#         return latestSoln
#
#     def getLatestDataFiles(self):
#         latestMesh = self.getLatestMesh()
#         latestSoln = self.getLatestSoln()
#         return latestMesh, latestSoln

    # def setRestart(self):
    #     # latestMesh, latestSoln = self.getLatestDataFiles()
    #     latestMesh = self.getLatestMesh()
    #     latestSoln = self.getLatestSoln()
    #     # with open(self.inputPath, 'r') as f:
    #     #     in_lines = f.readlines()
    #     in_lines = self.inputLines
    #     kw = 'RESTART_TYPE = GMSH'
    #     kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    #     in_lines[kw_line_i] = '#' + kw
    #     kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
    #     kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    #     in_lines[kw_line_i] = '#' + kw
    #     kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
    #     kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    #     in_lines[kw_line_i] = '#' + kw
    #     in_lines.append('RESTART_TYPE = XMF')
    #     in_lines.append('RESTART_XMF_SOLUTION = dump/' + latestXMF)
    #     # with open(self.inputPath, 'w') as f:
    #     #     f.writelines(in_lines)
    #     self.inputLines = in_lines
#
#     @property
#     def dumpDir(self):
#         return os.path.join(self.abs_path, 'dump')
# ###################
#    FUNCTIONS    #
###################
# def saveTxt(path, fname, data):
#     datFile = os.path.join(path, fname)
#     # save data as text file in directory
#     np.savetxt(datFile, data)

# from http://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
# import functools
# def run_once(f):
#     """Runs a function (successfully) only once.
#     The running can be reset by setting the `has_run` attribute to False
#     """
#     @functools.wraps(f)
#     def wrapper(*args, **kwargs):
#         if not wrapper.complete:
#             result = f(*args, **kwargs)
#             wrapper.complete = True
#             return result
#     wrapper.complete = False
#     return wrapper
#
# def calltracker(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         wrapper.complete = True
#         return result
#     wrapper.complete = False
#     return wrapper
