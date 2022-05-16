# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:33:00-05:00
# from pymooCFD.util.handleData import findKeywordLine
from pymoo.visualization.scatter import Scatter
import re
from pymooCFD.core.meshStudy import MeshStudy
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
    ###########################
    #    pymoo Constraints    #
    ###########################
    n_constr = None
    ######################
    #    pymoo Repair    #
    ######################
    repair = None

    ##### External Solver #######
    externalSolver = False
    onlyParallelizeSolve = False
    # parallelize_solve = True
    parallelize_preProc = True
    parallelize_postProc = True
    procLim = None
    nProc = None
    nTasks = None
    solverExecCmd = None

    def __init__(self, case_path, x,
                 validated=False,
                 mesh_study=None,
                 # externalSolver=False,
                 # var_labels=None, obj_labels=None,
                 meshFile=None,  # meshLines = None,
                 jobFile=None,  # job_lines_rw = None,
                 inputFile=None,  # input_lines_rw = None,
                 datFile=None,
                 # restart=False,
                 # solverExecCmd=None,
                 # *args,
                 **kwargs
                 ):
        if len(x) != self.n_var:
            raise Exception(f'input x must be of length {self.n_var}')
        self.meshSF = kwargs.get('meshSF', 1)
        self.validated = validated
        ###########################
        #    RESTART VARIABLES    #
        ###########################
        # self.complete = False
        # self.externalSolver = self.externalSolver
        self.setup_parallelize(self.externalSolver)
        self._x = x

        #####################################
        #    CHECKPOINT/PICKLE PATH INIT    #
        #####################################
        super().__init__(dir_path=case_path)
        if self.cp_init:
            return
        if self.base_case_path is None:
            self.logger.debug(
                'SKIPPED: cls.base_case_path is None - base case not copied')
        else:
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
        # self.setup_parallelize(self.externalSolver)
        self.restart = False
        self.validated = validated
        # Default Attributes
        if mesh_study is None:
            self.mesh_study = MeshStudy(self)
        # os.makedirs(self.basecase_path, exist_ok=True)
        # Using kwargs (not an option with labels as class variables)
        # self.var_labels = kwargs.get('var_labels')
        # self.obj_labels = kwargs.get('obj_labels')
        # Design and Objective Space Labels
        if self.var_labels is None:
            self.var_labels = [f'var{x_i}' for x_i in range(self.n_var)]
        if self.obj_labels is None:
            self.obj_labels = [f'obj{x_i}' for x_i in range(self.n_obj)]
        self.var_labels = list(self.var_labels)
        self.obj_labels = list(self.obj_labels)
        ###################################################
        #    Attributes To Be Set Later During Each Run   #
        ###################################################
        self.f = None  # if not None then case complete
        self.g = None
        # meshing attributes
        self.msCases = None
        self.numElem = None

        # case file lines
        self.input_lines_rw = None
        self.job_lines_rw = None

        self.solnTime = None

        self.logger.info('CASE INTITIALIZED')
        self.logger.debug('INITIAL CASE DICTONARY')
        for key, val in self.__dict__.items():
            self.logger.debug(f'\t{key}: {val}')
        # self.genMesh()
        ### Save Checkpoint ###
        self.save_self()

    ###  Parallel Processing  ###
    @classmethod
    def setup_parallelize(cls, externalSolver):
        if cls.nTasks is None:
            if cls.nProc is not None and cls.nProc is not None:
                cls.nTasks = int(cls.procLim / cls.nProc)
            else:
                cls.nTasks = config.MP_POOL_NTASKS_MAX
        if externalSolver:
            assert cls.solverExecCmd is not None
            assert cls.nTasks is not None
            cls._solve = cls.solveExternal
            cls.Pool = mp.pool.ThreadPool
            print('Setup thread pool: ', end='')
        else:
            cls._solve = cls._solve
            cls.Pool = mp.Pool
            print('Setup multiprocessing pool: ', end='')
        print('number of tasks =', cls.nTasks)
    #     if cls.parallelize_preProc and cls.parallelize_postProc:
    #         def func(self):
    #             self.preProc()
    #             self.solve()
    #             return self.postProc()
    #     elif not cls.parallelize_preProc and not cls.parallelize_postProc:
    #         def func(self):
    #             self.solve()
    #     elif cls.parallelize_preProc and not cls.parallelize_postProc:
    #         def func(self):
    #             self.preProc()
    #             self.solve()
    #     else:
    #         def func(self):
    #             self.solve()
    #             return self.postProc()
    #     cls.exec_parallel = func
    #
    # def exec_parallel(self):
    #     self.solve()

    @classmethod
    def parallelize(cls, cases, externalSolver=None):
        if externalSolver is None:
            externalSolver = cls.externalSolver
        cls.setup_parallelize(externalSolver)
        #cls.logger.info('PARALLELIZING . . .')
        with cls.Pool(cls.nTasks) as pool:
            if cls.onlyParallelizeSolve:
                for case in cases:
                    case.preProc()
                print('PARALLELIZING . . .')
                for case in cases:
                    pool.apply_async(case.solve, ())
                pool.close()
                pool.join()
                for case in cases:
                    case.postProc()
            else:
                print('PARALLELIZING . . .')
                for case in cases:
                    pool.apply_async(case.run, ())
                pool.close()
                pool.join()

    def solve(self):
        if not hasattr(self, 'Pool'):
            self.setup_parallelize(self.externalSolver)
        if self.f is None or not np.isfinite(np.sum(self.f)):
            # try to prevent re-run if execution done
            if self._solveDone() and self.restart:
                self.logger.debug(
                    'self.solve() called but self._solveDone() and self.restart is True')
                self.logger.warning('TRYING: POST-PROCESS BEFORE SOLVE')
                try:
                    self.postProc()
                except Exception as err:  # FileNotFoundError, TypeError
                    self.logger.error(err)
                    self.f = None
        if self.f is None or not np.isfinite(np.sum(self.f)):
            self.restart = True
            self.save_self()
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

            if self._solveDone():
                self.logger.info('COMPLETE: SOLVE')
                self.solnTime = dt
            else:
                self.logger.info('FAILED: SOLVE')
            self.logger.info(f'\tSolve Time: {t_str}')
            if int(dt) < 1:
                self.logger.warning('Solve executed in less than 1 second')
        else:
            self.logger.warning('SKIPPED: SOLVE')
        self.save_self()

    def solveExternal(self):
        self.logger.info('SOLVING AS SUBPROCESS...')
        self.logger.info(f'\tcommand: {self.solverExecCmd}')
        subprocess.run(self.solverExecCmd, cwd=self.abs_path,
                       stdout=subprocess.DEVNULL)

    def run(self, max_reruns=3, n_reruns=0):
        # print('RUNNING')
        if self.f is None or not np.isfinite(np.sum(self.f)):
            self.preProc()
            self.solve()
            if not self._solveDone():
                if n_reruns < max_reruns:
                    self.logger.warning('FAILED: SOLVE')
                    self.logger.info('RE-RUNNING')
                    n_reruns += 1
                    self.run(max_reruns=max_reruns, n_reruns=n_reruns)
                else:
                    self.logger.warning(
                        f'MAX NUMBER OF RE-RUNS ({max_reruns}) REACHED')
            self.postProc()
        else:
            self.logger.warning(
                'SKIPPED: RUN - self.run() called but case already complete')

    def preProc(self):
        if self.f is None or not np.isfinite(np.sum(self.f)):
            if self.restart:
                self.logger.info(
                    'PRE-PROCESS RESTART - Using self._preProc_restart()')
                self._preProc_restart()
            else:
                self._preProc()
            self.logger.info('COMPLETE: PRE-PROCESS')
        else:
            self.logger.warning('SKIPPED: PRE-PROCESS')
        self.save_self()

    def postProc(self):
        if self.f is None or not np.isfinite(np.sum(self.f)):
            self._postProc()
        else:
            self.logger.warning('SKIPPED: POST-PROCESSING')
            self.logger.debug(
                'self.postProc() called but self.f is not None or NaN so no action was taken')
        # Check Completion
        if self.f is None or not np.isfinite(np.sum(self.f)):
            self.logger.error('INCOMPLETE: POST-PROCESS')
        else:
            self.logger.info('COMPLETE: POST-PROCESS')
        self.logger.info(f'\tParameters: {self.x}')
        self.logger.info(f'\tObjectives: {self.f}')
        self.save_self()
        return self.f

    def genMesh(self):
        if self.meshPath is None:
            self.logger.warning(
                'self.genMesh() called but self.meshPath is None')
        if self.meshSF is None:
            self.logger.warning(
                'self.genMesh() called but self.meshPath is None')
        self.logger.info('GENERATING MESH . . .')
        start = time.time()
        self._genMesh()
        end = time.time()
        dt = end - start
        self.logger.info('MESH GENERATED - Using self._genMesh()')
        self.logger.info(f'\tMesh Generation Time: {int(dt)} seconds')

    ##########################
    #    CLASS PROPERTIES    #
    ##########################
    ### Case File Paths ###
    @property
    def plots_path(self):
        path = os.path.join(self.abs_path, 'plots')
        os.makedirs(path, exist_ok=True)
        return path

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

    # @property
    # def mesh_studyDir(self):
    #     return os.path.join(self.abs_path, 'mesh_study')

    ### Job Lines ###
    @property
    def job_lines_rw(self):
        if self.jobPath is None:
            self.logger.warning(
                'self.jobPath is None: self.job_lines_rw is empty list')
            return []
        with open(self.jobPath, 'r') as f:
            job_lines_rw = f.readlines()
        return job_lines_rw

    @job_lines_rw.setter
    def job_lines_rw(self, lines):
        if self.jobPath is None:
            self.logger.warning(
                'self.jobPath is None: self.job_lines_rw not written')
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
    def input_lines_rw(self):
        if self.inputPath is None:
            self.logger.warning(
                'self.inputPath is None: self.input_lines_rw is empty list')
            return []
        with open(self.inputPath, 'r') as f:
            input_lines_rw = f.readlines()
        return input_lines_rw

    @input_lines_rw.setter
    def input_lines_rw(self, lines):
        if self.inputPath is None:
            self.logger.info(
                'self.inputPath is None: self.input_lines_rw not written')
        elif lines is None:
            pass
        else:
            for i, line in enumerate(lines):
                if not line.endswith('\n'):
                    lines[i] += '\n'
            with open(self.inputPath, 'w+') as f:
                f.writelines(lines)
    # @job_lines_rw.deleter
    # def job_lines_rw(self):
    #     self.job_lines_rw = None

    ### Data File Lines ###
    @property
    def dat_lines_r(self):
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
            saveTxt(self.abs_path, 'const.txt', g)
            if np.isnan(np.sum(g)):
                self.logger.warning(f'CONSTRAINT(S) CONTAINS NaN VALUE - {g}')
                for const_i, const in enumerate(g):
                    if np.isnan(const):
                        g[const_i] = np.NINF
                        self.logger.warning(
                            f'\t Constraint {const_i}: {const} -> {np.NINF}')
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
            self._f = f
            # check for None or nan values, change those values to infinity
            if None in f:
                self.logger.warning(f'OBJECTIVE CONTAINS None VALUE - {f}')
                for obj_i, obj in enumerate(f):
                    if obj is None:
                        f[obj_i] = np.inf
                        self.logger.warning(
                            f'\t Objective {obj_i}: {obj} -> {np.inf}')
                        self._f = f
            if np.isnan(np.sum(f)):
                self.logger.warning(f'OBJECTIVE CONTAINS NaN VALUE - {f}')
                for obj_i, obj in enumerate(f):
                    if np.isnan(obj):
                        f[obj_i] = np.inf
                        self.logger.warning(
                            f'\t Objective {obj_i}: {obj} -> {np.inf}')
                        self._f = f

    def _update_filter(self, loaded_self):
        if np.array_equal(self._x, loaded_self._x):
            if loaded_self.meshSF != self.meshSF:
                self.logger.warning(f'{loaded_self.meshSF} != {self.meshSF}')
                self.logger.warning(
                    'Running genMesh() to reflect change in mesh size factor')
                self.numElem = None
                self.genMesh()
                # loaded_self.meshSF = self.meshSF
                self.logger.info(
                    'Mesh size factor changed, mesh generated, self.f and self.numElem set to None')
                loaded_self.f = None
            loaded_self.base_case_path = self.base_case_path
            return loaded_self
        else:
            self.logger.info(f'Given Parameters: {self._x}')
            self.logger.info(f'Checkpoint Parameters: {loaded_self._x}')
            question = f'\nCASE PARAMETERS DO NOT MATCH.\nOVERWRITE {self.rel_path}?'
            overwrite = yes_or_no(question)
            if overwrite:
                shutil.rmtree(self.abs_path)
                self.__init__(self.abs_path, self.x)
                return self
            else:
                self.logger.exception(
                    'GIVEN PARAMETERS DO NOT MATCH CHECKPOINT PARAMETERS.')

    # def update_self(self):
    #     cp = self.load_self()
    #     # print(cp._x)
    #     if np.array_equal(self._x, cp._x):
    #         if self.abs_path != cp.abs_path:
    #             self.logger.warning(
    #                 'CASE DIRECTORY CHANGED BETWEEN CHECKPOINTS')
    #             self.logger.debug(str(cp.rel_path) + ' -> ' + str(self.rel_path))
    #         if cp.cp_path != self.cp_path:
    #             self.logger.warning(f'{cp.cp_path} != {self.cp_path}')
    #         if cp.meshSF != self.meshSF:
    #             self.logger.warning(f'{cp.meshSF} != {self.meshSF}')
    #             self.logger.warning(
    #                 'Running genMesh() to reflect change in mesh size factor')
    #             self.numElem = None
    #             self.genMesh()
    #             # cp.meshSF = self.meshSF
    #             self.logger.info(
    #                 'Mesh size factor changed, mesh generated, self.f and self.numElem set to None')
    #             cp.f = None
    #         cp.abs_path = self.abs_path
    #         # cp.cp_path = self.cp_path
    #         # cp.mesh_studyDir = self.mesh_studyDir
    #         # cp.meshSFs = self.meshSFs
    #         cp.base_case_path = self.base_case_path
    #         # cp.logger = self.getLogger()
    #         self.update_self(loaded_self=cp)
    #         self.logger.info(f'CHECKPOINT LOADED - {self.cp_rel_path}')
    #     else:
    #         self.logger.info(f'Given Parameters: {self._x}')
    #         self.logger.info(f'Checkpoint Parameters: {cp._x}')
    #         question = f'\nCASE PARAMETERS DO NOT MATCH.\nOVERWRITE {self.rel_path}?'
    #         overwrite = yes_or_no(question)
    #         if overwrite:
    #             shutil.rmtree(self.abs_path)
    #             self.__init__(self.abs_path, self.x)
    #         else:
    #             self.logger.exception(
    #                 'GIVEN PARAMETERS DO NOT MATCH CHECKPOINT PARAMETERS.')
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
                    for line_i, _ in kw_lines:
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

    def _solveDone(self):
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
    def _solveDone(self):
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
    #     in_lines = self.input_lines_rw
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
    #     self.input_lines_rw = in_lines

    @property
    def dumpDir(self):
        return os.path.join(self.abs_path, 'dump')


class FluentCase(CFDCase):
    pass


class OpenFOAMCase(CFDCase):
    pass
