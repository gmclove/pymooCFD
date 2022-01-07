# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:33:00-05:00
import numpy as np
import os
import subprocess
import shutil
import logging
import copy
import matplotlib.pyplot as plt
import multiprocessing as mp
import multiprocessing.pool


class CFDCase:  # (PreProcCase, PostProcCase)
    ####### Define Design Space #########
    n_var = None
    var_labels = None
    varType = None  # OPTIONS: 'int' or 'real'
    xl = None  # lower limits of parameters/variables
    xu = None  # upper limits of variables
    # if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
    #     raise ExceptionDesign Space Definition Incorrect")
    ####### Define Objective Space ########
    obj_labels = None
    n_obj = None
    ####### Define Constraints #########
    n_constr = None

    ##### External Solver #######
    externalSolver = False
    procLim = None
    nProc = None
    solverExecCmd = None
    # if procLim is None:
    #     nTasks = 1000000
    # else:
    #     nTasks = int(procLim/nProc)
    # if externalSolver:
    #     assert solverExecCmd is not None
    #     assert nProc is not None
    #     assert procLim is not None
    #     solve = solveExternal
    #     pool = mp.pool.ThreadPool(nTasks)
    # else:
    #     solve = _solve
    #     pool = Pool(nTasks)

    def __init__(self, baseCaseDir, caseDir, x,
                 meshSF=1, meshSFs=np.arange(0.5, 1.5, 0.1),
                 # externalSolver=False,
                 # var_labels = None, obj_labels = None,
                 meshFile=None,  # meshLines = None,
                 jobFile=None,  # jobLines = None,
                 inputFile=None,  # inputLines = None,
                 datFile=None,
                 # restart=False,
                 # solverExecCmd=None,
                 *args, **kwargs
                 ):
        super().__init__()
        self.complete = False
        self.restart = False
        self.parallelizeInit(self.externalSolver)
        self.baseCaseDir = baseCaseDir
        self.caseDir = caseDir
        self.cpPath = os.path.join(caseDir, 'case')
        if os.path.exists(caseDir):
            try:
                self.loadCP()
                self.logger.info('RESTART CASE')
                return
            except FileNotFoundError:
                self.logger = self.getLogger()
                self.logger.info(
                    f'OVERRIDE CASE - {caseDir} already exists but {self.cpPath} does not')
                self.copy()
        else:
            os.makedirs(caseDir, exist_ok=True)
            self.logger = self.getLogger()
            self.logger.info(f'NEW CASE - {caseDir} did not exist')
            self.copy()
        # If solverExecCmd is provided use
        # if self.solverExecCmd is None:
        #     externalSolver = False
        # elif self.nProc is not None and self.procLim is not None:
        #     externalSolver = True
        # self.externalSolver = externalSolver

        # if not os.path.exists(caseDir):
        #     os.makedirs(caseDir)
        #     msg = f'NEW CASE - {caseDir} did not exist'
        #     print(msg)
        # Required Arguments -> Attributes
        # self.logger = self.getLogger()
        # self.copy()  # create directory before start
        self.x = x
        # Default Attributes
        # os.makedirs(self.baseCaseDir, exist_ok=True)
        self.meshSF = meshSF
        self.meshSFs = meshSFs
        # Optional Attributes
        self.meshFile = meshFile
        self.jobFile = jobFile
        self.datFile = datFile
        self.inputFile = inputFile
        # generate file path attributes when possible
        self.meshStudyDir = os.path.join(self.caseDir, 'meshStudy')

        if inputFile is None:
            self.inputPath = None
        else:
            self.inputPath = os.path.join(self.caseDir, inputFile)

        if datFile is None:
            self.datPath = None
        else:
            self.datPath = os.path.join(self.caseDir, datFile)

        if meshFile is None:
            self.meshPath = None
        else:
            self.meshPath = os.path.join(self.caseDir, meshFile)

        if jobFile is None:
            self.jobPath = None
        else:
            self.jobPath = os.path.join(self.caseDir, jobFile)
        ###################################################
        #    Attributes To Be Set Later During Each Run   #
        ###################################################
        self.f = None  # if not None then case complete
        self.msCases = None
        self.numElem = None
        # self.var_labels = None
        # self.obj_labels = None
        self.n_var = None
        self.n_obj = None
        # self.jobLines = None #???????????
        self.preProcComplete = False
        self.postProcComplete = False
        self.execComplete = False

        ### Save Checkpoint ###
        # _, tail = os.path.split(caseDir)
        # self.cpPath = os.path.join(caseDir, tail+'.npy')
        self.saveCP()

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
    #     proc = subprocess.Popen(cmd, cwd=self.caseDir,
    #                             stdout=subprocess.DEVNULL)
    #     return proc
    ###  Parallel Processing  ###
    @classmethod
    def parallelizeInit(cls, externalSolver):
        if cls.procLim is None:
            nTasks = 1000000
        else:
            nTasks = int(cls.procLim / cls.nProc)
        if externalSolver:
            assert cls.solverExecCmd is not None
            assert cls.nProc is not None
            assert cls.procLim is not None
            cls.solve = cls.solveExternal
            cls.pool = mp.pool.ThreadPool(nTasks)
        else:
            cls.solve = cls._solve
            cls.pool = cls.Pool(nTasks)

    @classmethod
    def parallelize(cls, cases):
        for case in cases:
            cls.pool.apply_async(case.run, ())
        cls.pool.close()
        cls.pool.join()

    # def parallelizeCleanUp(self):
    #     self.pool.terminate()

    def solveExternal(self):
        self.logger.info('SOLVING AS SUBPROCESS...')
        self.logger.info(f'\tcommand: {self.solverExecCmd}')
        subprocess.run(self.solverExecCmd, cwd=self.caseDir,
                       stdout=subprocess.DEVNULL)

    def run(self):
        if not self.complete:
            self.preProc()
            self.logger.info('COMPLETED: PRE-PROCESS')
            self.solve()
            if self._execDone():
                self.logger.info('COMPLETED: SOLVE')
            else:
                self.logger.warning('RUN FAILED TO EXECUTE')
                self.logger.info('RE-RUNNING')
                self.run()
            self.postProc()
            self.logger.info('COMPLETED: POST-PROCESS')
            self.complete = True
        else:
            self.logger.info('self.run() called but case already complete')

    # def run(case):
    #     case.preProc()
    #     case.logger.info('COMPLETED: PRE-PROCESS')
    #     case.solve()
    #     case.logger.info('COMPLETED: SOLVE')
    #     case.postProc()
    #     case.logger.info('COMPLETED: POST-PROCESS')

    # def execCallback(self):
    #     if self._execDone():
    #         self.logger.info('RUN COMPLETE')
    #     else:
    #         self.pool.apply_async(self.solve, (self.caseDir,))

    # def solve(self):
    #     # if self.f is None: # and not self.restart:
    #     self.restart = True
    #     if self.solverExecCmd is None:
    #         self.logger.error('No external solver execution command give. \
    #                             Please override solve() method with python CFD \
    #                             solver or add solverExecCmd to CFDCase object.')
    #         raise Exception('No external solver execution command give. Please \
    #                         override solve() method with python CFD solver or \
    #                         add solverExecCmd to CFDCase object.')
    #     else:
    #         proc = subprocess.Popen(self.solverExecCmd, cwd=self.caseDir,
    #                                 stdout=subprocess.DEVNULL)
    #         return proc
        # else:
        #     self.logger.warning('SKIPPED SOLVE() METHOD')

    def preProc(self):
        if self.restart:
            # self.cpPath = os.path.join
            self.logger.info(
                'PRE-PROCESS RESTART - Using self._preProc_restart()')
            self._preProc_restart()
        else:
            self._preProc()
        # save variables in case directory as text file after completing pre-processing
        # saveTxt(self.caseDir, 'var.txt', self.x)
        self.restart = True  # ??????????????????
        self.saveCP()
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
        #     subprocess.run(self.solverExecCmd, cwd=self.caseDir,
        #                    stdout=subprocess.DEVNULL)

    def postProc(self):
        if self.f is None:
            self._postProc()
            ###### SAVE VARIABLES AND OBJECTIVES TO TEXT FILES #######
            # save variables in case directory as text file after completing post-processing
            # saveTxt(self.caseDir, 'var.txt', self.x)
            # save objectives in text file
            # saveTxt(self.caseDir, 'obj.txt', self.f)
        else:
            self.logger.info('self.postProc() called but self.f is not None')
        self.saveCP()
        self.logger.info(f'\t{self.caseDir}: {self.f}')
        return self.f

    def genMesh(self):
        self._genMesh()

    ####################
    #    MESH STUDY    #
    ####################
    def genMeshStudy(self):
        print(f'GENERATING MESH STUDY: {self}')
        print(f'\t{self.meshSFs}')
        # Pre-Process
        study = []
        var = []
        a_numElem = []
        self.msCases = []
        for sf in self.meshSFs:
            # Deep copy case instance
            # msCase = copy.deepcopy(self)
            # self.msCases.append(msCase)
            # msCase.logger = msCase.getLogger()
            # msCase.restart = False
            # msCase.meshSFs = None
            # msCase.meshStudyDir = None
            # msCase.caseDir = os.path.join(self.meshStudyDir, f'meshSF-{sf}')
            # msCase.meshSF = sf
            # msCase.copy()
            msCase = copy.deepcopy(self)
            self.msCases.append(msCase)
            path = os.path.join(self.meshStudyDir, f'meshSF-{sf}')
            msCase.__init__(self.baseCaseDir, path, self.x)
            msCase.meshSFs = None
            msCase.meshSF = sf
            # only pre-processing needed is generating mesh
            msCase.genMesh()
            a_numElem.append(msCase.numElem)
            path = os.path.join(msCase.caseDir, 'numElem.txt')
            np.savetxt(path, [msCase.numElem])
            study.append([msCase.caseDir, str(msCase.numElem)])
            var.append(msCase.x)
        print(f'\t{a_numElem}')
        # study = np.array(study)
        print(study)
        print('\t' + str(study).replace('\n', '\n\t'))
        path = os.path.join(self.meshStudyDir, 'study.txt')
        np.savetxt(path, study, fmt="%s")
        path = os.path.join(self.meshStudyDir, 'studyX.txt')
        np.savetxt(path, var)
        obj = np.array([case.f for case in self.msCases])
        path = os.path.join(self.meshStudyDir, 'studyF.txt')
        np.savetxt(path, obj)

    def plotMeshStudy(self):
        _, tail = os.path.split(self.caseDir)
        a_numElem = np.array([case.numElem for case in self.msCases])
        msObj = np.array([case.f for case in self.msCases])
        print(msObj)
        print(a_numElem)
        # Plot
        for obj_i, obj_label in enumerate(self.obj_labels):
            print(f'\tPLOTTING OBJECTIVE {obj_i}: {obj_label}')
            plt.plot(a_numElem, msObj[:, obj_i])
            plt.suptitle('Mesh Sensitivity Study')
            plt.title(tail)
            plt.xlabel('Number of Elements')
            plt.ylabel(obj_label)
            fName = f'ms_plot-{tail}-obj{obj_i}.png'
            fPath = os.path.join(self.meshStudyDir, fName)
            plt.savefig(fPath)
            plt.clf()

    def execMeshStudy(self):
        self.parallelize(self.msCases)
        # nTask = int(self.procLim/self.BaseCase.nProc)
        # pool = mp.Pool(nTask)
        # for case in self.msCases:
        #     pool.apply_async(case.run, ())
        # pool.close()
        # pool.join()

    def meshStudy(self, restart=True):  # , meshSFs=None):
        # if meshSFs is None:
        #     meshSFs = self.meshSFs
        # if self.msCases is None:
        #     self.genMeshStudy()
        if not restart or self.msCases is None:
            self.genMeshStudy()
        self.execMeshStudy()
        self.plotMeshStudy()

    # @calltracker
    # def postProc(self):
    #     if postProc.complete:
    #         path = os.path.join(sel# if self.solverExecCmd is None:
        #     self.logger.error('No external solver execution command give. \
        #                        Please override solve() method with python CFD \
        #                        solver or add solverExecCmd to CFDCase object.')
        #     raise Exception('No external solver execution command give. Please \
        #                     override solve() method with python CFD solver or \
        #                     add solverExecCmd to CFDCase object.')
        # else:
        #     subprocess.run(self.solverExecCmd, cwd=self.caseDir,
        #                    stdout=subprocess.DEVNULL)f.caseDir, 'obj.txt')
    #         obj = np.loadtxt(path)
    #     else:
    #         obj = self._postProc(self)
    #         self.completed = True
    #         self._f = obj
    #         ###### SAVE VARIABLES AND OBJECTIVES TO TEXT FILES #######
    #         # save variables in case directory as text file after completing post-processing
    #         saveTxt(self.caseDir, 'var.txt', self.x)
    #         # save objectives in text file
    #         saveTxt(self.caseDir, 'obj.txt', obj)
    #     return obj

    ##########################
    #    CLASS PROPERTIES    #
    ##########################
    # @property
    # def caseDir(self):
    #     return self.caseDir
    # @caseDir.setter
    # def caseDir(self, caseDir):
    #     os.makedirs(caseDir, exist_ok)
    #     self.
    #     self.caseDir = caseDir
    ### Job Lines ###
    @property
    def jobLines(self):
        with open(self.jobPath, 'r') as f:
            jobLines = f
        return jobLines

    @jobLines.setter
    def jobLines(self, lines):
        if self.jobPath is None:
            self.logger.info('self.jobPath is None: self.jobLines not written')
        else:
            for i, line in enumerate(lines):
                if not line.endswith('/n'):
                    lines[i] += '/n'
            with open(self.jobPath, 'w+') as f:
                f.writelines(lines)

    ### Input Lines ###
    @property
    def inputLines(self):
        with open(self.inputPath, 'r') as f:
            inputLines = f.readlines()
        return inputLines

    @inputLines.setter
    def inputLines(self, lines):
        if self.inputPath is None:
            self.logger.info('self.inputPath is None: Input lines not written')
        else:
            for i, line in enumerate(lines):
                if not line.endswith('/n'):
                    lines[i] += '/n'
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
        x = np.array(x)
        path = os.path.join(self.caseDir, 'var.txt')
        np.savetxt(path, x)
        self._x = x

    ### Objectives ###
    @property
    def f(self): return self._f

    @f.setter
    def f(self, f):
        if f is not None:
            f = np.array(f)
            path = os.path.join(self.caseDir, 'obj.txt')
            np.savetxt(path, f)
        self._f = f

    # @property
    # def msCases(self): return self._msCases
    # @msCases.setter
    # def msCases(self, cases):
    #     if cases is not None:
    #         path = os.path.join(self.caseDir, 'msCases.npy')

    ################
    #    LOGGER    #
    ################
    def getLogger(self):
        _, tail = os.path.split(self.caseDir)
        logFile = os.path.join(self.caseDir, f'{tail}.log')
        logger = logging.getLogger(logFile)
        logger.setLevel(logging.INFO)
        # define file handler and set formatter
        file_handler = logging.FileHandler(logFile)
        formatter = logging.Formatter(
            '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    #######################
    #    CHECKPOINTING    #
    #######################
    def saveCP(self):
        np.save(self.cpPath + '.temp.npy', self)
        if os.path.exists(self.cpPath + '.npy'):
            os.rename(self.cpPath + '.npy', self.cpPath + '.old.npy')
        os.rename(self.cpPath + '.temp.npy', self.cpPath + '.npy')
        if os.path.exists(self.cpPath + '.old.npy'):
            os.remove(self.cpPath + '.old.npy')

    def loadCP(self):
        if os.path.exists(self.cpPath + '.old'):
            os.rename(self.cpPath + '.old', self.cpPath + '.npy')
        cp, = np.load(self.cpPath + '.npy', allow_pickle=True).flatten()
        self.__dict__.update(cp.__dict__)
        self.logger.info(f'\tCHECKPOINT LOADED - from {self.cpPath}.npy')

    ########################
    #    HELPER METHODS    #
    ########################
    def copy(self):
        # if os.path.exists(self.caseDir):
        #     self.logger.warning('CASE OVERRIDE - self.caseDir already existed')
        shutil.copytree(self.baseCaseDir, self.caseDir, dirs_exist_ok=True)
        self.logger.info(f'COPIED: {self.baseCaseDir} -> {self.caseDir}')

    def findKeywordLine(self, kw, file_lines):
        kw_line = -1
        kw_line_i = -1
        for line_i in range(len(file_lines)):
            line = file_lines[line_i]
            if line.find(kw) >= 0:
                kw_line = line
                kw_line_i = line_i
        return kw_line, kw_line_i

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
        # with np.printoptions(suppress=True):
        return f'Directory: {self.caseDir} | Parameters: {self.x}'
    # __repr__ = __str__

    # def __deepcopy__(self, memo):
    #     # shutil.copytree(self.baseCaseDir, self.caseDir, dirs_exist_ok=True)
    #     # print('COPIED:', self.baseCaseDir, '->', self.caseDir)
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for k, v in self.__dict__.items():
    #         setattr(result, k, copy.deepcopy(v, memo))
    #     return result

    # Calling destructor
    # def __del__(self):
    #     # self.saveCP()
    #     # shutil.rmtree(caseDir)
    #     self.logger.info('EXITED')
    #     print('EXITED:', self.caseDir)

    # ==========================================================================
    # TO BE OVERWRITTEN
    # ==========================================================================
    def _preProc(self):
        pass

    def _preProc_restart(self):
        self._preProc()
        pass

    # def _pySolve(self):
    #     pass

    def _solve(self):
        print('OVERRIDE _solve(self) method to execute internal python solver OR use externalSolver=True')

    def _execDone(self):
        return True

    def _postProc(self):
        pass

    def _genMesh(self):
        pass


###################
#    FUNCTIONS    #
###################
def saveTxt(path, fname, data):
    datFile = os.path.join(path, fname)
    # save data as text file in directory
    np.savetxt(datFile, data)

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
