# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:33:00-05:00
import numpy as np
from glob import glob
import os
import subprocess
import time
import sys
import shutil
import logging
import copy
import matplotlib.pyplot as plt
plt.set_loglevel("info")
import multiprocessing as mp
import multiprocessing.pool
from pymooCFD.util.sysTools import saveTxt, yes_or_no
from pymooCFD.util.loggingTools import MultiLineFormatter, DispNameFilter

# import matplotlib.dates as mdates
# from matplotlib.ticker import AutoMinorLocator


class CFDCase:  # (PreProcCase, PostProcCase)
    baseCaseDir = None
    datFile = None
    inputFile = None
    meshFile = None
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
    onlyParallelizeSolve = False
    procLim = None
    nProc = None
    nTasks = None
    solverExecCmd = None


    # def plotVarSpace(cls, X, path):
    #     plt.scatter(X[:,0]
    #     plt.title('Design Space')


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

    def __init__(self, caseDir, x,
                 meshSF=1, meshSFs=np.around(np.arange(0.5, 1.5, 0.1), decimals=2),
                 # externalSolver=False,
                 var_labels = None, obj_labels = None,
                 meshFile=None,  # meshLines = None,
                 jobFile=None,  # jobLines = None,
                 inputFile=None,  # inputLines = None,
                 datFile=None,
                 # restart=False,
                 # solverExecCmd=None,
                 *args, **kwargs
                 ):
        super().__init__()
        if not len(self.xl) == len(self.xu) and len(self.xu) == len(self.var_labels) and len(self.var_labels) == self.n_var:
            raise Exception("Design Space Definition Incorrect")
        if not isinstance(self.baseCaseDir, str):
            raise TypeError(f'{self.baseCaseDir} - must be a string')
        ## These attributes are not taken from checkpoint
        self.caseDir = caseDir
        # self.cpPath = os.path.join(self.caseDir, 'case')
        # self.meshStudyDir = os.path.join(self.caseDir, 'meshStudy')
        self.meshSF = meshSF

        ###########################
        #    RESTART VARIABLES    #
        ###########################
        # self.complete = False
        self.restart = False
        self.parallelizeInit(self.externalSolver)
        self._x = np.array(x)

        #########################
        #    CHECKPOINT INIT    #
        #########################
        if os.path.exists(caseDir):
            self.logger = self.getLogger()
            try:
                self.loadCP()
                self.logger.info('RESTART CASE')
                return
            except FileNotFoundError as err:
                self.logger.error(err)
                self.logger.info(
                    f'OVERRIDE CASE - case directory already exists but {self.cpPath} does not exist')
                self.copy()
            # except ModuleNotFoundError as err:
            #     print(self.cpPath + '.npy')
            #     raise err
        else:
            os.makedirs(caseDir, exist_ok=True)
            self.logger = self.getLogger()
            self.logger.info('NEW CASE - directory did not exist')
            self.copy()

        #############################
        #    Optional Attributes    #
        #############################
        self.meshFile = meshFile
        self.jobFile = jobFile
        self.datFile = datFile
        self.inputFile = inputFile

        if inputFile is None:
            self.inputPath = None
        else:
            self.inputPath = os.path.join(self.caseDir, inputFile)

        # # if class level variable is None then assign instance attribute
        # if self.datFile is None:
        #     self.datFile = datFile
        if self.datFile is None:
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
        ####################
        #    Attributes    #
        ####################
        self.x = np.array(x)
        # Default Attributes
        # os.makedirs(self.baseCaseDir, exist_ok=True)



        ### Design and Objective Space Labels
        if self.var_labels is None:
            self.var_labels = [
                f'var{x_i}' for x_i in range(self.n_var)]
        # else:
        #     self.var_labels = var_labels
        if self.obj_labels is None:
            self.obj_labels = [
                f'obj{x_i}' for x_i in range(self.n_obj)]
        # else:
        #     self.obj_labels = obj_labels
        ###################################################
        #    Attributes To Be Set Later During Each Run   #
        ###################################################
        self.f = None  # if not None then case complete
        ## meshing attributes
        self.msCases = None
        self.numElem = None
        # class properties
        self._meshSF = None
        self.meshSF = meshSF
        self._meshSFs = None
        self.meshSFs = meshSFs

        self.inputLines = None
        self.jobLines = None

        self.logger.info('CASE INTITIALIZED')
        self.logger.debug('INITIAL CASE DICTONARY')
        for key in self.__dict__:
            self.logger.debug(f'\t{key}: {self.__dict__[key]}')
        # self.genMesh()
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
    def parallelizeInit(cls, externalSolver=None):
        if externalSolver is None:
            externalSolver = cls.externalSolver
        if cls.procLim is None:
            cls.nTasks = 1000000
        elif cls.nTasks is None:
            cls.nTasks = int(cls.procLim / cls.nProc)
        # else:
        #     nTasks = cls.nTasks
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
        cls.parallelizeInit()
        print('PARALLELIZING')
        print(cases)
        #cls.logger.info('PARALLELIZING . . .')
        if cls.onlyParallelizeSolve:
            print('\tParallelizing Only Solve')
            for case in cases:
                case.preProc()
            for case in cases:
                cls.pool.apply_async(case.solve, ())
            cls.pool.close()
            cls.pool.join()
            for case in cases:
                case.postProc()
        else:
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
        self.restart = True
        start = time.time()
        self._solve()
        end = time.time()
        dt = end-start
        if dt >= 3600:
            hrs = int(dt/3600)
            mins = dt%3600
            secs = dt%60
            t_str = '%i hrs | %i mins | %i secs'%(hrs, mins, secs)
        elif dt >= 60:
            mins = int(dt/60)
            secs = dt%60
            t_str = '%i mins | %i secs'%(mins, secs)
        else:
            t_str = '%i secs'%dt

        self.logger.info(f'Solve Time: {t_str}')

    def solveExternal(self):
        #self.restart = True
        #start = time.time()
        self.logger.info('SOLVING AS SUBPROCESS...')
        self.logger.info(f'\tcommand: {self.solverExecCmd}')
        subprocess.run(self.solverExecCmd, cwd=self.caseDir,
                       stdout=subprocess.DEVNULL)
        #end = time.time()
        #self.logger.info(f'Solve Time: {start-end}')

    def run(self):
        print('RUNNING', self)
        self._execDone()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(self._execDone())
        if self.f is None and not self._execDone():
            self.preProc()
            self.solve()
            if self._execDone():
                self.logger.info('COMPLETE: SOLVE')
            else:
                self.logger.warning('RUN FAILED TO EXECUTE')
                self.logger.info('RE-RUNNING')
                self.run()
            self.postProc()
        elif self.f is None:
            self.postProc()
        else:
            self.logger.warning('SKIPPED: RUN: self.run() called but case already complete')

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
        self.logger.info('COMPLETE: PRE-PROCESS')
        #self.restart = True  # ??????????????????
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
        if self.f is None or np.isnan(np.sum(self.f)):
            self._postProc()
        else:
            self.logger.info('SKIPPED: POST-PROCESSING')
            self.logger.debug(
                'self.postProc() called but self.f is not None or NaN so no action was taken')
        # Check Completion
        if self.f is None or np.isnan(np.sum(self.f)):
            self.logger.error('INCOMPLETE: POST-PROCESS')
        else:
            self.logger.info('COMPLETE: POST-PROCESS')
        self.saveCP()
        self.logger.info(f'\tObjectives:{self.f}')
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

    ####################
    #    MESH STUDY    #
    ####################
    def genMeshStudy(self):
        if self.meshSFs is None:
            self.logger.warning('self.meshSFs is None but self.genMeshStudy() called')
            return
        self.logger.info('\tGENERATING MESH STUDY . . .')
        self.logger.info(f'\t\tFor Mesh Size Factors: {self.meshSFs}')
        # Pre-Process
        study = []
        var = []
        self.msCases = []
        for sf in self.meshSFs:
            msCase = copy.deepcopy(self)
            self.msCases.append(msCase)
            fName = f'meshSF-{sf}'
            path = os.path.join(self.meshStudyDir, fName)
            self.logger.info(f'\t\tInitializing {path} . . .')
            msCase.meshSFs = None
            msCase.msCases = None
            msCase.__init__(path, self.x)
            msCase.meshSFs = None
            msCase.msCases = None
            if msCase.meshSF != sf or msCase.numElem is None:
                # only pre-processing needed is generating mesh
                msCase.meshSF = sf
                msCase.genMesh() # NOT NESSECESARY BECAUSE FULL PRE-PROCESS DONE AT RUN
            else:
                self.logger.info(f'\t\t\t{msCase} already has number of elements: {msCase.numElem}')
            # sfToElem.append([msCase.meshSF, msCase.numElem])
            saveTxt(msCase.caseDir, 'numElem.txt', [msCase.numElem])
            study.append([msCase.caseDir, str(
                msCase.numElem), str(msCase.meshSF)])
            var.append(msCase.x)
        study = np.array(study)
        self.logger.info('\tStudy:\n\t\t'+str(study).replace('\n', '\n\t\t'))
        # self.sfToElem = np.array(sfToElem)
        # print('\t' + str(study).replace('\n', '\n\t'))
        path = os.path.join(self.meshStudyDir, 'study.txt')
        np.savetxt(path, study, fmt="%s")
        saveTxt(self.meshStudyDir, 'study.txt', study, fmt="%s")
        self.saveCP()
        # path = os.path.join(self.meshStudyDir, 'studyX.txt')
        # np.savetxt(path, var)

        # obj = np.array([case.f for case in self.msCases])
        # print('Objectives:\n\t', obj)
        # path = os.path.join(self.meshStudyDir, 'studyF.txt')
        # np.savetxt(path, obj)

    def plotMeshStudy(self):
        self.logger.info('\tPLOTTING MESH STUDY')
        _, tail = os.path.split(self.caseDir)
        a_numElem = np.array([case.numElem for case in self.msCases])
        a_sf = [case.meshSF for case in self.msCases]
        msObj = np.array([case.f for case in self.msCases])
        # Plot
        for obj_i, obj_label in enumerate(self.obj_labels):
            # Number of Elements Plot
            self.logger.info(f'\t\tPlotting Objective {obj_i}: {obj_label}')
            plt.plot(a_numElem, msObj[:, obj_i], 'o')
            plt.suptitle('Mesh Sensitivity Study')
            plt.title(tail)
            plt.xlabel('Number of Elements')
            plt.ylabel(obj_label)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            fName = f'ms_plot-{tail}-obj{obj_i}-numElem.png'
            fPath = os.path.join(self.meshStudyDir, fName)
            plt.tight_layout()
            plt.savefig(fPath, bbox_inches='tight')
            plt.clf()
            # Mesh Size Factor Plot
            plt.plot(a_sf, msObj[:, obj_i], 'o')
            plt.suptitle('Mesh Sensitivity Study')
            plt.title(tail)
            plt.xlabel('Mesh Size Factor')
            plt.ylabel(obj_label)
            fName = f'ms_plot-{tail}-obj{obj_i}-meshSFs.png'
            fPath = os.path.join(self.meshStudyDir, fName)
            plt.tight_layout()
            plt.savefig(fPath)
            plt.clf()
        self.saveCP()

    def execMeshStudy(self):
        self.logger.info('\tEXECUTING MESH STUDY')
        # self.logger.info(f'\t\tPARALLELIZING:\n\t\t {self.msCases}')
        self.parallelize(self.msCases)
        obj = np.array([case.f for case in self.msCases])
        self.logger.info('\tObjectives:\n\t\t'+str(obj).replace('\n', '\n\t\t'))
        self.saveCP()
        # nTask = int(self.procLim/self.BaseCase.nProc)
        # pool = mp.Pool(nTask)
        # for case in self.msCases:
        #     pool.apply_async(case.run, ())
        # pool.close()
        # pool.join()

    def meshStudy(self, restart=True):  # , meshSFs=None):
        if self.meshSFs is None:
            self.logger.error('EXITING MESH STUDY: Mesh Size Factors set to None. May be trying to do mesh study on a mesh study case.')
            return
        # if meshSFs is None:
        #     meshSFs = self.meshSFs
        # if self.msCases is None:
        #     self.genMeshStudy()
        self.logger.info(f'MESH STUDY')
        if self.msCases is None:
            self.logger.info('\tNo Mesh Cases Found: self.msCases is None')
            self.logger.info(f'\t {self.meshSFs}')
        else:
            prev_meshSFs = [case.meshSF for case in self.msCases]
            self.logger.info(f'\tCurrent Mesh Size Factors:\n\t\t{self.meshSFs}')
            self.logger.info(f'\tPrevious Mesh Study Size Factors:\n\t\t{prev_meshSFs}')
            if all(sf in prev_meshSFs for sf in self.meshSFs):
                self.logger.info('\tALL CURRENT MESH SIZE FACTORS IN PREVIOUS MESH SIZE FACTORS')
                self.plotMeshStudy()
                self.logger.info('SKIPPED: MESH STUDY')
                return
            else:
                self.logger.info(
                    '\t\tOLD MESH SIZE FACTORS != NEW MESH SIZE FACTORS')
        # if not restart or self.msCases is None:
        #     self.genMeshStudy()
        # else:
        #     print('\tRESTARTING MESH STUDY')
        # else:
        #     self.msCases =
        self.genMeshStudy()
        # Data
        a_numElem = [case.numElem for case in self.msCases]
        a_sf = [case.meshSF for case in self.msCases]
        dat = np.column_stack((a_numElem, a_sf))
        # Print
        self.logger.info(f'\tMesh Size Factors: {self.meshSFs}')
        self.logger.info(f'\tNumber of Elements: {a_numElem}')
        with np.printoptions(suppress=True):
            self.logger.info('\tNumber of Elements | Mesh Size Factor\n\t\t'+str(dat).replace('\n', '\n\t\t'))
            saveTxt(self.meshStudyDir, 'numElem-vs-meshSFs.txt', dat)

        self.execMeshStudy()
        self.plotMeshStudy()

    ##########################
    #    CLASS PROPERTIES    #
    ##########################
    # @propert
    # def caseDir(self):
    #     return self.caseDir
    # @caseDir.setter
    # def caseDir(self, caseDir):
    #     os.makedirs(caseDir, exist_ok)
    #     self.
    #     self.caseDir = caseDir

    @property
    def meshStudyDir(self):
        return os.path.join(self.caseDir, 'meshStudy')

    @property
    def cpPath(self):
        return os.path.join(self.caseDir, 'case.npy')


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
        with open(self.inputPath, 'r') as f:
            inputLines = f.readlines()
        return inputLines

    @inputLines.setter
    def inputLines(self, lines):
        if self.inputPath is None:
            self.logger.info('self.inputPath is None: Input lines not written')
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

    @property
    def meshSFs(self):return self._meshSFs
    @meshSFs.setter
    def meshSFs(self, meshSFs):
        if meshSFs is None:
            self._meshSFs = meshSFs
            return
        meshSFs, counts = np.unique(meshSFs, return_counts=True)
        for sf_i, n_sf in enumerate(counts):
            if n_sf > 1:
                self.logger.warning(f'REPEATED MESH SIZE FACTOR - {meshSFs[sf_i]} repeated {n_sf} times')

        if self.msCases is None:
            self._meshSFs = meshSFs
        else:
            prev_meshSFs = [case.meshSF for case in self.msCases]
            self.logger.debug(f'Current Mesh Size Factors:\n\t{self.meshSFs}')
            self.logger.debug(f'Previous Mesh Study Size Factors:\n\t{prev_meshSFs}')
            if all(sf in prev_meshSFs for sf in self.meshSFs):
                self.logger.debug('ALL CURRENT MESH SIZE FACTORS IN PREVIOUS MESH SIZE FACTORS')
                self._meshSFs = meshSFs
            else:
                self.logger.debug('OLD MESH SIZE FACTORS != NEW MESH SIZE FACTORS')
                self._meshSFs = meshSFs
                self.genMeshStudy()

    # @property
    # def meshSF(self): return self._meshSF
    # @meshSF.setter
    # def meshSF(self, meshSF):
    #     if meshSF != self._meshSF:
    #         self._meshSF = meshSF
    #         self.genMesh()
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
        ## Get Child Logger using hierarchical "dot" convention
        logger = logging.getLogger(__name__+'.'+self.caseDir)
        logger.setLevel(logging.DEBUG)
        ### Filters
        ## Filters added to logger do not propogate up logger hierarchy
        ## Filters added to handlers do propogate
        # filt = DispNameFilter(self.caseDir)
        # logger.addFilter(filt)
        ## File Handle
        logFile = os.path.join(self.caseDir, f'{tail}.log')
        fileHandler = logging.FileHandler(logFile)
        logger.addHandler(fileHandler)
        ### Stream Handler
        ## parent root logger takes care of stream display
        ### Formatter
        formatter = MultiLineFormatter(
            '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s')
        fileHandler.setFormatter(formatter)
        ### Initial Message
        logger.info('-'*30)
        logger.info('LOGGER INITIALIZED')
        return logger

    #######################
    #    CHECKPOINTING    #
    #######################
    def saveCP(self):
        cpPath = self.cpPath.replace('.npy', '')
        try:
            np.save(cpPath + '.temp.npy', self)
            if os.path.exists(cpPath + '.npy'):
                os.rename(cpPath + '.npy', cpPath + '.old.npy')
            os.rename(cpPath + '.temp.npy', cpPath + '.npy')
            if os.path.exists(cpPath + '.old.npy'):
                os.remove(cpPath + '.old.npy')
        except FileNotFoundError as err:
            self.logger.error(str(err))

    def loadCP(self):
        if os.path.exists(self.cpPath + '.old'):
            os.rename(self.cpPath + '.old', self.cpPath + '.npy')
        cp, = np.load(self.cpPath + '.npy', allow_pickle=True).flatten()
        self.logger.debug('\tRESTART DICTONARY:')
        for key in self.__dict__:
            self.logger.debug(f'\t\t{key}: {self.__dict__[key]}')
        self.logger.debug('\tCHECKPOINT DICTONARY:')
        for key in cp.__dict__:
            self.logger.debug(f'\t\t{key}: {cp.__dict__[key]}')
        # print(cp._x)
        if np.array_equal(self._x, cp._x):
            if self.caseDir != cp.caseDir:
                self.logger.warning('CASE DIRECTORY CHANGED BETWEEN CHECKPOINTS')
                self.logger.debug(str(cp.caseDir)+' -> '+str(self.caseDir))
            if cp.cpPath != self.cpPath:
                self.logger.warning(f'{cp.cpPath} != {self.cpPath}')
            if cp.meshSF != self.meshSF:
                self.logger.warning(f'{cp.meshSF} != {self.meshSF}')
                self.logger.warning('Running genMesh() to reflect change in mesh size factor')
                self.numElem = None
                self.genMesh()
                cp.meshSF = self.meshSF
                self.logger.info('Mesh size factor changed, mesh generated, self.f and self.numElem set to None')
                cp.f = None
            cp.caseDir = self.caseDir
            # cp.cpPath = self.cpPath
            # cp.meshStudyDir = self.meshStudyDir
            # cp.meshSFs = self.meshSFs
            cp.baseCaseDir = self.baseCaseDir
            # cp.logger = self.getLogger()
            self.__dict__.update(cp.__dict__)
            self.logger.info(f'CHECKPOINT LOADED - {self.cpPath}.npy')
        else:
            self.logger.info(f'Given Parameters: {self._x}')
            self.logger.info(f'Checkpoint Parameters: {cp._x}')
            question = f'\nCASE PARAMETERS DO NOT MATCH.\nEMPTY AND RESET {self.caseDir}?'
            delete = yes_or_no(question)
            if delete:
                shutil.rmtree(self.caseDir)
                self.__init__(self.caseDir, self.x)
            else:
                self.logger.exception('GIVEN PARAMETERS DO NOT MATCH CHECKPOINT PARAMETERS.')
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
        ## Directory Name Changes


    ########################
    #    HELPER METHODS    #
    ########################
    def copy(self):
        # if os.path.exists(self.caseDir):
        #     self.logger.warning('CASE OVERRIDE - self.caseDir already existed')
        shutil.copytree(self.baseCaseDir, self.caseDir, dirs_exist_ok=True)
        self.logger.info(f'COPIED FROM: {self.baseCaseDir}')

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
        s = f'Directory: {self.caseDir} | Parameters: {self.x}'
        if self._f is not None:
            s += f' | Objectives: {self._f}'
        return s

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
        # pass

    # def _pySolve(self):
    #     pass

    def _solve(self):
        self.logger.error('OVERRIDE _solve(self) method to execute internal python solver OR use CFDCase.externalSolver=True')

    def _execDone(self):
        return True

    def _postProc(self):
        pass

    def _genMesh(self):
        pass

from pymooCFD.util.handleData import findKeywordLine
import re

class YALES2Case(CFDCase):
    # def __init__(self, caseDir, x, *args, **kwargs):
    #     super().__init__(caseDir, x, *args, **kwargs)

    def solve(self):
        super().solve()
        self.wallTime = self.getWallTime()

    def getWallTime(self):
        fName, = glob('solver01_rank*.log')
        with open(fName, 'rb') as f:
            try:  # catch OSError in case of a one line file
                f.seek(-1020, os.SEEK_END)
            except OSError:
                f.seek(0)
            clock_line = f.readline().decode()
        if 'WALL CLOCK TIME' in clock_line:
            wall_time = int(float(clock_line[-13:]))
            self.logger.info(f'YALES2 Wall Clock Time: {wall_time} seconds')
        else:
            self.logger.warning('no wall clock time found')
            wall_time = None
        return wall_time

    def getLatestXMF(self):
        ents = os.listdir(self.dumpDir)
        ents.sort()
        for ent in ents:
            if ent.endswith('.xmf') and not re.search('.sol.+_.+\\.xmf', ent):
                latestXMF = ent
        return latestXMF

    def getLatestMesh(self):
        ents = os.listdir(self.dumpDir)
        ents.sort()
        for ent in ents:
            if ent.endswith('.mesh.h5'):
                latestMesh = ent
        return latestMesh

    def getLatestSoln(self):
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
        return os.path.join(self.caseDir, 'dump')
###################
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
