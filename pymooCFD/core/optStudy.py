# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:28:45-05:00
# import time
import sys
import logging
import os
import numpy as np
import shutil
# import subprocess
# from threading import Thread
# import multiprocessing as mp
import copy
# import shutil
# import h5py
# import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymooCFD.util.sysTools import emptyDir, copy_and_overwrite, saveTxt, yes_or_no
from pymooCFD.util.loggingTools import MultiLineFormatter, DispNameFilter
import pymooCFD.config as config
from pymoo.util.misc import termination_from_tuple


class OptStudy:
    def __init__(self, algorithm, problem, BaseCase,
                 # restart=True,
                 # optDatDir='opt_run',
                 optName=None,
                 n_opt=20,  # n_CP=10,
                 # CP_fName='checkpoint',
                 # plotsDir='plots', archiveDir='archive', mapGen1Dir='mapGen1',
                 runDir='run',
                 # procLim=os.cpu_count(),
                 # var_labels=None, obj_labels=None,
                 # client=None,
                 # *args, **kwargs
                 ):
        super().__init__()
        #######################################
        #    ATTRIBUTES NEEDED FOR RESTART    #
        #######################################
        if optName is None:
            self.optName = self.__class__.__name__
        else:
            self.optName = optName
        self.logger = self.getLogger()
        self.logger.info(f'OPTIMIZATION STUDY - {self.optName}')
        self.optDatDir = 'optStudy-' + self.optName
        self.logger.info(f'\tData Directory: {self.optDatDir}')
        self.CP_path = os.path.join(self.optDatDir, f'{self.optName}-CP')
        self.logger.info(f'\tCheckpoint Path: {self.CP_path}')

        # if not restart:
        # if os.path.exists(self.optDatDir):
        #     if os.path.exists(self.CP_path):
        #         self.loadCP()
        #         self.logger.info('RESTARTED FROM', self.CP_path)
        #         return
        #     else:
        #         question = f'{self.CP_path} does not exist.\nOVERWRITE {self.optDatDir}?'
        #         overwrite = yes_or_no(question)
        #         if overwrite:
        #             os.removedirs(self.optDatDir)
        #             os.makedirs(optDatDir, exist_ok=True)
        #         else:
        #             question = f'{self.CP_path} does not exist.\n?'
        #             overwrite = yes_or_no(question)

        if os.path.isdir(self.optDatDir):
            try:
                self.loadCP()
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
            # except FileNotFoundError:
            #     print('OVERRIDE OPTIMIZATION STUDY -')
            #     print('\t{self.CP_path} already exists but {self.cpPath} does not')
            #     self.copy()
        # else:
        #     os.makedirs(caseDir, exist_ok=True)
        #     self.logger = self.getLogger()
        #     self.logger.info(f'NEW CASE - {caseDir} did not exist')
        #     self.copy()
        #############################
        #    Required Attributes    #
        #############################
        self.problem = problem
        self.algorithm = algorithm  # algorithm.setup(problem)
        # initialize baseCase
        self.BaseCase = BaseCase
        # self.baseCaseDir = baseCaseDir
        # self.genTestCase()
        #####################################
        #    Default/Optional Attributes    #
        #####################################
        self.runDir = runDir
        # self.runDir = os.path.join(self.optDatDir, runDir)
        # os.makedirs(self.runDir, exist_ok=True)

        ### Data Handling ###
        # self.archiveDir = os.path.join(self.optDatDir, archiveDir)
        # os.makedirs(self.archiveDir, exist_ok=True)
        # self.n_CP = n_CP  # number of generations between extra checkpoints

        ### Optimization Pre/Post Processing ###
        # self.procOptDir = os.path.join(self.optDatDir, procOptDir)
        # os.makedirs(self.procOptDir, exist_ok=True)
        # Pareto Front Directory
        # directory to save optimal solutions (a.k.a. Pareto Front)
        # self.pfDir = os.path.join(self.runDir, pfDir)
        # os.makedirs(self.pfDir, exist_ok=True)
        # number of optimal points along Pareto front to save
        self.n_opt = int(n_opt)
        # Plots Directory
        # self.plotDir = os.path.join(self.runDir, plotsDir)
        # os.makedirs(self.plotDir, exist_ok=True)
        # Mapping Objectives vs. Variables Directory
        # self.mapDir = os.path.join(self.runDir, mapGen1Dir)
        # os.makedirs(self.mapDir, exist_ok=True)
        #### Mesh Sensitivity Studies ###
        # self.studyDir = os.path.join(self.optDatDir, meshStudyDir)
        # os.makedirs(self.studyDir, exist_ok=True)
        # self.meshSFs = meshSFs  # mesh size factors
        # self.procLim = procLim
        ##### Processing #####
        # self.client = client
        ###################################
        #    Attributes To Be Set Later   #
        ###################################
        self.gen1Pop = None
        self.cornerCases = None
        self.bndCases = None
        ### Test Case ###
        self.testCase = None
        self.genTestCase()
        self.saveCP()

    def newAlg(self):
        self.logger.info('INITIALIZING NEW OPTIMIZATION AlGORITHM')
        # archive/empty previous runs data directory
        # emptyDir(self.optDatDir)
        self.algorithm.setup(self.problem,
                             seed=self.algorithm.seed,
                             verbose=self.algorithm.verbose,
                             save_history=self.algorithm.save_history,
                             return_least_infeasible=self.algorithm.return_least_infeasible
                             )
        self.algorithm.callback.__init__()
        self.saveCP()

    def initAlg(self):
        if self.algorithm.is_initialized:
            self.loadCP()
            self.logger.info(f'Loaded Algorithm Checkpoint: {self.algorithm}')
            self.logger.info(
                f'\tLast checkpoint at generation {self.algorithm.callback.gen}')
        else:
            self.newAlg()

    def run(self, delPrevGen=True):
        self.logger.info('STARTING: OPTIMIZATION ALGORITHM RUN')
        self.algorithm.save_history = True
        self.initAlg()
        # if self.algorithm.problem is None or not restart:
        # if self.algorithm.is_initialized:
        #     self.loadCP()
        #     self.logger.info(f'Loaded Checkpoint: {self.algorithm}')
        #     self.logger.info(
        #         f'Last checkpoint at generation {self.algorithm.callback.gen}')
        #
        # else:
        #     self.newStudy()
        # self.logger.info('STARTING NEW OPTIMIZATION STUDY')
        # # archive/empty previous runs data directory
        # emptyDir(self.optDatDir)
        # self.algorithm.setup(self.problem,
        #                      seed=self.algorithm.seed,
        #                      verbose=self.algorithm.verbose,
        #                      save_history=self.algorithm.save_history,
        #                      return_least_infeasible=self.algorithm.return_least_infeasible
        #                      )
        # restart client if being used
        # if self.client is not None:
        #     self.client.restart()
        #     self.logger.info("CLIENT RESTARTED")

        ######    OPTIMIZATION    ######
        # until the algorithm has not terminated
        while self.algorithm.has_next():
            # print('RESTART WHILE LOOP')
            # print('n_gen:', self.algorithm.n_gen)
            # print('history:', self.algorithm.history)
            # print('alg. off.:', self.algorithm.off)
            # print('opt:', self.algorithm.opt)
            # if self.algorithm.pop is not None:
            # print('BEFORE ASK:')
            # print('gen', self.algorithm.callback.gen)
            # print(self.algorithm.pop.get('F'))
            # print('history[-1].off', self.algorithm.history[-1].off)
            # First generation
            # population is None so ask for new pop
            if self.algorithm.pop is None:
                self.logger.info('\tSTART-UP: first generation')
                evalPop = self.algorithm.ask()
                self.algorithm.pop = evalPop
                self.algorithm.off = evalPop
            # Previous generation complete
            # If current objective does not have None values then get new pop
            # ie previous pop is complete
            # evaluate new pop
            # elif None not in self.algorithm.pop.get('F'):
            elif self.algorithm.off is None:
                self.logger.info('\tSTART-UP: new generation')
                evalPop = self.algorithm.ask()
                self.algorithm.off = evalPop
            # Mid-generation start-up
            # evaluate offspring population
            else:
                self.logger.info('\tSTART-UP: mid-generation')
                evalPop = self.algorithm.off
                # self.algorithm.callback.gen -= 1
            # print('AFTER ASK:')
            # print('\talgorithm.n_gen:', self.algorithm.n_gen)
            # print('\tcallbck.gen:', self.algorithm.callback.gen)
            # print(self.algorithm.pop.get('F'))
            # print('alg. off.:', self.algorithm.off)
            # print('evalPop:', evalPop)
            # print('opt:', self.algorithm.opt)
            # input('any key to continue')
            # save checkpoint before evaluation
            self.saveCP()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            self.algorithm.evaluator.eval(self.problem, evalPop)
            # print('self.algorithm.callback.gen:', self.algorithm.callback.gen)
            # print('self.algorithm.n_gen:', self.algorithm.n_gen)

            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=evalPop)
            # print('AFTER TELL:')
            # print('\tself.algorithm.callback.gen =',
            #       self.algorithm.callback.gen)
            # print('\tself.algorithm.n_gen =', self.algorithm.n_gen)
            # save top {n_opt} optimal evaluated cases in pf directory
            compGen = self.algorithm.callback.gen - 1
            for off_i, off in enumerate(self.algorithm.off):
                for opt_i, opt in enumerate(self.algorithm.opt[:self.n_opt]):
                    if np.array_equal(off.X, opt.X):
                        optDir = os.path.join(self.pfDir, f'opt{opt_i+1}')
                        offDir = os.path.join(
                            self.runDir, f'gen{compGen}', f'ind{off_i+1}')
                        self.logger.info(
                            f'\tUpdating Pareto front folder: {offDir} -> {optDir}')
                        try:
                            copy_and_overwrite(offDir, optDir)
                        except FileNotFoundError as err:
                            self.logger.error(str(err))
                            self.logger.warning('SKIPPED: UPDATE PARETO FRONT')
            # do some more things, printing, logging, storing or even modifying the algorithm object
            # self.logger.info(algorithm.n_gen, algorithm.evaluator.n_eval)
            # self.logger.info('Parameters:')
            # self.logger.info(algorithm.pop.get('X'))
            # self.logger.info('Objectives:')
            # self.logger.info(algorithm.pop.get('F'))
            # algorithm.display.do(algorithm.problem,
            #                      algorithm.evaluator,
            #                      algorithm
            #                      )
            # checkpoint saved
            # print('AFTER TELL:')
            # print('n_gen:', self.algorithm.n_gen)
            # print('gen', self.algorithm.callback.gen)
            # print(self.algorithm.pop.get('F'))
            # print('alg. off.:', self.algorithm.off)
            # print('opt:', self.algorithm.opt)
            # input('any key to continue')
            self.algorithm.off = None
            self.saveCP()
            if delPrevGen and not compGen == 1:
                direct = os.path.join(
                    self.runDir, f'gen{compGen}')
                shutil.rmtree(direct)
        # obtain the result objective from the algorithm
        res = self.algorithm.result()
        # calculate a hash to show that all executions end with the same result
        self.logger.info(f'hash {res.F.sum()}')
        # self.saveCP()

    def execGen1(self):
        self.algorithm.termination = termination_from_tuple(('n_gen', 1))
        # no checkpoint saved before evaluation
        # therefore self.slgorithm not stuck with after gen.1 termination criteria
        self.algorithm.next()

    def runGen1(self):
        self.logger.info('RUNNING GENERATION 1')
        self.initAlg()
        try:
            gen1Alg = self.algorithm.history[0]
            popX = gen1Alg.pop.get('X')
            popF = gen1Alg.pop.get('F')
            self.logger.info('\tCurrent Generation 1:')
            for i in len(popX):
                self.logger.info(
                    f'\t\tParameters-{popX[i]} | Objectives- {popF[i]}')
        except TypeError:
            self.logger.info('\tNo Generation 1 Population Found')
            self.logger.info(f'\tAlgorithm History: {self.algorithm.history}')
            self.algorithm.termination = termination_from_tuple(('n_gen', 1))
            # no checkpoint saved before evaluation
            # therefore self.slgorithm not stuck with after gen.1 termination criteria
            self.algorithm.next()
        self.logger.info('COMPLETE: RUN GENERATION 1')
        self.saveCP()

    ################
    #    LOGGER    #
    ################

    def getLogger(self):
        # create root logger
        logger = logging.getLogger()
        logger.setLevel(config.OPT_STUDY_LOGGER_LEVEL)
        # define handlers
        # if not logger.handlers:
        # file handler
        fileHandler = logging.FileHandler(f'{self.optName}.log')
        logger.addHandler(fileHandler)
        # stream handler
        streamHandler = logging.StreamHandler()  # sys.stdout)
        streamHandler.setLevel(logging.DEBUG)
        if streamHandler not in logger.handlers:
            logger.addHandler(streamHandler)
        # define filter
        # filt = DispNameFilter(self.optName)
        # logger.addFilter(filt)
        # define formatter
        formatter = MultiLineFormatter(
            '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s')
        # formatter = MultiLineFormatter(
        #     f'%(asctime)s :: %(levelname)-8s :: {self.optName} :: %(message)s')
        # formatter = logging.Formatter(
        #     f'%(asctime)s.%(msecs)03d :: %(levelname)-8s :: {self.optName} :: %(message)s')
        # ' %(name)s :: %(levelname)-8s :: %(message)s')
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)
        logger.info('~' * 30)
        logger.info('NEW RUN')
        return logger

    ####################
    #    MESH STUDY    #
    ####################
    def meshStudy(self, cases):
        for case in cases:
            self.logger.info(f'MESH STUDY - {case}')
            case.meshStudy()

    #######################
    #    CHECKPOINTING    #
    #######################
    # def saveCP(self): np.save(self.CP_path, self)
    # def loadCP(self):
    #     cp, = np.load(self.CP_path, allow_pickle=True).flatten()
    #     self.__dict__.update(cp.__dict__)
    #     self.logger.info('LOADING CHECKPOINT')
    #######################
    #    CHECKPOINTING    #
    #######################
    # def saveCP(self):
    #     np.save(self.CP_path + '.temp.npy', self)
    #     if os.path.exists(self.CP_path + '.npy'):
    #         os.rename(self.CP_path + '.npy', self.CP_path + '.old.npy')
    #     os.rename(self.CP_path + '.temp.npy', self.CP_path + '.npy')
    #     if os.path.exists(self.CP_path + '.old.npy'):
    #         os.remove(self.CP_path + '.old.npy')
    #
    # def loadCP(self):
    #     if os.path.exists(self.CP_path + '.old'):
    #         os.rename(self.CP_path + '.old', self.CP_path + '.npy')
    #     cp, = np.load(self.CP_path + '.npy', allow_pickle=True).flatten()
    #     self.__dict__.update(cp.__dict__)
    #     self.logger.info(f'\tCHECKPOINT LOADED - from {self.CP_path}.npy')

    def loadCP(self, hasTerminated=False):
        if os.path.exists(self.CP_path + '.old'):
            os.rename(self.CP_path + '.old', self.CP_path + '.npy')
        cp, = np.load(self.CP_path + '.npy', allow_pickle=True).flatten()

        # logging
        self.logger.info(f'\tCHECKPOINT LOADED: {self.CP_path}.npy')
        self.logger.debug('\tRESTART DICTONARY')
        for key in self.__dict__:
            self.logger.debug(f'\t\t{key}: {self.__dict__[key]}')
        self.logger.debug('\tCHECKPOINT DICTONARY')
        for key in cp.__dict__:
            self.logger.debug(f'\t\t{key}: {cp.__dict__[key]}')

        if cp.algorithm is not None:
            self.logger.debug('\tOPTIMIZATION ALGORITHM DICTONARY:')
            for key, val in cp.algorithm.__dict__.items():
                self.logger.debug(f'\t\t{key}: {val}')
        # # update paths and loggers for
        # cp.CP_path = self.CP_path
        # cp.optName = self.optName
        # cp.optDatDir = self.optDatDir
        # cp.logger = self.logger
        # cp.logger = self.getLogger()
        #### TEMPORARY CODE ##########
        # TRANSITION BETWEEN CHECKPOINTS

        self.__dict__.update(cp.__dict__)
        # self.logger.info(f'\tCHECKPOINT LOADED - from {self.CP_path}.npy')
        # only necessary if for the checkpoint the termination criterion has been met
        try:
            self.algorithm.has_terminated = hasTerminated
        except AttributeError as err:
            self.logger.error(err)
        # alg = self.algorithm
        # # Update any changes made to the algorithms between runs
        # alg.termination = self.algorithm.termination
        # alg.pop_size = self.algorithm.pop_size
        # alg.n_offsprings = self.algorithm.n_offsprings
        # alg.problem.xl = np.array(self.problem.xl)
        # alg.problem.xu = np.array(self.problem.xu)
        # self.algorithm = alg

    def saveCP(self):  # , alg=None):
        gen = self.algorithm.callback.gen
        self.logger.info(f'SAVING CHECKPOINT - GENERATION {gen}')
        if self.algorithm.pop is not None:
            genX = self.algorithm.pop.get('X')
            if None not in genX:
                saveTxt(self.runDir, f'gen{gen}X.txt', genX)
            genF = self.algorithm.pop.get('F')
            if None not in genF:
                saveTxt(self.runDir, f'gen{gen}F.txt', genF)
            if self.algorithm.off is None:
                self.logger.info(f'\tgeneration {gen} complete')

        elif self.algorithm.off is not None:  # and self.algorithm.pop is not None
            self.logger.info('\tmid-generation checkpoint')
        # except TypeError:
        #     self.logger.info('\tmid-generation')
        # save checkpoint
        np.save(self.CP_path + '.temp.npy', self)
        if os.path.exists(self.CP_path + '.npy'):
            os.rename(self.CP_path + '.npy', self.CP_path + '.old.npy')
        os.rename(self.CP_path + '.temp.npy', self.CP_path + '.npy')
        if os.path.exists(self.CP_path + '.old.npy'):
            os.remove(self.CP_path + '.old.npy')
        # Checkpoint each cfdCase object stored in optStudy
        try:
            self.testCase.saveCP()
        except AttributeError:
            self.logger.debug('No Test Case to Save Checkpoint for')
        except FileNotFoundError:
            self.logger.debug('No Test Case to Save Checkpoint for')
        try:
            for case in self.bndCases:
                case.saveCP()
        except TypeError:
            self.logger.debug('No Boundary Cases to Save Checkpoints for')
        except AttributeError:
            self.logger.debug('No Boundary Cases to Save Checkpoints for')
            # self.logger.error(e)
        try:
            for case in self.cornerCases:
                case.saveCP()
        except TypeError:
            self.logger.debug('No Corner Cases to Save Checkpoints for')
        except AttributeError:
            self.logger.debug('No Corner Cases to Save Checkpoints for')
            # self.logger.error(e)

        # # default use self.algorithm
        # if alg is None:
        #     alg = self.algorithm
        # # if give alg assign it to self.algorithm
        # else:
        #     self.algorithm = alg

        # genDir = f'gen{gen}'
        # os.path.join(optDatDir, 'checkpoint')
        # np.save(self.CP_path, alg)
        # gen0 and every nCP generations save additional static checkpoint
        # if gen % self.n_CP == 1:
        #     np.save(os.path.join(self.optDatDir, f'checkpoint-gen{gen}'), self)
        # save text file of variables and objectives as well
        # this provides more options for post-processesing data
        # genX = self.algorithm.pop.get('X')
        # saveTxt(self.optDatDir, f'gen{gen}X.txt', genX)
        # try:
        #     genF = self.algorithm.pop.get('F')
        #     saveTxt(self.optDatDir, f'gen{gen}F.txt', genF)
        #     # path = os.path.join(self.optDatDir, f'gen{gen}F.txt')
        #     # np.savetxt(path, genF)
        # except TypeError:  # AttributeError
        #     self.logger.info('\tmid-generation')

    def checkCPs(self):
        self.logger.info(f'CHECKPOINT CHECK - {self.CP_path}.npy')
        if os.path.exists(f'{self.CP_path}.npy'):
            for cp in np.load(self.CP_path):
                self.logger.info(cp.__dict__)
        else:
            self.logger.info(f'\t{self.CP_path} does not exist')

    # def saveCP(self, alg=None):
    #     # default use self.algorithm
    #     if alg is None:
    #         alg = self.algorithm
    #     # if give alg assign it to self.algorithm
    #     else:
    #         self.algorithm = alg
    #     gen = alg.callback.gen
    #     self.logger.info(f'SAVING CHECKPOINT - GENERATION {gen}')
    #     # genDir = f'gen{gen}'
    #     # os.path.join(optDatDir, 'checkpoint')
    #     np.save(self.CP_path, alg)
    #     # gen0 and every nCP generations save additional static checkpoint
    #     if gen % self.n_CP == 1:
    #         np.save(os.path.join(self.optDatDir, f'checkpoint-gen{gen}'), alg)
    #     # save text file of variables and objectives as well
    #     # this provides more options for post-processesing data
    #     genX = alg.pop.get('X')
    #     path = os.path.join(self.optDatDir, f'gen{gen}X.txt')
    #     np.savetxt(path, genX)
    #     try:
    #         genF = alg.pop.get('F')
    #         path = os.path.join(self.optDatDir, f'gen{gen}F.txt')
    #         np.savetxt(path, genF)
    #     except TypeError:  # AttributeError
    #         self.logger.info('\tmid-generation')

    ###################
    #    TEST CASE    #
    ###################
    def genTestCase(self, testCaseDir='test_case'):
        self.logger.info('\tGENERATING TEST CASE')
        # shutil.rmtree('test_case', ignore_errors=True)
        xl = self.problem.xl
        xu = self.problem.xu
        x_mid = [xl[x_i] + (xu[x_i] - xl[x_i]) / 2
                 for x_i in range(self.problem.n_var)]
        # for x_i, type in enumerate(self.BaseCase.varType):
        #     if type == 'int':
        #         x_mid[x_i] = int(x_mid[x_i])
        # x_mid = [int(x_mid[x_i]) for x_i, type in enumerate(self.BaseCase.varType)
        #          if type == 'int']
        for x_i, varType in enumerate(self.BaseCase.varType):
            if varType.lower() == 'int':
                x_mid[x_i] = int(x_mid[x_i])
        testCaseDir = os.path.join(self.runDir, testCaseDir)
        self.testCase = self.BaseCase(testCaseDir, x_mid)  # , restart=True)

    def runTestCase(self):
        self.logger.info('TEST CASE RUN . . .')
        # if self.testCase is None:
        self.genTestCase()
        self.logger.info('\tRUNNING TEST CASE')
        self.testCase.run()
        self.logger.info(f'\tParameters: {self.testCase.x}')
        self.logger.info(f'\tObjectives: {self.testCase.f}')
        self.logger.info('TEST CASE COMPLETE ')
        self.saveCP()
        return self.testCase

    #################################
    #    RUN POPULATION OF CASES    #
    #################################
    def runGen(self, X, out):
        gen = self.algorithm.callback.gen
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(self.runDir, f'gen{gen}')
        # create sub-directories for each individual
        indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
        cases = self.genCases(indDirs, X)
        self.BaseCase.parallelize(cases)
        # self.runPop(cases)
        for case in cases:
            print(case.caseDir, case.f, case.x)
        print(np.array([case.f for case in cases]))
        out['F'] = np.array([case.f for case in cases])
        if gen == 1:
            self.gen1Pop = cases
        return out

    # def runPop(self, cases):
    #     nTask = int(self.procLim/self.BaseCase.nProc)
    #     pool = mp.Pool(nTask)
    #     for case in cases:
    #         pool.apply_async(case.run, ())
    #     pool.close()
    #     pool.join()
    # def runPop(self, cases):
    #     self.preProcPop(cases)
    #     self.execPop(cases)
    #     self.postProcPop(cases)
    # def preProcPop(self, cases):
    #     for case in cases:
    #         case.preProc()
    # def execPop(self, cases):
    #     self._execPop(cases)
    # def postProcPop(self, cases):
    #     for case in cases:
    #         case.postProc()

    ########################
    #    BOUNDARY CASES    #
    ########################
    def getLimPerms(self):
        from itertools import product
        xl = self.problem.xl
        xu = self.problem.xu
        n_var = self.problem.n_var
        n_perm = 2**n_var
        # find every binary permutations of length n_var
        bin_perms = [list(i) for i in product([0, 1], repeat=n_var)]
        lims = np.column_stack((xl, xu))
        # limit permutations
        lim_perms = np.zeros((n_perm, n_var))
        for perm_i, bin_perm in enumerate(bin_perms):
            for var, lim_i in enumerate(bin_perm):
                lim_perms[perm_i, var] = lims[var][lim_i]

        self.logger.info('Limit Permutations: ')
        self.logger.info(lim_perms)
        return lim_perms

    def genCornerCases(self):
        lim_perms = self.getLimPerms()
        with np.printoptions(precision=3, suppress=True):
            cornerCases = []
            for perm in lim_perms:
                with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
                    caseName = str(perm).replace(
                        '[', '').replace(']', '').replace(' ', '_')
                cornerCaseDir = os.path.join(
                    self.optDatDir, 'corner-cases', caseName)
                cornerCase = self.BaseCase(cornerCaseDir, perm)
                cornerCases.append(cornerCase)
        self.cornerCases = cornerCases

    def runCornerCases(self):
        '''
        Finds binary permutations of limits and runs these cases.

        runCornerCases(alg)
        -------------------
        Running these simulations during the optimization studies pre-processing
        provides insight into what is happening at the most extreme limits of the
        parameter space.

        These simulations may provide insight into whether the user should expand
        or restrict the parameter space. They may reveal that the model becomes
        non-physical in these extreme cases.

        Exploring the entire parameter space becomes important expecially in
        optimization studies with geometric parameters. This is due to the
        automated meshing introducting more variablility into the CFD case
        workflow.
        '''
        if self.cornerCases is None:
            self.genCornerCases()
        else:
            self.logger.warning(
                'SKIPPED: GENERATE CORNER CASES - call self.genCornerCases() directly to create new corner cases')
        self.BaseCase.parallelize(self.cornerCases)

    def runBndCases(self, n_pts=2, getDiags=False, doMeshStudy=False):
        if self.bndCases is None:
            self.genBndCases(n_pts=n_pts, getDiags=getDiags)
        else:
            self.logger.warning(
                'SKIPPED: GENERATE BOUNDARY CASES - call self.genBndCases() directly to create new boundary cases')
        self.BaseCase.parallelize(self.bndCases)
        self.saveCP()
        obj = [case.f for case in self.bndCases]
        self.plotBndPtsObj(obj)
        self.saveCP()
        if doMeshStudy:
            self.meshStudy(self.bndCases)
            self.saveCP()

    def plotBndPtsObj(self, F):
        plot = Scatter(title='Objective Space: Boundary Cases',
                       legend=True, labels=self.BaseCase.obj_labels)
        for obj in F:
            # nComp = len(obj)
            # label = '['
            # for i in range(nComp):
            #     if i != nComp-1:
            #         label += '%.2g, '%obj[i]
            #     else:
            #         label += '%.2g'%obj[i]
            # label += ']'
            # plot.add(obj, label=label)
            plot.add(np.array(obj), label=self.getPointLabel(obj))
        path = os.path.join(self.runDir, 'boundary-cases',
                            'bndPts_plot-objSpace.png')
        plot.save(path, dpi=100)
        # plot.show()
        # if F.shape[1] == 2:
        #     # self.logger.debug(f'Plotting:\n\t{str(F).replace('\n', '\n\t')}')
        #     plt.scatter(F[:,0], F[:,1])
        #     plt.suptitle('Boundary Points')
        #     plt.title('Objective Space')
        #     plt.xlabel(self.BaseCase.obj_labels[0])
        #     plt.ylabel(self.BaseCase.obj_labels[1])
        #     for pt in F:
        #         label = f'[{pt[0]}, {pt[1]}]'
        #         plt.annotate(label, pt, textcoords='offset points', xytext=(1,50), color='r')
        #     # plt.show()
        #     path = os.path.join(self.optDatDir, 'boundary-cases', 'bndPts_plot-objSpace.png')
        #     plt.savefig(path)
        #     plt.clf()
        # else:
        #     self.logger.info('Boundary points not plotted: F.shape[1] != 2')

    def plotBndPts(self, bndPts):
        plot = Scatter(title='Design Space: Boundary Cases',
                       legend=True,
                       labels=self.BaseCase.var_labels
                       # grid=True
                       )
        for var in bndPts:
            # nComp = len(var)
            # label = '['
            # for i in range(nComp):
            #     if i != nComp-1:
            #         label += '%.2g, '%var[i]
            #     else:
            #         label += '%.2g'%var[i]
            # label += ']'
            # plot.add(var, label=label)
            plot.add(np.array(var), label=self.getPointLabel(var))
        path = os.path.join(self.runDir, 'boundary-cases',
                            'bndPts_plot-varSpace.png')
        plot.save(path, dpi=100)
        # bndPts = np.array(bndPts)
        # if bndPts.shape[1] == 2:
        #     plt.scatter(bndPts[:,0], bndPts[:,1])
        #     plt.suptitle('Boundary Points')
        #     plt.title('Design Space')
        #     plt.xlabel(self.BaseCase.var_labels[0])
        #     plt.ylabel(self.BaseCase.var_labels[1])
        #     for pt in bndPts:
        #         label = f'[{pt[0]}, {pt[1]}]'
        #         plt.annotate(label, pt, textcoords='offset points', color='r')
        #     print(bndPts)
        #     plt.show()
        #     path = os.path.join(self.optDatDir, 'boundary-cases', 'bndPts_plot-varSpace.png')
        #     plt.savefig(path)
        #     plt.clf()
        # else:
        #     self.logger.info('Boundary points not plotted: bndPts.shape[1] != 2')

    def genBndCases(self, n_pts=2, getDiags=False):
        self.logger.info('GENERATING BOUNDARY CASES')
        bndPts = self.getBndPts(n_pts=n_pts, getDiags=getDiags)
        dirs = []
        for pt in bndPts:
            with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
                caseName = str(pt).replace(
                    '[', '').replace(']', '').replace(' ', '_')
            dirs.append(os.path.join(self.runDir,
                                     'boundary-cases', caseName))
        cases = self.genCases(dirs, bndPts)
        self.bndCases = cases
        self.plotBndPts(bndPts)
        self.saveCP()

    def getBndPts(self, n_pts=2, getDiags=False):
        xl = self.problem.xl
        xu = self.problem.xu
        diag = np.linspace(xl, xu, n_pts)
        lim_perms = self.getLimPerms()
        pts = []
        for diag_pt in diag:
            for perm in lim_perms:
                for diag_comp_i, diag_comp in enumerate(diag_pt):
                    for perm_comp_i, perm_comp in enumerate(perm):
                        # if diag_comp_i != perm_comp_i:
                        if diag.shape[1] >= 3 and getDiags:
                            # project diagnol onto each surface
                            temp_pt = diag_pt.copy()
                            temp_pt[perm_comp_i] = perm_comp
                            pts.append(temp_pt)
                        # project limit permutation onto corners
                        temp_pt = perm.copy()
                        temp_pt[diag_comp_i] = diag_comp
                        pts.append(temp_pt)
        pts = np.array(pts)
        pts = np.unique(pts, axis=0)
        return pts

    ###################################
    #    EXTERNAL SOLVER EXECUTION    #
    ###################################

    # def slurmExec(self, cases, batchSize=None):
    #     self.logger.info('EXECUTING BATCH OF SIMULATIONS')
    #     self.logger.info('SLURM EXECUTION')
    #     if batchSize is not None:
    #         self.logger.info(f'\t### Sending sims in batches of {batchSize}')
    #         cases_batches = [cases[i:i + batchSize]
    #                             for i in range(0, len(cases), batchSize)]
    #         for cases_batch in cases_batches:
    #             self.logger.info(f'     SUB-BATCH: {cases_batch}')
    #             self.slurmExec(cases_batch)
    #         return
    #     # Queue all the individuals in the generation using SLURM
    #     batchIDs = []  # collect batch IDs
    #     for case in cases:
    #         out = subprocess.check_output(['sbatch', case.jobFile], cwd = case.caseDir)
    #         # Extract number from following: 'Submitted batch job 1847433'
    #         # self.logger.info(int(out[20:]))
    #         batchIDs.append([int(out[20:]), case])
    #     # batchIDs = np.array(batchIDs)
    #     self.logger.info('     slurm job IDs:')
    #     self.logger.info('\t\tJob ID')
    #     self.logger.info('\t\t------')
    #     for e in batchIDs: self.logger.info(f'\t\t{e[0]} | {e[1]}')
    #     waiting = True
    #     count = np.ones(len(batchIDs))
    #     prev_count = np.ones(len(batchIDs)) #[0] #count
    #     threads = []
    #     flag = True # first time through while loop flag
    #     while waiting:
    #         time.sleep(10)
    #         for bID_i, bID in enumerate(batchIDs):
    #             # grep for batch ID of each individual
    #             out = subprocess.check_output(f'squeue | grep --count {bID[0]} || :', shell=True)  # '|| :' ignores non-zero exit status error
    #             count[bID_i] = int(out)
    #             # if simulation is no longer in squeue
    #             if count[bID_i] != prev_count[bID_i]:
    #                 ### check if simulation failed
    #                 complete = bID[1]._isExecDone()
    #                 ## if failed launch slurmExec as subprocess
    #                 if not complete:
    #                     print(f'\n\tJob ID: {bID[0]} | {bID[1]} INCOMPLETE')
    #                     t = Thread(target=self.slurmExec, args=([bID[1]],))
    #                     t.start()
    #                     threads.append(t)
    #                 else:
    #                     print(f'\n\tJobID:{bID[0]} | {bID[1]} COMPLETE', end='')
    #         ### update number of jobs waiting display
    #         if sum(count) != sum(prev_count) or flag:
    #             print(f'\n\tSimulations still running or queued = {int(sum(count))}', end='')
    #         else:
    #             print(' . ', end='')
    #         prev_count = count.copy()
    #         flag = False
    #         ### check if all batch jobs are done
    #         if sum(count) == 0:
    #             waiting = False
    #             print('\n\tDONE WAITING')
    #     ## wait for second run of failed cases to complete
    #     for thread in threads:
    #         thread.join()
    #     print()
    #     print('BATCH OF SLURM SIMULATIONS COMPLETE')
    #
    # def singleNodeExec(self, cases): #, procLim=procLim, nProc=nProc, solverFile=solverFile):
    #     print('EXECUTING BATCH OF SIMULATIONS')
    #     print('SINGLE NODE EXECUTION')
    #     # All processors will be queued until all are used
    #     n = 0
    #     currentP = 0
    #     procs = []
    #     proc_labels = {}
    #     n_sims = len(cases)
    #
    #     while n < n_sims:
    #         case = cases[n]
    #         caseDir = case.caseDir
    #         nProc = case.nProc
    #         if currentP + nProc <= self.procLim: # currentP != procLim:
    #             print(f'\t## Sending {caseDir} to simulation...')
    #             proc = subprocess.Popen(case.solverExecCmd, # '>', 'output.dat'],
    #                                    cwd = caseDir, stdout=subprocess.DEVNULL) #stdout=out)
    #             # Store the proc of the above process
    #             procs.append(proc)
    #             # store working directory of process
    #             proc_labels[proc.pid] = case.caseDir
    #             # counters
    #             n += 1
    #             currentP = currentP + nProc
    #         # Then, wait until completion and fill processors again
    #         else:
    #             print('\tWAITING')
    #             # wait for any processes to complete
    #             waiting = True
    #             while waiting:
    #                 # check all processes for completion every _ seconds
    #                 time.sleep(10)
    #                 for proc in procs:
    #                     if proc.poll() is not None:
    #                         # remove completed job from procs list
    #                         procs.remove(proc)
    #                         # reduce currentP by nProc
    #                         currentP -= nProc
    #                         # stop waiting
    #                         waiting = False
    #                         print('\tCOMPLETED: ', proc_labels[proc.pid])
    #     # Wait until all PID in the list has been completed
    #     print('\tWAITING')
    #     for proc in procs:
    #         proc.wait()
    #     print('BATCH OF SIMULATIONS COMPLETE')

    ########################
    #    HELPER METHODS    #
    ########################
    def appendMeshSFs(self, case, addedMeshSFs):
        try:
            self.logger.info(f'\tAppending Mesh Size Factors: {addedMeshSFs}')
            self.logger.info(f'\t   To Case - {case}')
            newMeshSFs = np.append(case.meshSFs, addedMeshSFs)
            case.meshSFs = newMeshSFs
            case.saveCP()
        except AttributeError as err:
            self.logger.error(str(err))

    def genCases(self, paths, X):  # , baseCase=None):
        # if baseCase is None:
        #     baseCase = self.BaseCase
        assert len(paths) == len(X), 'genCases(paths, X): len(paths) == len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.BaseCase(paths[x_i], x)
            cases.append(case)
        return cases

    @staticmethod
    def getPointLabel(pt):
        nComp = len(pt)
        label = '['
        for i in range(nComp):
            if i != nComp - 1:
                label += '%.2g, ' % pt[i]
            else:
                label += '%.2g' % pt[i]
        label += ']'
        return label

    @staticmethod
    def loadCases(directory):
        '''
        Parameter: directory - searches every directory in given directory for
                                case.npy file and tries to load if it exists.
        '''
        cases = []
        ents = os.listdir(directory)
        for ent in ents:
            ent_path = os.path.join(directory, ent)
            if os.path.isdir(ent_path):
                for e in os.listdir(ent_path):
                    caseCP = os.path.join(ent_path, 'case.npy')
                    if os.path.exists(caseCP):
                        case, = np.load(caseCP, allow_pickle=True).flatten()
                        cases.append(case)
        return cases

    #####################
    #     PROPERTIES    #
    #####################
    @property
    def runDir(self):
        return self._runDir

    @runDir.setter
    def runDir(self, runDir):
        head, tail = os.path.split(runDir)
        if head == self.optDatDir:
            self.logger.debug('runDir setter: head == self.optDatDir')
            self._runDir = os.path.join(self.optDatDir, tail)
        else:
            self._runDir = os.path.join(self.optDatDir, runDir)
        os.makedirs(self._runDir, exist_ok=True)

    @property
    def plotDir(self):
        plotDir = os.path.join(self.runDir, 'plots')
        os.makedirs(plotDir, exist_ok=True)
        return plotDir

    @property
    def mapDir(self):
        mapDir = os.path.join(self.runDir, 'mapGen1')
        os.makedirs(mapDir, exist_ok=True)
        return mapDir

    @property
    def archiveDir(self):
        archiveDir = os.path.join(self.runDir, 'archive')
        os.makedirs(archiveDir, exist_ok=True)
        return archiveDir

    @property
    def pfDir(self):
        pfDir = os.path.join(self.runDir, 'pareto_front')
        os.makedirs(pfDir, exist_ok=True)
        return pfDir
    # @property
    # def algorithm(self):
    #     return self._algorithm
    #
    # @algorithm.setter
    # def algorithm(self, alg):
    #     path = os.path.join(self.optDatDir, 'algorithm.npy')
    #     np.save(path, alg, allow_pickle=True)
    #     self._algorithm = alg
    #
    # @property
    # def gen1Pop(self):
    #     return self._gen1Pop
    #     # path = os.path.join(self.optDatDir, 'gen1Pop.npy')
    #     # gen1Pop = np.load(path, allow_pickle=True).flatten()
    #     # return gen1Pop
    #
    # @gen1Pop.setter
    # def gen1Pop(self, cases):
    #     if cases is not None:
    #         for case in cases:
    #             if isinstance(case, self.BaseCase):
    #                 case.saveCP()
    #     # path = os.path.join(self.optDatDir, 'gen1Pop.npy')
    #     # np.save(path, cases, allow_pickle=True)
    #     self._gen1Pop = cases
    #
    # @property
    # def cornerCases(self):
    #     return self._cornerCases
    #     # path = os.path.join(self.optDatDir, 'cornerCases.npy')
    #     # cornerCases = np.load(path, allow_pickle=True).flatten()
    #     # return cornerCases
    #
    # @cornerCases.setter
    # def cornerCases(self, cases):
    #     if cases is not None:
    #         for case in cases:
    #             if isinstance(case, self.BaseCase):
    #                 case.saveCP()
    #     # path = os.path.join(self.optDatDir, 'cornerCases.npy')
    #     # np.save(path, cases, allow_pickle=True)
    #     self._cornerCases = cases
    #
    #
    # @property
    # def bndCases(self):
    #     return self._bndCases
    #     # path = os.path.join(self.optDatDir, 'bndCases.npy')
    #     # bndCases = np.load(path, allow_pickle=True).flatten()
    #     # return bndCases
    #
    # @bndCases.setter
    # def bndCases(self, cases):
    #     if cases is not None:
    #         for case in cases:
    #             if isinstance(case, self.BaseCase):
    #                 case.saveCP()
    #     # path = os.path.join(self.optDatDir, 'bndCases.npy')
    #     # np.save(path, cases, allow_pickle=True)
    #     self._bndCases = cases
    #
    # @property
    # def testCase(self):
    #     return self._testCase
    #     # path = os.path.join(self.optDatDir, 'testCase.npy')
    #     # testCase = np.load(path, allow_pickle=True).flatten()
    #     # return testCase
    #
    # @testCase.setter
    # def testCase(self, case):
    #     if isinstance(case, self.BaseCase):
    #         case.saveCP()
    #     self._testCase = case
        # path = os.path.join(self.optDatDir, 'testCase.npy')
        # np.save(path, case, allow_pickle=True)

    # ==========================================================================
    # TO BE OVERWRITTEN
    # ==========================================================================
    def _execPop(self, cases):
        pass

    def _meshStudy(self):
        pass

    def _preProc(self):
        pass

    def _postProc(self):
        pass

    def normalize(self, cases):
        pass

    #####################################
    #######  EXTERNAL CFD SOLVER  #######
    #####################################
    ####### OPTION ########
    # 1) slurm / multiple nodes
    # 2) local / single node
    # def execute(self, paths):
    #     from pymooCFD.execSimsBatch import singleNodeExec
    #     singleNodeExec(paths)
    # def execute(self, cases):
    #     if self.client is None:
    #         self.singleNodeExec(cases)
    #     else:
    #         ## Use client to parallelize
    #         jobs = [self.client.submit(case.run) for case in cases]
    #         # obj = np.row_stack([job.result() for job in jobs])
    #

    # def meshStudy(self, cases): #, meshSFs=None):
    #     # if meshSFs is None:
    #     #     meshSFs = self.meshSFs
    #     for case in cases:
    #         _, tail = os.path.split(case.caseDir)
    #         meshStudyDir = os.path.join(case.studyDir, tail)
    #         print(f'MESH STUDY - {meshStudyDir}')
    #         print(f'\t{case.meshSFs}')
    #         ### Pre-Process
    #         study = []
    #         var = []
    #         a_numElem = []
    #         msCases = []
    #         for sf in case.meshSFs:
    #             ### Deep copy case instance
    #             msCase = copy.deepcopy(case)
    #             msCases.append(msCase)
    #             msCase.caseDir = os.path.join(meshStudyDir, f'meshSF-{sf}')
    #             msCase.meshSF = sf
    #             ### only pre-processing needed is generating mesh
    #             numElem = msCase.genMesh()
    #             a_numElem.append(numElem)
    #             np.savetxt(os.path.join(msCase.caseDir, 'numElem.txt'), [numElem])
    #             study.append([msCase.caseDir, str(numElem)])
    #             var.append(msCase.x)
    #         print(f'\t{a_numElem}')
    #         # study = np.array(study)
    #         print(study)
    #         path = os.path.join(meshStudyDir, 'study.txt')
    #         np.savetxt(path, study, fmt="%s")
    #         path = os.path.join(meshStudyDir, 'studyX.txt')
    #         np.savetxt(path, var)
    #         ### Execute
    #         self.execute(msCases)
    #         ### Post-Process
    #         obj = []
    #         for msCase_i, msCase in enumerate(msCases):
    #             f = msCase.postProc()
    #             obj.append(f)
    #             study[msCase_i].extend(f)
    #         obj = np.array(obj)
    #         path = os.path.join(meshStudyDir, 'studyF.txt')
    #         np.savetxt(path, obj)
    #         print('\t' + str(study).replace('\n', '\n\t'))
    #         #path = os.path.join(meshStudyDir, 'study.txt')
    #         #np.savetxt(path, study)
    #
    #         ##### PLOT #####
    #         print('obj', obj)
    #         for obj_i, obj_label in enumerate(case.obj_labels):
    #              plt.plot(a_numElem, obj[:, obj_i])
    #              plt.suptitle('Mesh Sensitivity Study')
    #              plt.title(tail)
    #              plt.xlabel('Number of Elements')
    #              plt.ylabel(obj_label)
    #              plt.savefig(os.path.join(studyDir, f'meshStudy_plot-{tail}-{obj_label}.png'))
    #              plt.clf()
    #
    #     return study
