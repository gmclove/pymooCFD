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
from scipy.stats import linregress
# import subprocess
# from threading import Thread
# import multiprocessing as mp
import copy
# import shutil
# import h5py
# import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.core.evaluator import set_cv
from pymooCFD.core.pymooBase import CFDGeneticProblem
from pymooCFD.util.sysTools import emptyDir, copy_and_overwrite, saveTxt, yes_or_no
from pymooCFD.util.loggingTools import MultiLineFormatter, DispNameFilter
from pymooCFD.core.picklePath import PicklePath
import pymooCFD.config as config
from pymoo.util.misc import termination_from_tuple


class OptStudy(PicklePath):
    def __init__(self, algorithm, problem,
                 run_path='run-defualt',
                 n_opt=20,
                 # restart=True,
                 # optDatDir='opt_run',
                 # optName=None,
                 # n_CP=10,
                 # CP_fName='checkpoint',
                 # plotsDir='plots', archiveDir='archive', mapGen1Dir='mapGen1',
                 # procLim=os.cpu_count(),
                 # var_labels=None, obj_labels=None,
                 # client=None,
                 # *args, **kwargs
                 ):
        super().__init__(dir_path=run_path)
        #######################################
        #    ATTRIBUTES NEEDED FOR RESTART    #
        #######################################
        # self.logger = self.getLogger()
        self.logger.info(f'OPTIMIZATION STUDY- {self.rel_path}')
        # self.cp_path = os.path.join(self.optDatDir, f'{self.optName}-CP')
        self.logger.info(f'\tCheckpoint Path: {self.cp_path}')

        # if not restart:
        # if os.path.exists(self.optDatDir):
        #     if os.path.exists(self.cp_path):
        #         self.loadCP()
        #         self.logger.info('RESTARTED FROM', self.cp_path)
        #         return
        #     else:
        #         question = f'{self.cp_path} does not exist.\nOVERWRITE {self.optDatDir}?'
        #         overwrite = yes_or_no(question)
        #         if overwrite:
        #             os.removedirs(self.optDatDir)
        #             os.makedirs(optDatDir, exist_ok=True)
        #         else:
        #             question = f'{self.cp_path} does not exist.\n?'
        #             overwrite = yes_or_no(question)

        # if os.path.isdir(self.optDatDir):
        #     try:
        #         self.loadCP()
        #         self.logger.info('RESTARTED FROM CHECKPOINT')
        #         return
        #     except FileNotFoundError:
        #         question = f'\n{self.cp_path} does not exist.\nEMPTY {self.optDatDir} DIRECTORY?'
        #         overwrite = yes_or_no(question)
        #         if overwrite:
        #             shutil.rmtree(self.optDatDir)
        #             os.mkdir(self.optDatDir)
        #             self.logger.info(f'EMPTIED - {self.optDatDir}')
        #         else:
        #             self.logger.info(f'KEEPING FILES - {self.optDatDir}')
        #         self.logger.info('RE-INITIALIZING OPTIMIZATION ALGORITHM')
        # else:
        #     os.makedirs(self.optDatDir)
        #     self.logger.info(
        #         f'NEW OPTIMIZATION STUDY - {self.optDatDir} did not exist')
            # except FileNotFoundError:
            #     print('OVERRIDE OPTIMIZATION STUDY -')
            #     print('\t{self.cp_path} already exists but {self.cpPath} does not')
            #     self.copy()
        # else:
        #     os.makedirs(caseDir, exist_ok=True)
        #     self.logger = self.getLogger()
        #     self.logger.info(f'NEW CASE - {caseDir} did not exist')
        #     self.copy()
        #############################
        #    Required Attributes    #
        #############################
        self.problem = problem #CFDProblem_GA(BaseCase)
        #assert algorithm is setup
        self.algorithm = algorithm  # algorithm.setup(problem)
        # self.BaseCase = BaseCase

        #####################################
        #    Default/Optional Attributes    #
        #####################################
        self.run_path = run_path
        # self.run_path = os.path.join(self.optDatDir, run_path)
        # os.makedirs(self.run_path, exist_ok=True)

        ### Data Handling ###
        # self.archiveDir = os.path.join(self.optDatDir, archiveDir)
        # os.makedirs(self.archiveDir, exist_ok=True)
        # self.n_CP = n_CP  # number of generations between extra checkpoints

        ### Optimization Pre/Post Processing ###
        # self.procOptDir = os.path.join(self.optDatDir, procOptDir)
        # os.makedirs(self.procOptDir, exist_ok=True)
        # Pareto Front Directory
        # directory to save optimal solutions (a.k.a. Pareto Front)
        # self.pfDir = os.path.join(self.run_path, pfDir)
        # os.makedirs(self.pfDir, exist_ok=True)
        # number of optimal points along Pareto front to save
        self.n_opt = int(n_opt)

        ###################################
        #    Attributes To Be Set Later   #
        ###################################
        self.gen1Pop = None
        self.cornerCases = None
        self.bndCases = None
        ### Test Case ###
        self.testCase = None
        self.genTestCase()
        self.save_self()
        # self.save_self()

    # def newProb(self, BaseCase):
    #     self.logger.info('INITIALIZING NEW OPTIMIZATION PROBLEM')
    #     self.problem = CFDProblem_GA(BaseCase)
    #     self.save_self()
    #     return self.problem

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
        self.save_self()

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
        # restart client if being used
        # if self.client is not None:
        #     self.client.restart()
        #     self.logger.info("CLIENT RESTARTED")

        ######    OPTIMIZATION    ######
        # until the algorithm has not terminated
        while self.algorithm.has_next():
            # First generation
            # population is None so ask for new pop
            if self.algorithm.pop is None:
                self.logger.info('\tSTART-UP: first generation')
                evalPop = self.algorithm.ask()
                self.algorithm.pop = evalPop
                self.algorithm.off = evalPop
            # Previous generation complete
            # If current objective does not have None values then get new pop
            # ie previous pop is complete evaluate new pop
            elif self.algorithm.off is None:
                self.logger.info('\tSTART-UP: new generation')
                evalPop = self.algorithm.ask()
                self.algorithm.off = evalPop
            # Mid-generation start-up
            # evaluate offspring population
            else:
                self.logger.info('\tSTART-UP: mid-generation')
                evalPop = self.algorithm.off
            # save checkpoint before evaluation
            self.save_self()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            evalPop = self.algorithm.evaluator.eval(self.problem, evalPop,
                                                run_path=self.run_path,
                                                gen=self.algorithm.callback.gen
                                                # alg=self.algorithm
                                                )
            # self.algorithm.evaluator.eval(self.problem, evalPop)
            # evalPop = self.runGen(evalPop)
            # print('self.algorithm.callback.gen:', self.algorithm.callback.gen)
            # print('self.algorithm.n_gen:', self.algorithm.n_gen)

            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=evalPop)

            # save top {n_opt} optimal evaluated cases in pf directory
            compGen = self.algorithm.callback.gen - 1
            for off_i, off in enumerate(self.algorithm.off):
                for opt_i, opt in enumerate(self.algorithm.opt[:self.n_opt]):
                    if np.array_equal(off.X, opt.X):
                        optDir = os.path.join(self.pfDir, f'opt{opt_i+1}')
                        offDir = os.path.join(
                            self.run_path, f'gen{compGen}', f'ind{off_i+1}')
                        self.logger.info(
                            f'\tUpdating Pareto front folder: {offDir} -> {optDir}')
                        try:
                            copy_and_overwrite(offDir, optDir)
                        except FileNotFoundError as err:
                            self.logger.error(str(err))
                            self.logger.warning('SKIPPED: UPDATE PARETO FRONT')
            # do some more things, printing, logging, storing or even modifying the algorithm object
            self.algorithm.off = None
            self.save_self()
            if delPrevGen and not compGen == 1:
                direct = os.path.join(
                    self.run_path, f'gen{compGen}')
                shutil.rmtree(direct)
        # obtain the result objective from the algorithm
        res = self.algorithm.result()
        # calculate a hash to show that all executions end with the same result
        self.logger.info(f'hash {res.F.sum()}')
        # self.save_self()

    def execGen1(self):
        self.algorithm.termination = termination_from_tuple(('n_gen', 1))
        # no checkpoint saved before evaluation
        # therefore self.algorithm not stuck with after gen.1 termination criteria
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
        self.save_self()

    ################
    #    LOGGER    #
    ################

    # def getLogger(self):
    #     # create root logger
    #     logger = logging.getLogger()
    #     logger.setLevel(config.OPT_STUDY_LOGGER_LEVEL)
    #     # define handlers
    #     # if not logger.handlers:
    #     # file handler
    #     fileHandler = logging.FileHandler(f'{self.optName}.log')
    #     logger.addHandler(fileHandler)
    #     # stream handler
    #     streamHandler = logging.StreamHandler()  # sys.stdout)
    #     streamHandler.setLevel(logging.DEBUG)
    #     if streamHandler not in logger.handlers:
    #         logger.addHandler(streamHandler)
    #     # define filter
    #     # filt = DispNameFilter(self.optName)
    #     # logger.addFilter(filt)
    #     # define formatter
    #     formatter = MultiLineFormatter(
    #         '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s')
    #     #     f'%(asctime)s :: %(levelname)-8s :: {self.optName} :: %(message)s')
    #     #     f'%(asctime)s.%(msecs)03d :: %(levelname)-8s :: {self.optName} :: %(message)s')
    #     #     '%(name)s :: %(levelname)-8s :: %(message)s')
    #     fileHandler.setFormatter(formatter)
    #     streamHandler.setFormatter(formatter)
    #     logger.info('~' * 30)
    #     logger.info('NEW RUN')
    #     return logger

    ####################
    #    MESH STUDY    #
    ####################
    # def meshStudy(self, cases):
    #     for case in cases:
    #         self.logger.info(f'MESH STUDY - {case}')
    #         case.meshStudy()

    #######################
    #    CHECKPOINTING    #
    #######################
<<<<<<< HEAD
    # def loadCP(self, hasTerminated=False):
    #     if os.path.exists(self.cp_path + '.old'):
    #         os.rename(self.cp_path + '.old', self.cp_path + '.npy')
    #     cp, = np.load(self.cp_path + '.npy', allow_pickle=True).flatten()
    #
    #     # logging
    #     self.logger.info(f'\tCHECKPOINT LOADED: {self.cp_path}.npy')
    #     self.logger.debug('\tRESTART DICTONARY')
    #     for key in self.__dict__:
    #         self.logger.debug(f'\t\t{key}: {self.__dict__[key]}')
    #     self.logger.debug('\tCHECKPOINT DICTONARY')
    #     for key in cp.__dict__:
    #         self.logger.debug(f'\t\t{key}: {cp.__dict__[key]}')
    #
    #     if cp.algorithm is not None:
    #         self.logger.debug('\tOPTIMIZATION ALGORITHM DICTONARY:')
    #         for key, val in cp.algorithm.__dict__.items():
    #             self.logger.debug(f'\t\t{key}: {val}')
    #     #### TEMPORARY CODE ##########
    #     # TRANSITION BETWEEN CHECKPOINTS
    #
    #     self.__dict__.update(cp.__dict__)
    #     # only necessary if for the checkpoint the termination criterion has been met
    #     try:
    #         self.algorithm.has_terminated = hasTerminated
    #     except AttributeError as err:
    #         self.logger.error(err)
=======
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
        #### TEMPORARY CODE ##########
        # TRANSITION BETWEEN CHECKPOINTS
        self.__dict__.update(cp.__dict__)
        # only necessary if for the checkpoint the termination criterion has been met
        try:
            self.algorithm.has_terminated = hasTerminated
        except AttributeError as err:
            self.logger.error(err)

>>>>>>> devel

    #
    # def save_self(self):  # , alg=None):
    #     gen = self.algorithm.callback.gen
    #     self.logger.info(f'SAVING CHECKPOINT - GENERATION {gen}')
    #     if self.algorithm.pop is not None:
    #         genX = self.algorithm.pop.get('X')
    #         if None not in genX:
    #             saveTxt(self.run_path, f'gen{gen}X.txt', genX)
    #         genF = self.algorithm.pop.get('F')
    #         if None not in genF:
    #             saveTxt(self.run_path, f'gen{gen}F.txt', genF)
    #         if self.algorithm.off is None:
    #             self.logger.info(f'\tgeneration {gen-1} complete')
    #
    #     elif self.algorithm.off is not None:  # and self.algorithm.pop is not None
    #         self.logger.info('\tmid-generation checkpoint')
    #     # except TypeError:
    #     #     self.logger.info('\tmid-generation')
    #     # save checkpoint
    #     np.save(self.cp_path + '.temp.npy', self)
    #     if os.path.exists(self.cp_path + '.npy'):
    #         os.rename(self.cp_path + '.npy', self.cp_path + '.old.npy')
    #     os.rename(self.cp_path + '.temp.npy', self.cp_path + '.npy')
    #     if os.path.exists(self.cp_path + '.old.npy'):
    #         os.remove(self.cp_path + '.old.npy')
    #     # Checkpoint each cfdCase object stored in optStudy
    #     try:
    #         self.testCase.save_self()
    #     except AttributeError:
    #         self.logger.debug('No Test Case to Save Checkpoint for')
    #     except FileNotFoundError:
    #         self.logger.debug('No Test Case to Save Checkpoint for')
    #     try:
    #         for case in self.bndCases:
    #             case.save_self()
    #     except TypeError:
    #         self.logger.debug('No Boundary Cases to Save Checkpoints for')
    #     except AttributeError:
    #         self.logger.debug('No Boundary Cases to Save Checkpoints for')
    #         # self.logger.error(e)
    #     try:
    #         for case in self.cornerCases:
    #             case.save_self()
    #     except TypeError:
    #         self.logger.debug('No Corner Cases to Save Checkpoints for')
    #     except AttributeError:
    #         self.logger.debug('No Corner Cases to Save Checkpoints for')
            # self.logger.error(e)

        # # default use self.algorithm
        # if alg is None:
        #     alg = self.algorithm
        # # if give alg assign it to self.algorithm
        # else:
        #     self.algorithm = alg

    # def checkCPs(self):
    #     self.logger.info(f'CHECKPOINT CHECK - {self.cp_path}.npy')
    #     if os.path.exists(f'{self.cp_path}.npy'):
    #         for cp in np.load(self.cp_path):
    #             self.logger.info(cp.__dict__)
    #     else:
    #         self.logger.info(f'\t{self.cp_path} does not exist')


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
        for x_i, var_type in enumerate(self.problem.BaseCase.var_type):
            if var_type.lower() == 'int':
                x_mid[x_i] = int(x_mid[x_i])
        testCaseDir = os.path.join(self.run_path, testCaseDir)
        self.testCase = self.problem.BaseCase(testCaseDir, x_mid)  # , restart=True)

    def runTestCase(self):
        self.logger.info('TEST CASE RUN . . .')
        # if self.testCase is None:
        self.genTestCase()
        self.logger.info('\tRUNNING TEST CASE')
        self.testCase.run()
        self.logger.info(f'\tParameters: {self.testCase.x}')
        self.logger.info(f'\tObjectives: {self.testCase.f}')
        self.logger.info('TEST CASE COMPLETE ')
        self.save_self()
        return self.testCase

    #################################
    #    RUN POPULATION OF CASES    #
    #################################
    # def runGen(self, X, out):
    #     gen = self.algorithm.callback.gen
    #     # create generation directory for storing data/executing simulations
    #     genDir = os.path.join(self.run_path, f'gen{gen}')
    #     # create sub-directories for each individual
    #     indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
    #     cases = self.genCases(indDirs, X)
    #     self.problem.BaseCase.parallelize(cases)
    #     # self.runPop(cases)
    #     for case in cases:
    #         print(case.abs_path, case.f, case.x)
    #     print(np.array([case.f for case in cases]))
    #     out['F'] = np.array([case.f for case in cases])
    #     if gen == 1:
    #         self.gen1Pop = cases
    #     return out

    def runGen(self, pop):
        # get the design space values of the algorithm
        X = pop.get("X")
        # implement your evluation
        gen = self.algorithm.callback.gen
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(self.run_path, f'gen{gen}')
        # create sub-directories for each individual
        indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
        cases = self.genCases(indDirs, X)
        self.problem.BaseCase.parallelize(cases)
        F = np.array([case.f for case in cases])
        G = np.array([case.g for case in cases])
        # objectives
        pop.set("F", F)
        # for constraints
        pop.set("G", G)
        # this line is necessary to set the CV and feasbility status - even for unconstrained
        set_cv(pop)
        if gen == 1:
            self.gen1Pop = cases
            self.mapGen1()
        self.plotGen()
        return pop

    def plotGen(self, gen=None, max_leg_len=10):
        if gen is None:
            gen = self.algorithm.n_gen
        pop = self.algorithm.history[gen-1].pop
        if len(pop) <= max_leg_len:
            leg = True
        else:
            leg = False
        #### Parameter Space Plot ####
        popX = pop.get('X')
        var_plot = Scatter(title=f'Generation {gen} Design Space',
                       legend=leg,
                       labels=self.problem.BaseCase.var_labels,
        #                figsize=(10,8)
                      )
        for ind_i, ind in enumerate(popX):
            var_plot.add(ind, label=f'IND {ind_i+1}')
        # save parameter space plot
        var_plot.save(os.path.join(self.plotDir, f'gen{gen}_obj_space.png'), dpi=100)

        #### Objective Space Plot ####
        popF = pop.get('F')
        obj_plot = Scatter(title=f'Generation {gen} Objective Space',
                       legend=leg,
                       labels=self.problem.BaseCase.obj_labels
                      )
        for ind_i, ind in enumerate(popF):
            obj_plot.add(ind, label=f'IND {ind_i+1}')
        # save parameter space plot
        obj_plot.save(os.path.join(self.plotDir, f'gen{gen}_obj_space.png'), dpi=100)
        self.logger.info(f'PLOTTED: Generation {gen} Design and Objective Spaces')
        return var_plot, obj_plot

    def mapGen1(self):
        ##### Variable vs. Objective Plots ######
        # extract objectives and variables columns and plot them against each other
        var_labels = self.problem.BaseCase.var_labels
        obj_labels = self.problem.BaseCase.obj_labels
        popX = self.algorithm.history[0].pop.get('X').astype(float)
        popF = self.algorithm.history[0].pop.get('F').astype(float)
        mapPaths = []
        plots = []
        for x_i, x in enumerate(popX.transpose()):
            for f_i, f in enumerate(popF.transpose()):
                plot = Scatter(title=f'{var_labels[x_i]} vs. {obj_labels[f_i]}',
                               labels=[var_labels[x_i], obj_labels[f_i]],
        #                        figsize=(10,8)
        #                        legend = True,
                              )
                xf = np.column_stack((x,f))
                plot.add(xf)
                ### Polynomial best fit lines
                plot.do()
                plot.legend = True
                c = ['r', 'g', 'm']
                for d in range(1, 3+1):
                    coefs = np.polyfit(x, f, d)
                    y = np.polyval(coefs, x)
                    xy = np.column_stack((x, y))
                    xy = xy[xy[:, 0].argsort()]
                    label = f'Order {d} Best Fit'
                    plot.ax.plot(xy[:,0], xy[:,1], label=label, c=c[d-1])
                plot.do()
                var_str = var_labels[x_i].replace(" ", "_").replace("/", "|").replace('%', 'precentage').replace("\\", "|")
                obj_str = obj_labels[f_i].replace(" ", "_").replace("/", "|").replace('%', 'precentage').replace("\\", "|")
                fName = f'{var_str}-vs-{obj_str}.png'
                path = os.path.join(self.mapDir, fName)
                mapPaths.append(path)
                plot.save(path, dpi=100)
                plots.append(plot)
        return plots, mapPaths

    # def runPop(self, cases):
    #     nTask = int(self.procLim/self.problem.BaseCase.nProc)
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
                    self.run_path, 'corner-cases', caseName)
                cornerCase = self.problem.BaseCase(cornerCaseDir, perm)
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
        self.problem.BaseCase.parallelize(self.cornerCases)

    def runBndCases(self, n_pts=2, getDiags=False, doMeshStudy=False):
        if self.bndCases is None:
            self.genBndCases(n_pts=n_pts, getDiags=getDiags)
        else:
            self.logger.warning(
                'SKIPPED: GENERATE BOUNDARY CASES - call self.genBndCases() directly to create new boundary cases')
        self.problem.BaseCase.parallelize(self.bndCases)
        self.save_self()
        self.plotBndPtsObj()
        if doMeshStudy:
            self.meshStudy(self.bndCases)
            self.save_self()

    def plotBndPtsObj(self):
        plot = Scatter(title='Objective Space: Boundary Cases',
                       legend=True, labels=self.problem.BaseCase.obj_labels)
        F = np.array([case.f for case in self.bndCases])
        for obj in F:
            plot.add(obj, label=self.getPointLabel(obj))
        path = os.path.join(self.run_path, 'boundary-cases',
                            'bndPts_plot-objSpace.png')
        plot.save(path, dpi=100)

    def plotBndPts(self):
        plot = Scatter(title='Design Space: Boundary Cases',
                       legend=True,
                       labels=self.problem.BaseCase.var_labels
                       # grid=True
                       )
        bndPts = np.array([case.x for case in self.bndCases])
        for var in bndPts:
            plot.add(var, label=self.getPointLabel(var))
        path = os.path.join(self.run_path, 'boundary-cases',
                            'bndPts_plot-varSpace.png')
        plot.save(path, dpi=100)

    def genBndCases(self, n_pts=2, getDiags=False):
        self.logger.info('GENERATING BOUNDARY CASES')
        bndPts = self.getBndPts(n_pts=n_pts, getDiags=getDiags)
        dirs = []
        for pt in bndPts:
            with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
                caseName = str(pt).replace(
                    '[', '').replace(']', '').replace(' ', '_')
            dirs.append(os.path.join(self.run_path,
                                     'boundary-cases', caseName))
        cases = self.genCases(dirs, bndPts)
        self.bndCases = cases
        self.plotBndPts()
        self.save_self()

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

    ########################
    #    HELPER METHODS    #
    ########################
    def appendMeshSFs(self, case, addedMeshSFs):
        try:
            self.logger.info(f'\tAppending Mesh Size Factors: {addedMeshSFs}')
            self.logger.info(f'\t   To Case - {case}')
            newMeshSFs = np.append(case.meshSFs, addedMeshSFs)
            case.meshSFs = newMeshSFs
            case.save_self()
        except AttributeError as err:
            self.logger.error(str(err))

    def genCases(self, paths, X):  # , baseCase=None):
        # if baseCase is None:
        #     baseCase = self.problem.BaseCase
        assert len(paths) == len(X), 'genCases(paths, X): len(paths) == len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.problem.BaseCase(paths[x_i], x)
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
    def loadCase(case_path):
        caseCP = os.path.join(case_path, 'case.npy')
        try:
            case, = np.load(caseCP, allow_pickle=True).flatten()
            return case
        except FileNotFoundError as err:
            print(err)


    @classmethod
    def loadCases(cls, directory):
        '''
        Parameter: directory - searches every directory in given directory for
                                case.npy file and tries to load if it exists.
        '''

        cases = []
        ents = os.listdir(directory)
        for ent in ents:
            case_path = os.path.join(directory, ent)
            if os.path.isdir(case_path):
                case = cls.loadCase(case_path)
                cases.append(case)
        return cases

    #####################
    #     PROPERTIES    #
    #####################
    # @property
    # def run_path(self):
    #     return self._run_path
    #
    # @run_path.setter
    # def run_path(self, run_path):
    #     head, tail = os.path.split(run_path)
    #     if head == self.optDatDir:
    #         self.logger.debug('run_path setter: head == self.optDatDir')
    #         self._run_path = os.path.join(self.optDatDir, tail)
    #     else:
    #         self._run_path = os.path.join(self.optDatDir, run_path)
    #     os.makedirs(self._run_path, exist_ok=True)

    @property
    def plotDir(self):
        plotDir = os.path.join(self.run_path, 'plots')
        os.makedirs(plotDir, exist_ok=True)
        return plotDir

    @property
    def mapDir(self):
        mapDir = os.path.join(self.run_path, 'mapGen1')
        os.makedirs(mapDir, exist_ok=True)
        return mapDir

    @property
    def archiveDir(self):
        archiveDir = os.path.join(self.run_path, 'archive')
        os.makedirs(archiveDir, exist_ok=True)
        return archiveDir

    @property
    def pfDir(self):
        pfDir = os.path.join(self.run_path, 'pareto_front')
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
    #             if isinstance(case, self.problem.BaseCase):
    #                 case.save_self()
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
    #             if isinstance(case, self.problem.BaseCase):
    #                 case.save_self()
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
    #             if isinstance(case, self.problem.BaseCase):
    #                 case.save_self()
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
    #     if isinstance(case, self.problem.BaseCase):
    #         case.save_self()
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
    #         _, tail = os.path.split(case.abs_path)
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
    #             msCase.abs_path = os.path.join(meshStudyDir, f'meshSF-{sf}')
    #             msCase.meshSF = sf
    #             ### only pre-processing needed is generating mesh
    #             numElem = msCase.genMesh()
    #             a_numElem.append(numElem)
    #             np.savetxt(os.path.join(msCase.abs_path, 'numElem.txt'), [numElem])
    #             study.append([msCase.abs_path, str(numElem)])
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
