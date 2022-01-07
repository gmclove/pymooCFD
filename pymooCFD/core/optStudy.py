# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:28:45-05:00
# import time
import os
import numpy as np
# import subprocess
# from threading import Thread
import multiprocessing as mp
import copy
import shutil
# import h5py
# import matplotlib.pyplot as plt
from pymooCFD.util.sysTools import emptyDir, copy_and_overwrite, saveTxt


class OptStudy:
    def __init__(self, algorithm, problem, BaseCase,
                 archiveDir='archive', n_CP=10, n_opt=20,
                 optDatDir='opt_run', CP_fName='checkpoint',
                 pfDir='pareto_front', baseCaseDir='base_case',
                 procOptDir='procOpt', plotsDir='plots',
                 mapGen1Dir='mapGen1', meshStudyDir='meshStudy',
                 meshSFs=np.arange(0.5, 1.5, 0.1),
                 # procLim=os.cpu_count(),
                 var_labels=None, obj_labels=None,
                 # client=None,
                 *args, **kwargs
                 ):
        super().__init__()
        #############################
        #    Required Attributes    #
        #############################
        self.problem = problem
        self.algorithm = algorithm  # algorithm.setup(problem)
        # initialize baseCase
        self.BaseCase = BaseCase
        self.baseCaseDir = baseCaseDir
        # self.genTestCase()
        # Variable and Objective Labels are passed to self.BaseCase instance
        if var_labels is None:
            self.var_labels = [
                f'var{x_i}' for x_i in range(self.problem.n_var)]
        else:
            self.var_labels = var_labels
        if obj_labels is None:
            self.obj_labels = [
                f'obj{x_i}' for x_i in range(self.problem.n_obj)]
        else:
            self.obj_labels = obj_labels
        self.BaseCase.var_labels = self.var_labels
        self.BaseCase.obj_labels = self.obj_labels
        #####################################
        #    Default/Optional Attributes    #
        #####################################
        ### Data Handling ###
        self.archiveDir = archiveDir
        os.makedirs(self.archiveDir, exist_ok=True)
        self.n_CP = n_CP  # number of generations between extra checkpoints
        self.optDatDir = optDatDir
        os.makedirs(self.optDatDir, exist_ok=True)
        self.CP_path = os.path.join(optDatDir, CP_fName)
        ### Optimization Pre/Post Processing ###
        self.procOptDir = procOptDir
        os.makedirs(self.procOptDir, exist_ok=True)
        # Pareto Front Directory
        # directory to save optimal solutions (a.k.a. Pareto Front)
        self.pfDir = pfDir
        self.n_opt = n_opt  # number of optimal points along Pareto front to save
        # Plots Directory
        self.plotDir = os.path.join(self.procOptDir, plotsDir)
        os.makedirs(self.plotDir, exist_ok=True)
        # Mapping Objectives vs. Variables Directory
        self.mapDir = os.path.join(self.procOptDir, mapGen1Dir)
        os.makedirs(self.mapDir, exist_ok=True)
        #### Mesh Sensitivity Studies ###
        self.studyDir = os.path.join(procOptDir, meshStudyDir)
        os.makedirs(self.studyDir, exist_ok=True)
        self.meshSFs = meshSFs  # mesh size factors
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

    def run(self, restart=True):
        if restart:
            self.loadCP()
            print("Loaded Checkpoint:", self.algorithm)
            print(
                f'Last checkpoint at generation {self.algorithm.callback.gen}')
            # restart client if being used
            if self.client is not None:
                self.client.restart()
                print("CLIENT RESTARTED")
        else:
            print('STARTING NEW OPTIMIZATION STUDY')
            # archive/empty previous runs data directory
            emptyDir(self.optDatDir)
            # load algorithm defined in setupOpt.py module
            self.algorithm.setup(self.problem,
                                 seed=self.algorithm.seed,
                                 verbose=self.algorithm.verbose,
                                 save_history=self.algorithm.save_history,
                                 return_least_infeasible=self.algorithm.return_least_infeasible
                                 )
            # start client if being used
            if self.client is not None:
                self.client()
                print("CLIENT STARTED")
        ######    OPTIMIZATION    ######
        # until the algorithm has not terminated
        while self.algorithm.has_next():
            # First generation
            # population is None so ask for new pop
            if self.algorithm.pop is None:
                print('     START-UP: first generation')
                evalPop = self.algorithm.ask()
                self.algorithm.pop = evalPop
                self.algorithm.off = evalPop
            # Previous generation complete
            # If current objective does not have None values then get new pop
            # ie previous pop is complete
            # evaluate new pop
            elif None not in self.algorithm.pop.get('F'):
                print('     START-UP: new generation')
                evalPop = self.algorithm.ask()
                self.algorithm.off = evalPop
            # Mid-generation start-up
            # evaluate offspring population
            else:
                print('     START-UP: mid-generation')
                evalPop = self.algorithm.off
            # save checkpoint before evaluation
            self.saveCP()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            self.algorithm.evaluator.eval(self.problem, evalPop)
            # returned the evaluated individuals which have been evaluated or even modified
            # checkpoint saved after in algorithm.callback.notify() method
            self.algorithm.tell(infills=evalPop)
            # save top {n_opt} optimal evaluated cases in pf directory
            for off_i, off in enumerate(self.algorithm.off):
                for opt_i, opt in enumerate(self.algorithm.opt[:self.n_opt]):
                    if np.array_equal(off.X, opt.X):
                        optDir = os.path.join(
                            self.optDatDir, self.pfDir, f'opt{opt_i+1}')
                        offDir = os.path.join(
                            self.optDatDir, f'gen{self.algorithm.n_gen}', f'ind{off_i+1}')
                        print(
                            f'     Updating Pareto front folder: {offDir} -> {optDir}')
                        copy_and_overwrite(offDir, optDir)
            # do some more things, printing, logging, storing or even modifying the algorithm object
            # print(algorithm.n_gen, algorithm.evaluator.n_eval)
            # print('Parameters:')
            # print(algorithm.pop.get('X'))
            # print('Objectives:')
            # print(algorithm.pop.get('F'))
            # algorithm.display.do(algorithm.problem,
            #                      algorithm.evaluator,
            #                      algorithm
            #                      )
        # obtain the result objective from the algorithm
        res = self.algorithm.result()
        # calculate a hash to show that all executions end with the same result
        print("hash", res.F.sum())

    def runGen1(self, restart=True):
        if not restart or self.gen1Pop is None:
            alg = copy.deepcopy(self.algorithm)
            alg.setup(self.problem, ('n_gen', 1))
            alg.next()
        else:
            print('self.gen1Pop already exists')

    ####################
    #    MESH STUDY    #
    ####################
    def meshStudy(self, cases):
        for case in cases:
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
    def saveCP(self):
        np.save(self.CP_path + '.temp.npy', self)
        if os.path.exists(self.CP_path + '.npy'):
            os.rename(self.CP_path + '.npy', self.CP_path + '.old.npy')
        os.rename(self.CP_path + '.temp.npy', self.CP_path + '.npy')
        if os.path.exists(self.CP_path + '.old.npy'):
            os.remove(self.CP_path + '.old.npy')

    def loadCP(self):
        if os.path.exists(self.CP_path + '.old'):
            os.rename(self.CP_path + '.old', self.CP_path + '.npy')
        cp, = np.load(self.CP_path + '.npy', allow_pickle=True).flatten()
        self.__dict__.update(cp.__dict__)
        self.logger.info(f'\tCHECKPOINT LOADED - from {self.CP_path}.npy')

    def loadCP(self, hasTerminated=False):
        try:
            if os.path.exists(self.CP_path + '.old'):
                os.rename(self.CP_path + '.old', self.CP_path + '.npy')
            cp, = np.load(self.CP_path + '.npy', allow_pickle=True).flatten()
            self.__dict__.update(cp.__dict__)
            print(f'\tCHECKPOINT LOADED - from {self.CP_path}.npy')
            # self.logger.info(f'\tCHECKPOINT LOADED - from {self.CP_path}.npy')
            # only necessary if for the checkpoint the termination criterion has been met
            self.algorithm.has_terminated = hasTerminated
            alg = self.algorithm
            # Update any changes made to the algorithms between runs
            alg.termination = self.algorithm.termination
            alg.pop_size = self.algorithm.pop_size
            alg.n_offsprings = self.algorithm.n_offsprings
            alg.problem.xl = np.array(self.problem.xl)
            alg.problem.xu = np.array(self.problem.xu)
            self.algorithm = alg
        except FileNotFoundError as err:
            print(err)
            raise Exception(f'{self.CP_path} load failed.')

    def saveCP(self):  # , alg=None):
        # # default use self.algorithm
        # if alg is None:
        #     alg = self.algorithm
        # # if give alg assign it to self.algorithm
        # else:
        #     self.algorithm = alg
        gen = self.algorithm.callback.gen
        print(f'SAVING CHECKPOINT - GENERATION {gen}')
        np.save(self.CP_path + '.temp.npy', self)
        if os.path.exists(self.CP_path + '.npy'):
            os.rename(self.CP_path + '.npy', self.CP_path + '.old.npy')
        os.rename(self.CP_path + '.temp.npy', self.CP_path + '.npy')
        if os.path.exists(self.CP_path + '.old.npy'):
            os.remove(self.CP_path + '.old.npy')
        # genDir = f'gen{gen}'
        # os.path.join(optDatDir, 'checkpoint')
        # np.save(self.CP_path, alg)
        # gen0 and every nCP generations save additional static checkpoint
        if gen % self.n_CP == 1:
            np.save(os.path.join(self.optDatDir, f'checkpoint-gen{gen}'), self)
        # save text file of variables and objectives as well
        # this provides more options for post-processesing data
        genX = self.algorithm.pop.get('X')
        saveTxt(self.optDatDir, f'gen{gen}X.txt', genX)
        try:
            genF = self.algorithm.pop.get('F')
            saveTxt(self.optDatDir, f'gen{gen}F.txt', genF)
            # path = os.path.join(self.optDatDir, f'gen{gen}F.txt')
            # np.savetxt(path, genF)
        except TypeError:  # AttributeError
            print('\tmid-generation')

    # def saveCP(self, alg=None):
    #     # default use self.algorithm
    #     if alg is None:
    #         alg = self.algorithm
    #     # if give alg assign it to self.algorithm
    #     else:
    #         self.algorithm = alg
    #     gen = alg.callback.gen
    #     print(f'SAVING CHECKPOINT - GENERATION {gen}')
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
    #         print('\tmid-generation')

    ###################
    #    TEST CASE    #
    ###################
    def genTestCase(self, testCaseDir='test_case'):
        # shutil.rmtree('test_case', ignore_errors=True)
        xl = self.problem.xl
        xu = self.problem.xu
        x_mid = [xl[x_i] + (xu[x_i] - xl[x_i]) /
                 2 for x_i in range(self.problem.n_var)]
        self.testCase = self.BaseCase(
            self.baseCaseDir, testCaseDir, x_mid)  # , restart=True)

    def runTestCase(self):
        print('TEST CASE RUNNING')
        if self.testCase is None:
            self.genTestCase()
        self.testCase.run()
        print('Parameters:', self.testCase.x)
        print('Objectives:', self.testCase.f)
        print('TEST CASE COMPLETE ')
        return self.testCase

    #################################
    #    RUN POPULATION OF CASES    #
    #################################
    def runGen(self, X, out):
        gen = self.algorithm.callback.gen
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(self.optDatDir, f'gen{gen}')
        # create sub-directories for each individual
        indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
        cases = self.genCases(indDirs, X)
        self.BaseCase.parallelize(cases)
        # self.runPop(cases)
        out['F'] = [case.f for case in cases]
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

        print('Limit Permutations: ')
        print(lim_perms)
        return lim_perms

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
        lim_perms = self.getLimPerms()
        with np.printoptions(precision=3, suppress=True):
            cornerCases = []
            for perm in lim_perms:
                with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
                    caseName = str(perm).replace(
                        '[', '').replace(']', '').replace(' ', '_')
                cornerCaseDir = os.path.join(
                    self.procOptDir, 'corner-cases', caseName)
                cornerCase = self.BaseCase(
                    self.baseCaseDir, cornerCaseDir, perm)
                cornerCases.append(cornerCase)
        self.BaseCase.parallelize(cornerCases)

    def runBndCases(self, n_pts, getDiags=False, doMeshStudy=False):
        self.genBndCases(n_pts, getDiags=getDiags)
        self.runPop(self.bndCases)
        if doMeshStudy:
            for case in self.bndCases:
                case.meshStudy()

    def genBndCases(self, n_pts, getDiags=False):
        bndPts = self.getBndPts(n_pts, getDiags=getDiags)
        dirs = []
        for pt in bndPts:
            with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
                caseName = str(pt).replace(
                    '[', '').replace(']', '').replace(' ', '_')
            dirs.append(os.path.join(self.procOptDir,
                                     'boundary-cases', caseName))
        cases = self.genCases(dirs, bndPts)
        self.bndCases = cases

    def getBndPts(self, n_pts, getDiags=False):
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
    #     print('EXECUTING BATCH OF SIMULATIONS')
    #     print('SLURM EXECUTION')
    #     if batchSize is not None:
    #         print(f'\t### Sending sims in batches of {batchSize}')
    #         cases_batches = [cases[i:i + batchSize]
    #                             for i in range(0, len(cases), batchSize)]
    #         for cases_batch in cases_batches:
    #             print(f'     SUB-BATCH: {cases_batch}')
    #             self.slurmExec(cases_batch)
    #         return
    #     # Queue all the individuals in the generation using SLURM
    #     batchIDs = []  # collect batch IDs
    #     for case in cases:
    #         out = subprocess.check_output(['sbatch', case.jobFile], cwd = case.caseDir)
    #         # Extract number from following: 'Submitted batch job 1847433'
    #         # print(int(out[20:]))
    #         batchIDs.append([int(out[20:]), case])
    #     # batchIDs = np.array(batchIDs)
    #     print('     slurm job IDs:')
    #     print('\t\tJob ID')
    #     print('\t\t------')
    #     for e in batchIDs: print(f'\t\t{e[0]} | {e[1]}')
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
    def genCases(self, paths, X):  # , baseCase=None):
        # if baseCase is None:
        #     baseCase = self.BaseCase
        assert len(paths) == len(X), 'genCases(paths, X): len(paths) == len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.BaseCase(self.baseCaseDir, paths[x_i], x)
            cases.append(case)
        return cases

    def loadCases(directory):
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
    def gen1Pop(self):
        path = os.path.join(self.optDatDir, 'gen1Pop.npy')
        gen1Pop = np.load(path, allow_pickle=True).flatten()
        return gen1Pop

    @gen1Pop.setter
    def gen1Pop(self, cases):
        path = os.path.join(self.optDatDir, 'gen1Pop.npy')
        np.save(path, cases, allow_pickle=True)

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
