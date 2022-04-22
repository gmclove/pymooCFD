# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:28:45-05:00
# import time
import os
import numpy as np
import shutil
# import subprocess
# from threading import Thread
# import multiprocessing as mp
# import shutil
# import h5py
# import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymooCFD.util.sysTools import copy_and_overwrite, saveTxt
from pymooCFD.core.picklePath import PicklePath


class OptRun(PicklePath):
    def __init__(self, algorithm, problem,
                 run_path='run-defualt',
                 n_opt=20,
                 # restart=True,
                 # optDatDir='opt_run',
                 # optName=None,
                 # procLim=os.cpu_count(),
                 # var_labels=None, obj_labels=None,
                 # client=None,
                 *args, **kwargs
                 ):
        #####################################
        #    Default/Optional Attributes    #
        #####################################
        ### Optimization Pre/Post Processing ###
        self.n_opt = int(n_opt)
        ##########################
        #    Pickle Path Init    #
        ##########################
        super().__init__(dir_path=run_path)
        self.run_path = self.abs_path
        self.logger.info(f'OPTIMIZATION STUDY- {self.rel_path}')
        self.logger.info(f'\tCheckpoint Path: {self.cp_path}')
        if self.cp_init:
            return
        #######################################
        #    ATTRIBUTES NEEDED FOR RESTART    #
        #######################################
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
        algorithm.setup(problem,
                        seed=algorithm.seed,
                        verbose=algorithm.verbose,
                        save_history=algorithm.save_history,
                        return_least_infeasible=algorithm.return_least_infeasible,
                        **kwargs)
        algorithm.callback.__init__()
        algorithm.save_history = True
        algorithm.seed = 1
        algorithm.return_least_infeasible = True
        algorithm.verbose = True

        self.algorithm = algorithm
        self.problem = problem
        self.gen_bnd_cases()
        self.gen_test_case()

        ###################################
        #    Attributes To Be Set Later   #
        ###################################
        self.gen1_pop = None

        self.save_self()
        # self.save_self()

    # def newAlg(self):
    #     self.logger.info('INITIALIZING NEW OPTIMIZATION AlGORITHM')
    #     # archive/empty previous runs data directory
    #     # emptyDir(self.optDatDir)
    #     self.algorithm.setup(self.problem,
    #                          seed=self.algorithm.seed,
    #                          verbose=self.algorithm.verbose,
    #                          save_history=self.algorithm.save_history,
    #                          return_least_infeasible=self.algorithm.return_least_infeasible
    #                          )
    #     self.algorithm.callback.__init__()
    #     self.save_self()
    #
    # def initAlg(self):
    #     if self.algorithm.is_initialized:
    #         self.loadCP()
    #         self.logger.info(f'Loaded Algorithm Checkpoint: {self.algorithm}')
    #         self.logger.info(
    #             f'\tLast checkpoint at generation {self.algorithm.callback.gen}')
    #     else:
    #         self.newAlg()

    def run(self, delPrevGen=True):
        self.logger.info('STARTING: OPTIMIZATION ALGORITHM RUN')
        self.algorithm.save_history = True
        self.algorithm.n_gen = self.algorithm.callback.gen
        ######    OPTIMIZATION    ######
        # until the algorithm has not terminated
        while self.algorithm.has_next():
            # First generation
            # population is None so ask for new pop
            if self.algorithm.pop is None:
                self.logger.info('\tSTART-UP: first generation')
                eval_pop = self.algorithm.ask()
                self.algorithm.pop = eval_pop
                self.algorithm.off = eval_pop
            # Previous generation complete
            # If current objective does not have None values then get new pop
            # ie previous pop is complete evaluate new pop
            elif self.algorithm.off is None:
                self.logger.info('\tSTART-UP: new generation')
                eval_pop = self.algorithm.ask()
                self.algorithm.off = eval_pop
            # Mid-generation start-up
            # evaluate offspring population
            else:
                self.logger.info('\tSTART-UP: mid-generation')
                eval_pop = self.algorithm.off
            gen = self.algorithm.callback.gen
            self.logger.info(f'\tGEN: {gen}')
            gen_path = os.path.join(self.abs_path, f'gen{gen}')
            X = eval_pop.get('X')
            ind_paths = [os.path.join(
                gen_path, f'ind{i+1}') for i in range(len(X))]
            # save checkpoint before evaluation
            os.makedirs(gen_path, exist_ok=True)
            saveTxt(gen_path, f'gen{gen}X.txt', X)
            self.save_self()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            eval_pop = self.algorithm.evaluator.eval(self.problem, eval_pop,
                                                     eval_paths=ind_paths,
                                                     # gen=self.algorithm.callback.gen
                                                     # alg=self.algorithm
                                                     )

            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=eval_pop)

            # save top {n_opt} optimal evaluated cases in pf directory
            # compGen = gen
            for off_i, off in enumerate(self.algorithm.off):
                for opt_i, opt in enumerate(self.algorithm.opt[:self.n_opt]):
                    if np.array_equal(off.X, opt.X):
                        opt_folder = f'opt{opt_i+1}'
                        optDir = os.path.join(self.pfDir, opt_folder)
                        ind_folder = f'ind{off_i+1}'
                        offDir = os.path.join(
                            self.abs_path, f'gen{gen}', ind_folder)
                        self.logger.info(
                            f'\tUpdating Pareto front folder: {ind_folder} -> {opt_folder}')
                        try:
                            copy_and_overwrite(offDir, optDir)
                        except FileNotFoundError as err:
                            self.logger.error(str(err))
                            self.logger.warning('SKIPPED: UPDATE PARETO FRONT')
            # do some more things, printing, logging, storing or even modifying the algorithm object
            self.algorithm.off = None
            self.save_self()
            self.plotGen()
            if gen == 1:
                self.gen1_pop = eval_pop
                self.map_gen1()
            if delPrevGen and not gen == 1:
                direct = os.path.join(
                    self.abs_path, f'gen{gen}')
                shutil.rmtree(direct)
        # obtain the result objective from the algorithm
        res = self.algorithm.result()
        # calculate a hash to show that all executions end with the same result
        self.logger.info(f'hash {res.F.sum()}')
        # self.save_self()

    # def execGen1(self):
    #     self.algorithm.termination = termination_from_tuple(('n_gen', 1))
    #     # no checkpoint saved before evaluation
    #     # therefore self.algorithm not stuck with after gen.1 termination criteria
    #     self.algorithm.next()
    #
    # def runGen1(self):
    #     self.logger.info('RUNNING GENERATION 1')
    #     # self.initAlg()
    #     try:
    #         gen1Alg = self.algorithm.history[0]
    #         popX = gen1Alg.pop.get('X')
    #         popF = gen1Alg.pop.get('F')
    #         self.logger.info('\tCurrent Generation 1:')
    #         for i in len(popX):
    #             self.logger.info(
    #                 f'\t\tParameters-{popX[i]} | Objectives- {popF[i]}')
    #     except TypeError:
    #         self.logger.info('\tNo Generation 1 Population Found')
    #         self.logger.info(f'\tAlgorithm History: {self.algorithm.history}')
    #         self.algorithm.termination = termination_from_tuple(('n_gen', 1))
    #         # no checkpoint saved before evaluation
    #         # therefore self.slgorithm not stuck with after gen.1 termination criteria
    #         self.algorithm.next()
    #     self.logger.info('COMPLETE: RUN GENERATION 1')
    #     self.save_self()

    ###################
    #    TEST CASE    #
    ###################
    def gen_test_case(self, test_caseDir='test_case'):
        self.logger.info('\tGENERATING TEST CASE')
        # shutil.rmtree('test_case', ignore_errors=True)
        xl = self.problem.xl
        xu = self.problem.xu
        x_mid = [xl[x_i] + (xu[x_i] - xl[x_i]) / 2
                 for x_i in range(self.problem.n_var)]
        for x_i, var_type in enumerate(self.problem.BaseCase.var_type):
            if var_type.lower() == 'int':
                x_mid[x_i] = int(x_mid[x_i])
        test_caseDir = os.path.join(self.abs_path, test_caseDir)
        self.test_case = self.problem.BaseCase(
            test_caseDir, x_mid)  # , restart=True)

    def run_test_case(self):
        self.logger.info('TEST CASE RUN . . .')
        # if self.test_case is None:
        self.gen_test_case()
        self.logger.info('\tRUNNING TEST CASE')
        self.test_case.run()
        self.logger.info(f'\tParameters: {self.test_case.x}')
        self.logger.info(f'\tObjectives: {self.test_case.f}')
        self.logger.info('TEST CASE COMPLETE ')
        self.save_self()
        return self.test_case

    #################################
    #    RUN POPULATION OF CASES    #
    #################################
    # def runGen(self, X, out):
    #     gen = self.algorithm.callback.gen
    #     # create generation directory for storing data/executing simulations
    #     genDir = os.path.join(self.abs_path, f'gen{gen}')
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
    #         self.gen1_pop = cases
    #     return out

    # def runGen(self, pop):
    #     # get the design space values of the algorithm
    #     X = pop.get("X")
    #     # implement your evluation
    #     gen = self.algorithm.callback.gen
    #     # create generation directory for storing data/executing simulations
    #     genDir = os.path.join(self.abs_path, f'gen{gen}')
    #     # create sub-directories for each individual
    #     indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
    #     cases = self.genCases(indDirs, X)
    #     self.problem.BaseCase.parallelize(cases)
    #     F = np.array([case.f for case in cases])
    #     G = np.array([case.g for case in cases])
    #     # objectives
    #     pop.set("F", F)
    #     # for constraints
    #     pop.set("G", G)
    #     # this line is necessary to set the CV and feasbility status - even for unconstrained
    #     set_cv(pop)
    #     if gen == 1:
    #         self.gen1_pop = cases
    #         self.mapGen1()
    #     self.plotGen()
    #     return pop

    def plotGen(self, gen=None, max_leg_len=10):
        if gen is None:
            gen = self.algorithm.n_gen
        pop = self.algorithm.history[gen - 1].pop
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
        var_plot.save(os.path.join(
            self.plotDir, f'gen{gen}_var_space.png'), dpi=100)

        #### Objective Space Plot ####
        popF = pop.get('F')
        obj_plot = Scatter(title=f'Generation {gen} Objective Space',
                           legend=leg,
                           labels=self.problem.BaseCase.obj_labels
                           )
        for ind_i, ind in enumerate(popF):
            obj_plot.add(ind, label=f'IND {ind_i+1}')
        # save parameter space plot
        obj_plot.save(os.path.join(
            self.plotDir, f'gen{gen}_obj_space.png'), dpi=100)
        self.logger.info(
            f'PLOTTED: Generation {gen} Design and Objective Spaces')
        return var_plot, obj_plot

    def map_gen1(self):
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
                xf = np.column_stack((x, f))
                plot.add(xf)
                # Polynomial best fit lines
                plot.do()
                plot.legend = True
                c = ['r', 'g', 'm']
                for d in range(1, 3 + 1):
                    coefs = np.polyfit(x, f, d)
                    # fit_res = np.polyfit(x, f, d, full=True)
                    # coefs = fit_res[0]
                    # rss = sum(fit_res[1])
                    y = np.polyval(coefs, x)
                    xy = np.column_stack((x, y))
                    xy = xy[xy[:, 0].argsort()]
                    label = f'Order {d} Best Fit'  # , rss={rss}'
                    plot.ax.plot(xy[:, 0], xy[:, 1], label=label, c=c[d - 1])
                plot.do()
                var_str = var_labels[x_i].replace(" ", "_").replace(
                    "/", "|").replace('%', 'precentage').replace("\\", "|")
                obj_str = obj_labels[f_i].replace(" ", "_").replace(
                    "/", "|").replace('%', 'precentage').replace("\\", "|")
                fName = f'{var_str}-vs-{obj_str}.png'
                path = os.path.join(self.mapDir, fName)
                mapPaths.append(path)
                plot.save(path, dpi=100)
                plots.append(plot)
        return plots, mapPaths

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

    # def genCornerCases(self):
    #     lim_perms = self.getLimPerms()
    #     with np.printoptions(precision=3, suppress=True):
    #         cornerCases = []
    #         for perm in lim_perms:
    #             with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
    #                 caseName = str(perm).replace(
    #                     '[', '').replace(']', '').replace(' ', '_')
    #             cornerCaseDir = os.path.join(
    #                 self.abs_path, 'corner-cases', caseName)
    #             cornerCase = self.problem.BaseCase(cornerCaseDir, perm)
    #             cornerCases.append(cornerCase)
    #     self.cornerCases = cornerCases
    #
    # def runCornerCases(self):
    #     '''
    #     Finds binary permutations of limits and runs these cases.
    #
    #     runCornerCases(alg)
    #     -------------------
    #     Running these simulations during the optimization studies pre-processing
    #     provides insight into what is happening at the most extreme limits of the
    #     parameter space.
    #
    #     These simulations may provide insight into whether the user should expand
    #     or restrict the parameter space. They may reveal that the model becomes
    #     non-physical in these extreme cases.
    #
    #     Exploring the entire parameter space becomes important expecially in
    #     optimization studies with geometric parameters. This is due to the
    #     automated meshing introducting more variablility into the CFD case
    #     workflow.
    #     '''
    #     if self.cornerCases is None:
    #         self.genCornerCases()
    #     else:
    #         self.logger.warning(
    #             'SKIPPED: GENERATE CORNER CASES - call self.genCornerCases() directly to create new corner cases')
    #     self.problem.BaseCase.parallelize(self.cornerCases)

    def run_bnd_cases(self, n_pts=2, getDiags=False, do_mesh_study=False):
        self.problem.BaseCase.parallelize(self.bnd_cases)
        self.save_self()
        self.plotBndPtsObj()
        if do_mesh_study:
            self.mesh_study(self.bnd_cases)
            self.save_self()

    def plotBndPtsObj(self):
        plot = Scatter(title='Objective Space: Boundary Cases',
                       legend=True, labels=self.problem.BaseCase.obj_labels)
        F = np.array([case.f for case in self.bnd_cases])
        for obj in F:
            plot.add(obj, label=self.getPointLabel(obj))
        path = os.path.join(self.abs_path, 'boundary-cases',
                            'bndPts_plot-objSpace.png')
        plot.save(path, dpi=100)

    def plotBndPts(self):
        plot = Scatter(title='Design Space: Boundary Cases',
                       legend=True,
                       labels=self.problem.BaseCase.var_labels
                       # grid=True
                       )
        bndPts = np.array([case.x for case in self.bnd_cases])
        for var in bndPts:
            plot.add(var, label=self.getPointLabel(var))
        path = os.path.join(self.abs_path, 'boundary-cases',
                            'bndPts_plot-varSpace.png')
        plot.save(path, dpi=100)

    def gen_bnd_cases(self, n_pts=2, getDiags=False):
        self.logger.info('GENERATING BOUNDARY CASES')
        bndPts = self.getBndPts(n_pts=n_pts, getDiags=getDiags)
        dirs = []
        for pt in bndPts:
            with np.printoptions(precision=3, suppress=True, formatter={'all': lambda x: '%.3g' % x}):
                caseName = str(pt).replace(
                    '[', '').replace(']', '').replace(' ', '_')
            dirs.append(os.path.join(self.abs_path,
                                     'boundary-cases', caseName))
        cases = self.genCases(dirs, bndPts)
        self.bnd_cases = cases
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

    def loadCase(self, case_path):
        # caseCP = os.path.join(case_path, self.problem.BaseCase.__name__ +
        #                       '.checkpoint.npy')
        caseCP = os.path.join(case_path, self.problem.BaseCase.__name__ +
                              '.checkpoint.npy')
        try:
            case, = np.load(caseCP, allow_pickle=True).flatten()
            return case
        except FileNotFoundError as err:
            print(err)

    def loadCases(self, directory):
        '''
        Parameter: directory - searches every directory in given directory for
                                case.npy file and tries to load if it exists.
        '''

        cases = []
        ents = os.listdir(directory)
        for ent in ents:
            case_path = os.path.join(directory, ent)
            if os.path.isdir(case_path):
                case = self.loadCase(case_path)
                cases.append(case)
        return cases

    #####################
    #     PROPERTIES    #
    #####################
    @property
    def plotDir(self):
        plotDir = os.path.join(self.abs_path, 'plots')
        os.makedirs(plotDir, exist_ok=True)
        return plotDir

    @property
    def mapDir(self):
        mapDir = os.path.join(self.abs_path, 'map_gen1')
        os.makedirs(mapDir, exist_ok=True)
        return mapDir

    @property
    def archiveDir(self):
        archiveDir = os.path.join(self.abs_path, 'archive')
        os.makedirs(archiveDir, exist_ok=True)
        return archiveDir

    @property
    def pfDir(self):
        pfDir = os.path.join(self.abs_path, 'pareto_front')
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
    # def bnd_cases(self):
    #     return self._bnd_cases
    #     # path = os.path.join(self.optDatDir, 'bnd_cases.npy')
    #     # bnd_cases = np.load(path, allow_pickle=True).flatten()
    #     # return bnd_cases
    #
    # @bnd_cases.setter
    # def bnd_cases(self, cases):
    #     if cases is not None:
    #         for case in cases:
    #             if isinstance(case, self.problem.BaseCase):
    #                 case.save_self()
    #     # path = os.path.join(self.optDatDir, 'bnd_cases.npy')
    #     # np.save(path, cases, allow_pickle=True)
    #     self._bnd_cases = cases
    #
    # @property
    # def test_case(self):
    #     return self._test_case
    #     # path = os.path.join(self.optDatDir, 'test_case.npy')
    #     # test_case = np.load(path, allow_pickle=True).flatten()
    #     # return test_case
    #
    # @test_case.setter
    # def test_case(self, case):
    #     if isinstance(case, self.problem.BaseCase):
    #         case.save_self()
    #     self._test_case = case
        # path = os.path.join(self.optDatDir, 'test_case.npy')
        # np.save(path, case, allow_pickle=True)

    # ==========================================================================
    # TO BE OVERWRITTEN
    # ==========================================================================
    # def _execPop(self, cases):
    #     pass
    #
    # def _mesh_study(self):
    #     pass

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

    # def mesh_study(self, cases): #, meshSFs=None):
    #     # if meshSFs is None:
    #     #     meshSFs = self.meshSFs
    #     for case in cases:
    #         _, tail = os.path.split(case.abs_path)
    #         mesh_studyDir = os.path.join(case.studyDir, tail)
    #         print(f'MESH STUDY - {mesh_studyDir}')
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
    #             msCase.abs_path = os.path.join(mesh_studyDir, f'meshSF-{sf}')
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
    #         path = os.path.join(mesh_studyDir, 'study.txt')
    #         np.savetxt(path, study, fmt="%s")
    #         path = os.path.join(mesh_studyDir, 'studyX.txt')
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
    #         path = os.path.join(mesh_studyDir, 'studyF.txt')
    #         np.savetxt(path, obj)
    #         print('\t' + str(study).replace('\n', '\n\t'))
    #         #path = os.path.join(mesh_studyDir, 'study.txt')
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
    #              plt.savefig(os.path.join(studyDir, f'mesh_study_plot-{tail}-{obj_label}.png'))
    #              plt.clf()
    #
    #     return study
