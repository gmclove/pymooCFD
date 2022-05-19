# @Author: glove
# @Date:   2021-12-10T10:31:58-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:28:45-05:00
# import time
import os
import random
import glob
import numpy as np
import shutil
# import subprocess
# from threading import Thread
# import multiprocessing as mp
# import shutil
# import h5py
import matplotlib.pyplot as plt
# from pymooCFD.core.meshStudy import MeshStudy
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator

from pymooCFD.util.sysTools import copy_and_overwrite, saveTxt
from pymooCFD.core.picklePath import PicklePath


class OptRun(PicklePath):
    def __init__(self, algorithm, problem,
                 run_path='default_run',
                 n_opt=100,
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
            # DELETE PREV. GEN
            prevGen = gen - 1
            if delPrevGen and prevGen > 1:
                direct = os.path.join(
                    self.abs_path, f'gen{prevGen}')
                try:
                    shutil.rmtree(direct)
                except FileNotFoundError as err:
                    self.logger.error('Previous generation file not found')
                    self.logger.error(err)
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
            self.update_pareto_front_folder(self.algorithm.off,
                                            self.algorithm.opt, gen)
            # do some more things, printing, logging, storing or even modifying the algorithm object
            self.algorithm.off = None
            self.save_self()
            self.plotGen()
            self.plotOpt()
            self.plotConv()
            self.plotAllOpt()
            self.plotGens()

            if gen == 1:
                self.gen1_pop = eval_pop
                self.map_gen1()
        # obtain the result objective from the algorithm
        res = self.algorithm.result()
        # calculate a hash to show that all executions end with the same result
        self.logger.info(f'hash {res.F.sum()}')
        # self.save_self()

    def update_pareto_front_folder(self, offspring, optimum, gen):
        for off_i, off in enumerate(offspring):
            for opt_i, opt in enumerate(optimum[:self.n_opt]):
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

    def plotGens(self, gens=None, **kwargs):
        if gens is None:
            gens = range(1, len(self.algorithm.history) + 1)
        pops = [self.algorithm.history[gen - 1].pop for gen in gens]
        popsX = [pop.get('X') for pop in pops]
        popsF = [pop.get('F') for pop in pops]
        gen_labels = [f'GEN {gen}' for gen in gens]
        var_plot = self.plotScatter(popsX, title='Design Space',
                                    ax_labels=self.problem.BaseCase.var_labels,
                                    dir_path=self.plotDir,
                                    fname=f'gens_{gens}_var_space.png',
                                    pt_labels=gen_labels,
                                    **kwargs)

        obj_plot = self.plotScatter(popsF, title='Objective Space',
                                    ax_labels=self.problem.BaseCase.obj_labels,
                                    dir_path=self.plotDir,
                                    fname=f'gens_{gens}_obj_space.png',
                                    pt_labels=gen_labels,
                                    **kwargs)
        return var_plot, obj_plot

        # pops = [self.algorithm.history[gen - 1].pop for gen in gens]
        # ##############################
        # #    Parameter Space Plot    #
        # ##############################
        # var_plot = Scatter(title='Design Space',
        #                    legend=leg,
        #                    labels=self.problem.BaseCase.var_labels,
        #                    #                figsize=(10,8)
        #                    )
        # for pop_i, pop in enumerate(pops):
        #     var_plot.add(pop.get('X'), label=f'GEN {gens[pop_i]}')
        # # save parameter space plot
        # var_plot.save(os.path.join(self.plotDir, f'gens_{gens}_var_space.png'),
        #               dpi=100)
        # #############################
        # #   Objective Space Plot    #
        # #############################
        # obj_plot = Scatter(title='Objective Space',
        #                    legend=leg,
        #                    labels=self.problem.BaseCase.obj_labels
        #                    )
        # for pop_i, pop in enumerate(pops):
        #     obj_plot.add(pop.get('F'), label=f'GEN {gens[pop_i]}')
        # # save parameter space plot
        # obj_plot.save(os.path.join(self.plotDir, f'gens_{gens}_obj_space.png'),
        #               dpi=100)
        #
        # return var_plot, obj_plot

    def plotGen(self, gen=None, **kwargs):
        if gen is None:
            gen = len(self.algorithm.history)  # self.algorithm.callback.gen
        pop = self.algorithm.history[gen - 1].pop
        popX = pop.get('X')
        popF = pop.get('F')
        pt_labels = ['IND ' + str(i + 1) for i in range(len(popX))]
        var_plot = self.plotScatter(popX, title=f'Generation {gen} - Design Space',
                                    ax_labels=self.problem.BaseCase.var_labels,
                                    dir_path=self.plotDir,
                                    fname=f'gen{gen}_var_space.png',
                                    pt_labels=pt_labels,
                                    **kwargs)

        obj_plot = self.plotScatter(popF, title=f'Generation {gen} - Objective Space',
                                    ax_labels=self.problem.BaseCase.obj_labels,
                                    dir_path=self.plotDir,
                                    fname=f'gen{gen}_obj_space.png',
                                    pt_labels=pt_labels,
                                    **kwargs)
        self.logger.info(
            f'PLOTTED: Generation {gen} - Design and Objective Spaces')
        return var_plot, obj_plot

    # def plotPop(self, pop, title):
    def plotConv(self, gen=None, **kwargs):
        plots = []
        if gen is None:
            gen = len(self.algorithm.history)
        hist = self.algorithm.history
        n_evals = [alg.evaluator.n_eval for alg in hist]
        # n_gen = [alg.n_gen for alg in hist]
        # CALCULATE MEAN OPTIMUM FOR EACH GENERATION
        opt_avg = []
        for h in hist:
            F_opt = np.array([o.F for o in h.opt])
            F_avg = np.mean(F_opt, axis=0)
            opt_avg.append(F_avg)
        opt_avg = np.array(opt_avg)
        # MEAN OPTIMUM
        for obj_i in range(self.problem.BaseCase.n_obj):
            opt_obj = opt_avg[:, obj_i]
            fig, ax = plt.subplots()
            plots.append(fig)
            ax.plot(n_evals, opt_obj, "--", **kwargs)
            fig.suptitle('Convergence of Mean Optimum')
            ax.set_title(
                f'Objective {obj_i+1}: {self.problem.BaseCase.obj_labels[obj_i]}')
            ax.set_xlabel('Number of Evaluations')
            ax.set_ylabel('Mean of Optimum')
            fig.savefig(os.path.join(
                self.plotDir, f'conv_mean_opt-obj{obj_i}'))
        if self.problem.n_obj == 1:
            # SINGLE OPTIMUM
            opt = np.array([alg.opt[0].F for alg in hist])
            # n_gen = [alg.n_gen for alg in hist]
            opt = opt[:]
            fig, ax = plt.subplots()
            plots.append(fig)
            ax.plot(n_evals, opt, "--", **kwargs)
            fig.suptitle('Convergence of Optimum')
            ax.set_title(self.problem.BaseCase.obj_labels[0])
            ax.set_xlabel('Number of Evaluations')
            ax.set_ylabel('Optimum')
            fig.savefig(os.path.join(self.plotDir, f'conv_opt'))
        # HYPERVOLUME
        gen1_F = self.algorithm.history[0].pop.get('F')
        ref_pt = []
        for obj_i, objs in enumerate(gen1_F.T):
            ref_pt.append(np.mean(objs))
            # ref_pt.append((max(objs)-min(objs))/2)
        ref_pt = np.array(ref_pt)
        calc_hv = get_performance_indicator("hv", ref_point=ref_pt)
        init_hv = calc_hv.do(gen1_F)
        hvs = []
        hvs_norm = []
        for alg in self.algorithm.history:
            F = alg.pop.get('F')
            hv = calc_hv.do(F)
            hvs.append(hv)
            hvs_norm.append((hv - init_hv)/init_hv)
        # plot
        ref_pt_str = np.array2string(ref_pt, precision=3)
        fig, ax = plt.subplots()
        plots.append(fig)
        ax.plot(n_evals, hvs, "--", **kwargs)
        fig.suptitle('Convergence of Hypervolume')
        ax.set_title(f'Reference Point: {ref_pt_str}')
        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('Hypervolume')
        fig.savefig(os.path.join(self.plotDir, 'conv_hv'))
        # plot
        fig, ax = plt.subplots()
        plots.append(fig)
        ax.plot(n_evals, hvs_norm, "--", **kwargs)
        fig.suptitle('Convergence of Hypervolume')
        ax.set_title(f'Reference Point: {ref_pt_str}')
        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('Normalized Hypervolume')
        fig.savefig(os.path.join(self.plotDir, 'conv_hv_norm'))

        return plots

    def plotOpt(self, gen=None, max_opt_len=20, **kwargs):
        if gen is None:
            gen = len(self.algorithm.history)
        pop = self.algorithm.history[gen - 1].opt
        if (max_opt_len is not None and max_opt_len < len(pop)):
            pop = pop[random.sample(range(len(pop)), max_opt_len)]
        popX = pop.get('X')
        popF = pop.get('F')
        pt_labels = ['OPT ' + str(i + 1) for i in range(len(popX))]
        var_plot = self.plotScatter(popX, title=f'20 Optimum After {gen} Generations - Design Space',
                                    ax_labels=self.problem.BaseCase.var_labels,
                                    dir_path=self.plotDir,
                                    fname=f'20_opt_gen{gen}_var_space.png',
                                    dif_markers=True, max_leg_len=20,
                                    pt_labels=pt_labels, s=10,
                                    **kwargs)
        obj_plot = self.plotScatter(popF, title=f'20 Optimum After {gen} Generations - Objective Space',
                                    ax_labels=self.problem.BaseCase.obj_labels,
                                    dir_path=self.plotDir,
                                    fname=f'20_opt_gen{gen}_obj_space.png',
                                    dif_markers=True, max_leg_len=20,
                                    pt_labels=pt_labels, s=20,
                                    **kwargs)
        self.logger.info(
            f'PLOTTED: Optimum after {gen} Generations - Design and Objective Spaces')
        return var_plot, obj_plot

    def plotAllOpt(self, gen=None, **kwargs):
        if gen is None:
            gen = len(self.algorithm.history)
        pop = self.algorithm.history[gen - 1].opt
        popX = pop.get('X')
        popF = pop.get('F')
        pt_labels = ['OPT ' + str(i + 1) for i in range(len(popX))]
        var_plot = self.plotScatter(popX, title=f'Optimum After {gen} Generations - Design Space',
                                    ax_labels=self.problem.BaseCase.var_labels,
                                    dir_path=self.plotDir,
                                    fname=f'opt_gen{gen}_var_space.png',
                                    dif_markers=True, pt_labels=pt_labels, s=10,
                                    **kwargs)
        obj_plot = self.plotScatter(popF, title=f'Optimum After {gen} Generations - Objective Space',
                                    ax_labels=self.problem.BaseCase.obj_labels,
                                    dir_path=self.plotDir,
                                    fname=f'opt_gen{gen}_obj_space.png',
                                    dif_markers=True, pt_labels=pt_labels, s=20,
                                    **kwargs)
        self.logger.info(
            f'PLOTTED: Optimum after {gen} Generations - Design and Objective Spaces')
        return var_plot, obj_plot

    def map_gen1(self):
        ##### Variable vs. Objective Plots ######
        # extract objectives and variables columns and plot them against each other
        var_labels = self.problem.BaseCase.var_labels
        obj_labels = self.problem.BaseCase.obj_labels
        popX = self.algorithm.history[0].pop.get('X').astype(float)
        popF = self.algorithm.history[0].pop.get('F').astype(float)
        map_paths = []
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
                    "/", "|").replace('%', 'percentage').replace("\\", "|")
                obj_str = obj_labels[f_i].replace(" ", "_").replace(
                    "/", "|").replace('%', 'percentage').replace("\\", "|")
                fName = f'{var_str}-vs-{obj_str}.png'
                path = os.path.join(self.mapDir, fName)
                map_paths.append(path)
                plot.save(path, dpi=200)
                plots.append(plot)
        return plots, map_paths

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

    def plotScatter(self, points, title=None, ax_labels='f',
                    pt_labels=None, max_leg_len=10, max_ax_label_len=20,
                    dir_path=None, fname=None, dpi=200, tight_layout=True,
                    dif_markers=False,
                    **kwargs):
        leg, labs, tit = False, 'f', None
        points = np.array(points)
        n_pts = len(points)
        # print(points.shape)
        if pt_labels is None:
            pt_labels = [str(i) for i in range(n_pts)]
        assert len(pt_labels) == n_pts
        if dir_path is None:
            dir_path = self.plotDir
        if fname is None:
            search_str = os.path.join(dir_path, 'space_plot-*.png')
            ents = [ent for ent in glob.glob(
                search_str) if os.path.isfile(ent)].sort()
            if ents:
                s = ents[-1]
                n_new = str(int(
                    s.replace('space_plot-', '').replace('.png', ''))
                    + 1).zfill(2)
            else:
                n_new = '00'
            fname = 'space_plot-' + n_new + '.png'
        n_axes = points.shape[-1]
        if n_axes <= 3:
            figsize = kwargs.pop('figsize', (8, 6))
            tit = title
            if points.shape[0] <= max_leg_len:
                leg = True
            labs = ax_labels
        else:
            if all(len(label) <= max_ax_label_len for label in ax_labels):
                labs = ax_labels
            if 'figsize' in kwargs:
                figsize = kwargs.pop('figsize')
            else:
                x_SF, y_SF = 8 / 2.5, 6 / 2.5
                figsize = (n_axes * x_SF, n_axes * y_SF)
        if dif_markers:
            all_markers = list({
                '.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down',
                '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right',
                '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right',
                '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star',
                'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x',
                'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline',
                'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright',
                2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright',
                6: 'caretup', 7: 'caretdown', 8: 'caretleftbase',
                9: 'caretrightbase', 10: 'caretupbase'})
            concat_markers = all_markers
            while len(concat_markers) < n_pts:
                concat_markers += all_markers
            markers = [m for i, m in enumerate(concat_markers) if i < n_pts]
        else:
            markers = ['o' for _ in range(n_pts)]
        # if legend is not None:
        #     leg = legend

        plot = Scatter(title=tit,
                       legend=leg,
                       labels=labs,
                       figsize=figsize,
                       tight_layout=tight_layout,
                       **kwargs
                       )

        for pt_i, pt in enumerate(points):
            plot.add(pt, label=pt_labels[pt_i], marker=markers[pt_i])
        plot.save(os.path.join(dir_path, fname), dpi=dpi)
        return plot

    def plotBndPts(self, **kwargs):
        bndPts = np.array([case.x for case in self.bnd_cases])
        pt_labels = [self.getPointLabel(pt) for pt in bndPts]
        plot = self.plotScatter(bndPts, title='Design Space: Boundary Cases',
                                ax_labels=self.problem.BaseCase.var_labels,
                                dir_path=self.bndCasesDir,
                                fname='bndPts_plot-varSpace.png',
                                pt_labels=pt_labels, dif_markers=True, s=20,
                                **kwargs)

        # var_leg, var_title, var_labels = False, None, 'f'
        #
        # if self.problem.BaseCase.n_var <= 3:
        #     if len(self.bnd_cases) <= max_leg_len:
        #         var_leg = True
        #     var_title = 'Design Space: Boundary Cases'
        #     if all(len(label) <= 20
        #            for label in self.problem.BaseCase.var_labels):
        #         var_labels = self.problem.BaseCase.var_labels
        # plot = Scatter(title='Design Space: Boundary Cases',
        #                legend=True,
        #                labels=self.problem.BaseCase.var_labels
        #                # grid=True
        #                )
        # bndPts = np.array([case.x for case in self.bnd_cases])
        # for var in bndPts:
        #     plot.add(var, label=self.getPointLabel(var))
        # path = os.path.join(self.abs_path, 'boundary-cases',
        #                     'bndPts_plot-varSpace.png')
        # plot.save(path, dpi=100)
        return plot

    def plotBndPtsObj(self, **kwargs):
        bndObjs = np.array([case.f for case in self.bnd_cases])
        pt_labels = [self.getPointLabel(pt) for pt in bndObjs]
        plot = self.plotScatter(bndObjs, title='Objective Space: Boundary Cases',
                                ax_labels=self.problem.BaseCase.obj_labels,
                                dir_path=self.bndCasesDir,
                                fname='bndPts_plot-objSpace.png',
                                pt_labels=pt_labels, dif_markers=True, s=20,
                                **kwargs)
        # legend display
        # obj_leg, obj_title, obj_labels = False, None, 'f'
        #
        # if self.problem.BaseCase.n_obj <= 3:
        #     if len(self.bnd_cases) <= max_leg_len:
        #         obj_leg = True
        #     obj_title = 'Objective Space: Boundary Cases'
        #     if all(len(label) <= 20
        #            for label in self.problem.BaseCase.obj_labels):
        #         obj_labels = self.problem.BaseCase.obj_labels
        #
        # plot = Scatter(title=obj_title,
        #                legend=obj_leg, labels=obj_labels, tight_layout=True)
        # F = np.array([case.f for case in self.bnd_cases])
        # for obj in F:
        #     plot.add(obj, label=self.getPointLabel(obj))
        # path = os.path.join(self.abs_path, 'boundary-cases',
        #                     'bndPts_plot-objSpace.png')
        # plot.save(path, dpi=100)
        return plot

    def gen_bnd_cases(self, n_pts=2, getDiags=False):
        self.logger.info('GENERATING BOUNDARY CASES')
        bndPts = self.getBndPts(n_pts=n_pts, getDiags=getDiags)
        dirs = []
        for pt in bndPts:
            with np.printoptions(precision=3, suppress=True,
                                 formatter={'all': lambda x: '%.3g' % x}):
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

    def run_bnd_cases(self, do_mesh_study=False):
        self.problem.BaseCase.parallelize(self.bnd_cases)
        self.save_self()
        self.plotBndPtsObj()
        if do_mesh_study:
            for case in self.bnd_cases:
                case.mesh_study.run()
            self.save_self()

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
    # @property
    # def plots_path(self):
    #     path = os.path.join(self.abs_path, 'plots')
    #     os.makedirs(path, exist_ok=True)
    #     return path
    @property
    def bndCasesDir(self):
        plotDir = os.path.join(self.abs_path, 'boundary-cases')
        os.makedirs(plotDir, exist_ok=True)
        return plotDir

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
