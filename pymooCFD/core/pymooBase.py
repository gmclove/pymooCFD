import numpy as np
import os

from pymoo.algorithms.base.genetic import GeneticAlgorithm \
    as PymooGeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.factory import get_termination

from pymoo.core.problem import Problem
# from pymoo.core.problem import ElementwiseProblem

#####################################
#### Genetic Algorithm Criteria #####
#####################################
# n_gen = 2
# pop_size = 2
# # = number of evaluations each generation
# n_offsprings = int(pop_size * (2 / 3))

#################
#    DISPLAY    #
#################


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(problem.n_obj):
            self.output.append(
                f"mean obj.{obj + 1}", np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append(
                f"best obj.{obj+1}", algorithm.pop.get('F')[:, obj].min())
        self.output.header()


display = MyDisplay()


##################
#    CALLBACK    #
##################
class PymooCFDCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.gen = 1
        self.data['best'] = []

    def notify(self, alg):
        # save checkpoint
        # optRun.saveCP(alg=alg)
        # increment generation
        self.gen += 1
        self.data["best"].append(alg.pop.get("F").min())
        # For longer runs to save memory may want to use callback.data
        # instead of using algorithm.save_history=True which stores deep
        # copy of algorithm object every generation.
        ## Example: self.data['var'].append(alg.pop.get('X'))


callback = PymooCFDCallback()


###############################
#    TERMINATION CRITERION    #
###############################
# https://pymoo.org/interface/termination.html
# from pymoo.factory import get_termination
# termination = get_termination("n_gen", n_gen)

# from pymoo.util.termination.default import MultiObjectiveDefaultTermination
# termination = MultiObjectiveDefaultTermination(
#     x_tol=1e-8,
#     cv_tol=1e-6,
#     f_tol=0.0025,
#     nth_gen=5,
#     n_last=30,
#     n_max_gen=1000,
#     n_max_evals=100000
# )

#################
#    PROBLEM    #
#################
class CFDGeneticProblem(Problem):
    def __init__(self, BaseCase,
                 xl, xu,
                 # n_var, n_obj, n_constr,
                 # var_labels=None,
                 # obj_labels=None,
                 *args, **kwargs):
        if not (len(xl) == len(xu) and
                len(xu) == len(BaseCase.var_labels) and
                len(BaseCase.var_labels) == BaseCase.n_var
                ):
            raise Exception("Design Space Definition Incorrect")
        super().__init__(n_var=BaseCase.n_var,
                         n_obj=BaseCase.n_obj,
                         n_constr=BaseCase.n_constr,
                         xl=np.array(xl),
                         xu=np.array(xu),
                         *args,
                         **kwargs
                         )
        self.BaseCase = BaseCase
        self.gen1Pop = None
        self.validated = False


    def _evaluate(self, X, out, *args, **kwargs):
        runDir = kwargs.get('runDir')
        gen = kwargs.get('gen')
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(runDir, f'gen{gen}')
        # create sub-directories for each individual
        indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
        # cases = self.genCases(indDirs, X)
        assert len(indDirs) == len(X), 'len(paths) == len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.BaseCase(indDirs[x_i], x, validated=self.validated)
            cases.append(case)
        self.BaseCase.parallelize(cases)
        F = np.array([case.f for case in cases])
        G = np.array([case.g for case in cases])
        out['F'] = F
        out['G'] = G


class CFDTestProblem(CFDGeneticProblem):
    def _evaluate(self, X, out, *args, **kwargs):
        run_path = kwargs.get('run_path')
        gen = kwargs.get('gen')
        print('GEN:', gen)
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(run_path, f'gen{gen}')
        # create sub-directories for each individual
        indDirs = [os.path.join(genDir, f'ind{i+1}') for i in range(len(X))]
        # cases = self.genCases(indDirs, X)
        assert len(indDirs) == len(X), 'len(paths) != len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.BaseCase(indDirs[x_i], x, validated=self.validated)
            cases.append(case)
        # self.BaseCase.parallelize(cases)
        F = np.ones((len(X), self.n_obj))
        G = np.zeros((len(X), self.n_obj)) - 1
        out['F'] = F
        out['G'] = G

###################
#    ALGORITHM    #
###################


def get_CFDGeneticAlgorithm(GeneticAlgorithm):
    assert PymooGeneticAlgorithm in GeneticAlgorithm.mro()
    global CFDGeneticAlgorithm

    class CFDGeneticAlgorithm(GeneticAlgorithm):
        def __init__(self, sampling, crossover, mutation,
                     pop_size=5,
                     n_offsprings=2,
                     eliminate_duplicates=True,

                     termination=get_termination("n_gen", 5),

                     callback=callback,
                     display=display,
                     **kwargs
                     ):
            super().__init__(pop_size=pop_size,
                             n_offsprings=n_offsprings,
                             eliminate_duplicates=eliminate_duplicates,

                             termination=termination,

                             sampling=sampling,
                             crossover=crossover,
                             mutation=mutation,

                             callback=callback,
                             display=display,
                             **kwargs
                             )

        # def setup(self, problem, **kwargs):
        #     super().setup(self, problem, **kwargs)
            # self.callback.__init__()
            # self.save_history = True
            # self.seed = 1
            # self.return_least_infeasible = True
            # self.verbose = True


    return CFDGeneticAlgorithm


CFDGeneticAlgorithm = get_CFDGeneticAlgorithm(NSGA2)
