import numpy as np
import os

from pymoo.algorithms.base.genetic import GeneticAlgorithm \
    as PymooGeneticAlgorithm
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.factory import get_termination

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

from pymoo.core.problem import Problem
# from pymoo.core.problem import ElementwiseProblem


#################
#    Output    #
#################
class PymooFEAOutput(Output):
    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(problem.n_obj):
            self.output.append(
                f"mean obj.{obj + 1}", np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append(
                f"best obj.{obj+1}", algorithm.pop.get('F')[:, obj].min())
        self.output.header()


output = PymooFEAOutput()


##################
#    CALLBACK    #
##################
class PymooFEACallback(Callback):
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


callback = PymooFEACallback()


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
class FEAGeneticProblem(Problem):
    def __init__(self, BaseCase, xl, xu, **kwargs):
        # check if problem is set up properly
        if not (len(xl) == len(xu) and
                len(xu) == len(BaseCase.var_labels) and
                len(BaseCase.var_labels) == BaseCase.n_var
                ):
            print(BaseCase.__dict__)
            raise Exception("Design Space Definition Incorrect")
        # initialize pymoo problem using keyword arguments from BaseCase
        super().__init__(xl=xl, xu=xu, **BaseCase.__dict__, **kwargs)
        # create new attributes specific to pymooFEA
        self.BaseCase = BaseCase
        self.gen1Pop = None
        self.validated = False

    def _evaluate(self, X, out, *args, **kwargs):
        # use keyword argument eval_paths to create and run cases using BaseCase
        eval_paths = kwargs.get('eval_paths')
        assert len(eval_paths) == len(X), 'len(eval_paths) != len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.BaseCase(eval_paths[x_i], x, validated=self.validated)
            cases.append(case)
        self.BaseCase.parallelize(cases)
        F = np.array([case.f for case in cases])
        G = np.array([case.g for case in cases])
        print('Objectives:')
        print('\t' + str(F).replace('\n', '\n\t'))
        print('Constraints:')
        print('\t' + str(G).replace('\n', '\n\t'))
        out['F'] = F
        out['G'] = G


class FEATestProblem(FEAGeneticProblem):
    def _evaluate(self, X, out, *args, **kwargs):
        eval_paths = kwargs.get('eval_paths')
        assert len(eval_paths) == len(X), 'len(eval_paths) != len(X)'
        cases = []
        for x_i, x in enumerate(X):
            case = self.BaseCase(eval_paths[x_i], x, validated=self.validated)
            cases.append(case)
        # skip execution of cases for testing purposes
        # fill in mock results
        F = np.ones((len(X), self.n_obj))
        G = np.zeros((len(X), self.n_obj)) - 1
        out['F'] = F
        out['G'] = G


###################
#    ALGORITHM    #
###################
def get_FEAGeneticAlgorithm(GeneticAlgorithm):
    assert PymooGeneticAlgorithm in GeneticAlgorithm.mro()
    global FEAGeneticAlgorithm

    class FEAGeneticAlgorithm(GeneticAlgorithm):
        def __init__(self,
                     pop_size=5,
                     n_offsprings=2,
                     eliminate_duplicates=True,

                     termination=get_termination("n_gen", 5),

                     callback=callback,
                     output=output,

                     n_gen=None,
                     **kwargs
                     ):
            if n_gen is not None:
                termination = get_termination("n_gen", n_gen)
            super().__init__(pop_size=pop_size,
                             n_offsprings=n_offsprings,
                             eliminate_duplicates=eliminate_duplicates,

                             termination=termination,

                             callback=callback,
                             output=output,
                             **kwargs
                             )

        # def setup(self, problem, **kwargs):
        #     super().setup(self, problem, **kwargs)
            # self.callback.__init__()
            # self.save_history = True
            # self.seed = 1
            # self.return_least_infeasible = True
            # self.verbose = True

    return FEAGeneticAlgorithm


FEAGeneticAlgorithm = get_FEAGeneticAlgorithm(NSGA2)
