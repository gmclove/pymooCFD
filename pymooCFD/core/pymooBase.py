import numpy as np
import os

from pymoo.core.problem import Problem
# from pymoo.core.problem import ElementwiseProblem


#################
#    PROBLEM    #
#################
class CFDProblem_GA(Problem):
    def __init__(self, BaseCase, *args, **kwargs):
        super().__init__(n_var=BaseCase.n_var,
                         n_obj=BaseCase.n_obj,
                         n_constr=BaseCase.n_constr,
                         xl=np.array(BaseCase.xl),
                         xu=np.array(BaseCase.xu),
                         *args,
                         **kwargs
                         )
        self.BaseCase = BaseCase
        self.gen1Pop = None
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
            case = self.BaseCase(indDirs[x_i], x)
            cases.append(case)
        self.BaseCase.parallelize(cases)
        F = np.array([case.f for case in cases])
        G = np.array([case.g for case in cases])
        out['F'] = F
        out['G'] = G
        if gen == 1:
            self.gen1Pop = cases

# problem = GA_CFD()
#
#
#
#
#
#
#
#
#
#
#
# #### Genetic Algorithm Criteria #####
# #####################################
# n_gen = 2
# pop_size = 2
# n_offsprings = int(pop_size * (2 / 3)) # = number of evaluations each generation
#
# #################
# #    PROBLEM    #
# #################
# from pymoo.core.problem import Problem
# import numpy as np
# # from pymoo.core.problem import ElementwiseProblem
#
# class GA_CFD(Problem):
#     def __init__(self, *args, **kwargs):
#         super().__init__(n_var=BaseCase.n_var,
#                          n_obj=BaseCase.n_obj,
#                          n_constr=BaseCase.n_constr,
#                          xl=np.array(BaseCase.xl),
#                          xu=np.array(BaseCase.xu),
#                          *args,
#                          **kwargs
#                          )
#     def _evaluate(self, X, out, *args, **kwargs):
#         out = optStudy.runGen(X, out)
#
# problem = GA_CFD()
#
#
#################
#    DISPLAY    #
#################
from pymoo.util.display import Display


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


# ##################
# #    CALLBACK    #
# ##################
# from pymoo.core.callback import Callback
#
# class MyCallback(Callback):
#     def __init__(self) -> None:
#         super().__init__()
#         self.gen = 1
#         self.data['best'] = []
#
#     def notify(self, alg):
#         # save checkpoint
#         optStudy.saveCP(alg=alg)
#         # increment generation
#         self.gen += 1
#         self.data["best"].append(alg.pop.get("F").min())
#         ### For longer runs to save memory may want to use callback.data
#         ## instead of using algorithm.save_history=True which stores deep
#         ## copy of algorithm object every generation.
#         ## Example: self.data['var'].append(alg.pop.get('X'))
#
# callback = MyCallback()
#
# ###################
# #    OPERATORS    #
# ###################
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
#
# sampling = MixedVariableSampling(BaseCase.varType, {
#     "real": get_sampling("real_lhs"),  # "real_random"),
#     "int": get_sampling("int_random")
# })
#
# crossover = MixedVariableCrossover(BaseCase.varType, {
#     "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
#     "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
# })
#
# mutation = MixedVariableMutation(BaseCase.varType, {
#     "real": get_mutation("real_pm", eta=3.0),
#     "int": get_mutation("int_pm", eta=3.0)
# })
#
#
# ###############################
# #    TERMINATION CRITERION    #
# ###############################
# # https://pymoo.org/interface/termination.html
# from pymoo.factory import get_termination
# termination = get_termination("n_gen", n_gen)
#
# # from pymoo.util.termination.default import MultiObjectiveDefaultTermination
# # termination = MultiObjectiveDefaultTermination(
# #     x_tol=1e-8,
# #     cv_tol=1e-6,
# #     f_tol=0.0025,
# #     nth_gen=5,
# #     n_last=30,
# #     n_max_gen=1000,
# #     n_max_evals=100000
# # )
#
#
# ###################
# #    ALGORITHM    #
# ###################
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# ### initialize algorithm here
# ## will be overwritten in runOpt() if checkpoint already exists
# algorithm = NSGA2(pop_size=pop_size,
#                   n_offsprings=n_offsprings,
#                   eliminate_duplicates=True,
#
#                   termination = termination,
#
#                   sampling = sampling,
#                   crossover = crossover,
#                   mutation = mutation,
#
#                   display = display,
#                   callback = callback,
#                   )
# # setup run specific criteria
# algorithm.save_history = True
# algorithm.seed = 1
# algorithm.return_least_infeasible = True
# algorithm.verbose = True
