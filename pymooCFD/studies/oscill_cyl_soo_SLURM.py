from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.oscill_cyl import OscillCylinder_SOO_SLURM as BaseCase
# from pymooCFD.core.pymooBase import CFDTestProblem


def exec_test():
    study = MinimizeCFD(BaseCase) #, CFDGeneticProblem=CFDTestProblem)
    run_dir = 'test_run'
    if run_dir in study.opt_runs:
        opt_run = study.opt_runs[run_dir]
    else:
        xl = [0.1, 0.2]  # lower limits of parameters/variables
        xu = [10, 1]  # upper limits of variables
        prob = study.get_problem(xl, xu)
        alg = study.get_algorithm(n_gen=2, pop_size=3,
                                  n_offsprings=2)
        opt_run = study.new_run(alg, prob, run_dir=run_dir)
    opt_run.test_case.run()
    # opt_run.run_bnd_cases()
    opt_run.run()


def exec_study():
    study = MinimizeCFD(BaseCase)
    # study.run_case('oscill_cyl_f-Strouhal', [4, 0.15])
    run_dir = 'default_run'
    if run_dir in study.opt_runs:
        opt_run = study.opt_runs[run_dir]
    else:
        xl = [0.1, 0.2]  # lower limits of parameters/variables
        xu = [10, 1]  # upper limits of variables
        prob = study.get_problem(xl, xu)
        alg = study.get_algorithm(n_gen=35, pop_size=50,
                                  n_offsprings=15)
        opt_run = study.new_run(alg, prob, run_dir=run_dir)
    opt_run.test_case.run()
    opt_run.test_case.mesh_study.run()
    opt_run.gen_bnd_cases(n_pts=3, getDiags=True)
    opt_run.run_bnd_cases(do_mesh_study=True)
    opt_run.run()


# # @Author: glove
# # @Date:   2021-12-14T16:02:45-05:00
# # @Last modified by:   glove
# # @Last modified time: 2021-12-15T16:36:54-05:00
#
# from pymooCFD.core.optRun import OptRun
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.factory import get_termination
# from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# from pymoo.core.callback import Callback
# from pymoo.util.display import Display
# from pymoo.core.problem import Problem
# import os
# import gmsh
# import numpy as np
# # import scipy
# from scipy.integrate import quad
#
# # from pymooCFD.util.yales2Tools import getLatestXMF
# # from pymooCFD.core.cfdCase import YALES2Case #CFDCase
# from pymooCFD.studies.oscillCyl_x2 import OscillCylinder
#
#
# class OscillCylinderSOO(OscillCylinder):
#     ####### Define Objective Space ########
#     obj_labels = ['Drag on Cylinder [N]']
#     n_obj = 1
#
#     nProc = 10
#     procLim = 40
#
#     def _preProc(self):
#         ### EXTRACT VAR ###
#         # Extract parameters for each individual
#         omega = self.x[0]
#         freq = self.x[1]
#         ### SIMULATION INPUT PARAMETERS ###
#         # open and read YALES2 input file to array of strings for each line
#         in_lines = self.input_lines_rw
#         # find line that must change using a keyword
#         keyword = 'CYL_ROTATION_PROP'
#         kw_lines = self.findKeywordLines(keyword, in_lines)
#         for line_i, line in kw_lines:
#             # create new string to replace line
#             newLine = f'{line[:line.index("=")]}= {omega} {freq} \n'
#             in_lines[line_i] = newLine
#         # REPEAT FOR EACH LINE THAT MUST BE CHANGED
#         self.input_lines_rw = in_lines
#
#     def _postProc(self):
#         ######## Compute Objectives ##########
#         ### Objective 1: Drag on Cylinder ###
#         U = 1
#         rho = 1
#         D = 1  # [m] cylinder diameter
#         # create string for directory of individual's data file
#         data = np.genfromtxt(self.datPath, skip_header=1)
#         # collect data after 100 seconds of simulation time
#         mask = np.where(data[:, 3] > 100)
#         # Surface integrals of Cp and Cf
#         # DRAG: x-direction integrals
#         F_P1, = data[mask, 4]
#         F_S1, = data[mask, 6]
#         F_drag = np.mean(F_P1 - F_S1)
#         C_drag = F_drag / ((1 / 2) * rho * U**2 * D**2)
#
#         self.f = C_drag
#         # print(self.f)
#
#
# class OscillCylinderOptSOO(OptRun):
#     def __init__(self, algorithm, problem, BaseCase,
#                  *args, **kwargs):
#         super().__init__(algorithm, problem, BaseCase,
#                          optName='SOO-test',
#                          n_opt=20,
#                          # base_case_path='base_cases/osc-cyl_base',
#                          # optDatDir='cyl-opt_run',
#                          *args, **kwargs)
#
#
# MyOptRun = OscillCylinderOptSOO
# BaseCase = OscillCylinderSOO
#
# ####################################
# #    Genetic Algorithm Criteria    #
# ####################################
# n_gen = 25
# pop_size = 50
# n_offsprings = int(pop_size * (1 / 2))  # = num. of evaluations each generation
#
# #################
# #    PROBLEM    #
# #################
# # from pymooCFD.core.pymooBase import GA_CFD
# # from pymoo.core.problem import ElementwiseProblem
#
# ####################################
# #    Genetic Algorithm Criteria    #
# ####################################
# n_gen = 25
# pop_size = 100
# n_offsprings = int(pop_size * (1 / 2))  # = num. of evaluations each generation
#
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
#
#     def _evaluate(self, X, out, *args, **kwargs):
#         out = optRun.runGen(X, out)
#
#
# problem = GA_CFD()
#
# #################
# #    DISPLAY    #
# #################
# # from pymooCFD.core.pymooBase import display
#
#
# class MyDisplay(Display):
#     def _do(self, problem, evaluator, algorithm):
#         super()._do(problem, evaluator, algorithm)
#         for obj in range(problem.n_obj):
#             self.output.append(
#                 f'mean obj.{obj + 1}', np.mean(algorithm.pop.get('F')[:, obj]))
#             self.output.append(
#                 f"best obj.{obj+1}", algorithm.pop.get('F')[:, obj].min())
#         self.output.header()
#
#
# display = MyDisplay()
#
# ##################
# #    CALLBACK    #
# ##################
#
#
# class MyCallback(Callback):
#     def __init__(self) -> None:
#         super().__init__()
#         self.gen = 1
#         self.data['best'] = []
#
#     def notify(self, alg):
#         # save checkpoint
#         optRun.saveCP()
#         # increment generation
#         self.gen += 1
#         self.data["best"].append(alg.pop.get("F").min())
#         # For longer runs to save memory may want to use callback.data
#         # instead of using algorithm.save_history=True which stores deep
#         # copy of algorithm object every generation.
#         ## Example: self.data['var'].append(alg.pop.get('X'))
#
#
# callback = MyCallback()
#
# ###################
# #    OPERATORS    #
# ###################
#
# sampling = MixedVariableSampling(BaseCase.var_type, {
#     "real": get_sampling("real_lhs"),  # "real_random"),
#     "int": get_sampling("int_random")
# })
#
# crossover = MixedVariableCrossover(BaseCase.var_type, {
#     "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
#     "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
# })
#
# mutation = MixedVariableMutation(BaseCase.var_type, {
#     "real": get_mutation("real_pm", eta=3.0),
#     "int": get_mutation("int_pm", eta=3.0)
# })
#
#
# ###############################
# #    TERMINATION CRITERION    #
# ###############################
# # https://pymoo.org/interface/termination.html
# termination = get_termination("n_gen", n_gen)
#
# ###################
# #    ALGORITHM    #
# ###################
# algorithm = GA(
#     pop_size=pop_size,
#     n_offsprings=n_offsprings,
#     eliminate_duplicates=True,
#
#     termination=termination,
#
#     sampling=sampling,
#     crossover=crossover,
#     mutation=mutation,
#
#     display=display,
#     callback=callback,
#
#     verbose=True
# )
# # setup run specific criteria
# algorithm.save_history = True
# algorithm.seed = 1
# algorithm.return_least_infeasible = True
# algorithm.verbose = True
#
# ################################################################################
# ########  Optimization Study Object Initialization ##########
# optRun = MyOptRun(algorithm, problem, BaseCase)
