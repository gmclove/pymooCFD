# @Author: glove
# @Date:   2021-12-14T16:02:45-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:54-05:00

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.core.problem import Problem
import os
import gmsh
import numpy as np
from glob import glob
# import scipy
# from scipy.integrate import quad

from pymooCFD.core.optRun import OptRun
from pymooCFD.studies.oscillCyl_x2 import OscillCylinder as OscCylX2
from pymooCFD.core.cfdCase import YALES2Case  # CFDCase
# from pymooCFD.util.yales2Tools import getLatestXMF


class OscillCylinder(OscCylX2):
    base_case_path = 'base_cases/osc-cyl_base'
    ####### Define Design Space #########
    n_var = 3
    var_labels = np.append(OscCylX2.var_labels, 'Reynolds Number')
    # options: 'int' or 'real'
    var_type = np.append(OscCylX2.var_type, 'int')
    # lower limits of parameters/variables
    xl = np.append(OscCylX2.xl, 1)
    xu = np.append(OscCylX2.xu, 5)  # upper limits of variables

    def _preProc(self):
        self.genMesh()
        ### EXTRACT VAR ###
        # Extract parameters for each individual
        omega = self.x[0]
        freq = self.x[1]
        Re = self.x[2]
        D = 1
        U = 1
        kin_visc = U * D / Re
        ### SIMULATION INPUT PARAMETERS ###
        # open and read YALES2 input file to array of strings for each line
        in_lines = self.inputLines
        # find line that must change using a keyword
        keyword = 'CYL_ROTATION_PROP'
        kw_lines = self.findKeywordLines(keyword, in_lines)
        for kw_line_i, kw_line in kw_lines:
            # create new string to replace line
            newLine = f'{kw_line[:kw_line.index("=")]}= {omega} {freq} \n'
            in_lines[kw_line_i] = newLine
        # find line that must change using a keyword
        keyword = 'KINEMATIC_VISCOSITY'
        kw_lines = self.findKeywordLines(keyword, in_lines)
        for kw_line_i, kw_line in kw_lines:
            # create new string to replace line
            newLine = f'{kw_line[:kw_line.index("=")]}= {kin_visc}'
            in_lines[kw_line_i] = newLine
        # REPEAT FOR EACH LINE THAT MUST BE CHANGED
        self.inputLines = in_lines

    def _postProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        A_omega = self.x[0]
        freq = self.x[1]
        Re = self.x[2]
        ######## Compute Objectives ##########
        ### Objective 1: Drag on Cylinder ###
        rho = 1
        D = 1  # [m] cylinder diameter
        U = 1
        kin_visc = U * D / Re
        data = np.genfromtxt(self.datPath, skip_header=1)
        # collect data after 100 seconds of simulation time
        mask = np.where(data[:, 3] > 100)
        # Surface integrals of Cp and Cf
        # DRAG: x-direction integrals
        F_P1, = data[mask, 4]
        F_S1, = data[mask, 6]
        F_drag = np.mean(F_P1 - F_S1)
        C_drag = F_drag / ((1 / 2) * rho * U**2 * D**2)
        C_drag_noOsc = [1.363317903314267276, 1.359490209650251247,
                        1.398579950573592079, 1.429212801757665297, 1.451893683272913238]
        Re_i = int(Re / 100) - 1
        prec_change = (C_drag - C_drag_noOsc[Re_i]) * 100 / C_drag_noOsc[Re_i]

        ### Objective 2 ###
        # Objective 2: Power consumed by rotating cylinder
        res_torque = data[mask, 9]
        abs_mean_res_torque = np.mean(abs(res_torque))
        F_res = abs_mean_res_torque * D / 2
        # F_res = abs_mean_res_torque * D / 2
        # t = 1  # [sec]
        # D = 1  # [m] cylinder diameter
        # th = 0.1  # [m] thickness of cylinder wall
        # r_o = D / 2  # [m] outer radius
        # r_i = r_o - th  # [m] inner radius
        # d = 2700  # [kg/m^3] density of aluminum
        # L = 1  # [m] length of cylindrical tube
        # V = L * np.pi * (r_o**2 - r_i**2)  # [m^3] volume of cylinder
        # m = d * V  # [kg] mass of cylinder
        # # [kg m^2] moment of inertia of a hollow cylinder
        # I = 0.5 * m * (r_i**2 + r_o**2)
        # KE_consu = 0.5 * I * omega**2 * 4 * np.pi * freq * \
        #     quad(lambda t: abs(np.sin(2 * np.pi * freq * t)
        #                        * np.cos(2 * np.pi * freq * t)), 0, 1)[0]
        # A_star = (A_omega*D)/(2*U)
        # f_star = freq/0.17001699830023273
        # KE_over_I = (A_star**2/(8*np.pi*f_star))*(2*np.pi*f_star - 0.5*np.sin(4*np.pi*f_star))
        # print(A_star)
        # print(f_star)
        # print(KE_over_I)
        self.f = [prec_change, F_res]

    def _getVar(self, x):
        if x[2] < 10:
            x[2] = x[2] * 100
        return x


class OscillCylinderOpt(OptRun):
    def __init__(self, algorithm, BaseCase,
                 # *args, **kwargs
                 ):
        super().__init__(algorithm, BaseCase,
                         optName='OscCylX3',
                         # n_opt = 20,
                         # base_case_path='base_cases/osc-cyl_base',
                         # optDatDir='cyl-opt_run',
                         # *args, **kwargs
                         )


MyOptRun = OscillCylinderOpt
BaseCase = OscillCylinder


#####################################
#### Genetic Algorithm Criteria #####
#####################################
n_gen = 25
pop_size = 50
n_offsprings = int(pop_size * (2 / 3))  # = num. of evaluations each generation

#################
#    PROBLEM    #
#################
# from pymooCFD.core.pymooBase import GA_CFD
# from pymoo.core.problem import ElementwiseProblem


class GA_CFD(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(n_var=BaseCase.n_var,
                         n_obj=BaseCase.n_obj,
                         n_constr=BaseCase.n_constr,
                         xl=np.array(BaseCase.xl),
                         xu=np.array(BaseCase.xu),
                         *args,
                         **kwargs
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        out = optRun.runGen(X, out)


problem = GA_CFD()


#################
#    DISPLAY    #
#################


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(problem.n_obj):
            self.output.append(
                f'mean obj.{obj + 1}', np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append(
                f"best obj.{obj+1}", algorithm.pop.get('F')[:, obj].min())
        self.output.header()


display = MyDisplay()


##################
#    CALLBACK    #
##################


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.gen = 1
        self.data['best'] = []

    def notify(self, alg):
        # save checkpoint
        optRun.saveCP()
        # increment generation
        self.gen += 1
        self.data["best"].append(alg.pop.get("F").min())
        # For longer runs to save memory may want to use callback.data
        # instead of using algorithm.save_history=True which stores deep
        # copy of algorithm object every generation.
        ## Example: self.data['var'].append(alg.pop.get('X'))


callback = MyCallback()

###################
#    OPERATORS    #
###################

print(BaseCase.var_type)


sampling = MixedVariableSampling(BaseCase.var_type, {
    "real": get_sampling("real_lhs"),  # "real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(BaseCase.var_type, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(BaseCase.var_type, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})


###############################
#    TERMINATION CRITERION    #
###############################
# https://pymoo.org/interface/termination.html
termination = get_termination("n_gen", n_gen)

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

###################
#    ALGORITHM    #
###################
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# initialize algorithm here
# will be overwritten in runOpt() if checkpoint already exists
algorithm = NSGA2(pop_size=pop_size,
                  n_offsprings=n_offsprings,
                  eliminate_duplicates=False,

                  termination=termination,

                  sampling=sampling,
                  crossover=crossover,
                  mutation=mutation,

                  display=display,
                  callback=callback,
                  )
# setup run specific criteria
algorithm.save_history = True
algorithm.seed = 1
algorithm.return_least_infeasible = True
algorithm.verbose = True

################################################################################
########  Optimization Study Object Initialization ##########
optRun = MyOptRun(algorithm, BaseCase)
