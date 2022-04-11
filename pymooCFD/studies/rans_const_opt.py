# @Author: glove
# @Date:   2021-12-14T16:02:45-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:54-05:00

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation

from pymooCFD.core.optStudy import OptStudy
from pymooCFD.core.pymooBase import callback, display
from pymooCFD.problems.rans_k_eps import BaseCase

class RANSConstOpt(OptStudy):
    def __init__(self, algorithm, BaseCase,
                 *args, **kwargs
                 ):
        super().__init__(algorithm, BaseCase,
                         # optName='',
                         # runDir='run'
                         # n_opt = 20,
                         # base_case_path='base_cases/osc-cyl_base',
                         # optDatDir='cyl-opt_run',
                         *args, **kwargs
                         )

MyOptStudy = RANSConstOpt

#####################################
#### Genetic Algorithm Criteria #####
#####################################
n_gen = 2
pop_size = 3
n_offsprings = int(pop_size * (1 / 2))  # = num. of evaluations each generation

###################
#    OPERATORS    #
###################

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
optStudy = MyOptStudy(algorithm, BaseCase)
