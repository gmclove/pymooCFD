# @Author: glove
# @Date:   2021-12-15T14:59:23-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T17:20:34-05:00

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_termination
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.core.problem import Problem
import numpy as np
# from pymooCFD.core.cfdCase import CFDCase
from pymooCFD.core.cfdCase import YALES2Case
from pymooCFD.core.optStudy import OptStudy


class LocalCompDistOpt(OptStudy):
    pass
    # def __init__(self):
    #     super().__init__()


class CompDistSLURM(YALES2Case):
    baseCaseDir = 'base_cases/osc-cyl_base'
    inputFile = '2D_cylinder.in'
    jobFile = 'jobslurm.sh'

    n_var = 2
    # , 'Time Step']
    var_labels = ['Number of Tasks', 'Number of CPUs per Task']
    varType = ['int', 'int']
    xl = [1, 1]
    xu = [30, 30]

    n_obj = 1
    obj_labels = ['Solve Time']  # , 'Fidelity']

    n_constr = 0

    externalSolver = True
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']

    def __init__(self, caseDir, x, meshSF=1.0,
                 # *args, **kwargs
                 ):
        super().__init__(caseDir, x,
                         meshSF=meshSF,
                         jobFile='jobslurm.sh',
                         meshFile='2D_cylinder.msh22',
                         # *args, **kwargs
                         )

    def _preProc(self):
        ntasks = self.x[0]
        c = self.x[1]
        # read job lines
        job_lines = self.jobLines
        if len(job_lines) > 0:
            newLine = f'#SBATCH --cpus-per-task={c}'
            kws = ['#SBATCH --cpus-per-task', '#SBATCH -c']
            job_lines = self.findAndReplaceKeywordLines(
                job_lines, newLine, kws)

            newLine = f'#SBATCH --ntasks={ntasks}'
            kws = ['#SBATCH --ntasks', '#SBATCH -n']
            job_lines = self.findAndReplaceKeywordLines(
                job_lines, newLine, kws)
            # kw_lines_1 = self.findKeywordLines(
            #     '#SBATCH --cpus-per-task', job_lines)
            # kw_lines_2 = self.findKeywordLines('#SBATCH -c', job_lines)
            # if len(kw_lines_1) > 0 or len(kw_lines_2) > 0:
            #     for line_i, line in kw_lines_1:
            #         job_lines[line_i] = newLine
            #     for line_i, line in kw_lines_2:
            #         job_lines[line_i] = newLine
            # else:
            #     job_lines.insert(0, newLine)
            # newLine = f'#SBATCH --ntasks={ntasks}'
            # kw_lines_1 = self.findKeywordLines('#SBATCH --ntasks', job_lines)
            # kw_lines_2 = self.findKeywordLines('#SBATCH -n', job_lines)
            # if len(job_lines) > 0:
            #     for line_i, line in kw_lines_1:
            #         job_lines[line_i] = newLine
            #     for line_i, line in kw_lines_2:
            #         job_lines[line_i] = newLine
            # else:
            #     job_lines.insert(0, newLine)
            # write job lines
            self.jobLines = job_lines
        elif self.jobFile in self.solverExecCmd:
            self.solverExecCmd.insert(
                1, '-c').insert(2, str(c)).insert(3, '-n').insert(4, str(ntasks))
        else:
            self.logger.warning('INCOMPLETE: PRE-PROCESSING')

    def _postProc(self):
        self.f = self.solnTime

    # def _execDone(self):
    #     return True


class SOO(OptStudy):
    def __init__(self, algorithm, problem, BaseCase,
                 # *args, **kwargs
                 ):
        super().__init__(algorithm, problem, BaseCase,
                         optName='CompDistSOO-test',
                         n_opt=20,
                         # baseCaseDir='base_cases/osc-cyl_base',
                         # optDatDir='cyl-opt_run',
                         # *args, **kwargs
                         )


MyOptStudy = SOO
BaseCase = CompDistSLURM

####################################
#    Genetic Algorithm Criteria    #
####################################
n_gen = 25
pop_size = 50
n_offsprings = int(pop_size * (1 / 2))  # = num. of evaluations each generation

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
        out = optStudy.runGen(X, out)


problem = GA_CFD()

#################
#    DISPLAY    #
#################
# from pymooCFD.core.pymooBase import display


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
        optStudy.saveCP()
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

sampling = MixedVariableSampling(BaseCase.varType, {
    "real": get_sampling("real_lhs"),  # "real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(BaseCase.varType, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(BaseCase.varType, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})


###############################
#    TERMINATION CRITERION    #
###############################
# https://pymoo.org/interface/termination.html
termination = get_termination("n_gen", n_gen)

###################
#    ALGORITHM    #
###################
algorithm = GA(
    pop_size=pop_size,
    n_offsprings=n_offsprings,
    eliminate_duplicates=True,

    termination=termination,

    sampling=sampling,
    crossover=crossover,
    mutation=mutation,

    display=display,
    callback=callback,

    verbose=True
)
# setup run specific criteria
algorithm.save_history = True
algorithm.seed = 1
algorithm.return_least_infeasible = True
algorithm.verbose = True

################################################################################
########  Optimization Study Object Initialization ##########
optStudy = MyOptStudy(algorithm, problem, BaseCase)
