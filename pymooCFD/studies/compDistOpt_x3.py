# @Author: glove
# @Date:   2021-12-15T14:59:23-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T17:20:34-05:00

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.core.problem import Problem
import numpy as np
from pymooCFD.studies.compDistOpt_x2 import CompDistSLURM
from pymooCFD.core.optRun import OptRun


# class LocalCompDistOpt(OptRun):
#     pass
# def __init__(self):
#     super().__init__()


class CompDistSLURM_YALES2(CompDistSLURM):
    base_case_path = 'base_cases/osc-cyl_base'
    inputFile = '2D_cylinder.in'
    jobFile = 'jobslurm.sh'
    datFile = 'FORCES_temporal.txt'

    n_var = 3
    var_labels = np.append(CompDistSLURM.var_labels,
                           'Number of Elements per Group')
    var_type = np.append(CompDistSLURM.var_type, 'real')
    xl = np.append(CompDistSLURM.xl, 50)
    xu = np.append(CompDistSLURM.xu, 1000)

    # n_constr = 0
    #
    # solveExternal = True
    # solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
    # def __init__(self, base_case_path, caseDir, x,
    #              *args, **kwargs):
    #     super().__init__(base_case_path, caseDir, x,
    #                      *args, **kwargs)

    def _preProc(self):
        # ntasks = self.x[0]
        # c = self.x[1]
        # # read job lines
        # job_lines = self.job_lines_rw
        # if job_lines:
        #     kw_lines = self.findKeywordLines(
        #         '#SBATCH --cpus-per-task', job_lines)
        #     for line_i, line in kw_lines:
        #         job_lines[line_i] = f'#SBATCH --cpu-per-task={c}'
        #     kw_lines = self.findKeywordLines('#SBATCH -c', job_lines)
        #     for line_i, line in kw_lines:
        #         job_lines[line_i] = f'#SBATCH --cpu-per-task={c}'
        #     kw_lines = self.findKeywordLines('#SBATCH --ntasks', job_lines)
        #     for line_i, line in kw_lines:
        #         job_lines[line_i] = f'#SBATCH --ntasks={ntasks}'
        #     kw_lines = self.findKeywordLines('#SBATCH -n', job_lines)
        #     for line_i, line in kw_lines:
        #         job_lines[line_i] = f'#SBATCH --ntasks={ntasks}'
        #     # write job lines
        #     self.job_lines_rw = job_lines
        # else:
        #     self.solverExecCmd.insert(
        #         '-c', 1).insert(str(c), 2).insert('-n', 3).insert(str(ntasks), 4)

        super()._preProc()
        in_lines = self.input_lines_rw
        kw_lines = self.findKeywordLines('NELEMENTPERGROUP', in_lines)
        for line_i, _ in kw_lines:
            in_lines[line_i] = f'NELEMENTPERGROUP = {self.x[2]}'
        # self.job_lines_rw = [
        #     '#!/bin/bash',
        #     "#SBATCH --partition=ib --constraint='ib&haswell_1'",
        #     f'#SBATCH --cpus-per-task={c}',
        #     f'#SBATCH --ntasks={ntasks}',
        #     '#SBATCH --time=03:00:00',
        #     '#SBATCH --mem-per-cpu=2G',
        #     '#SBATCH --job-name=compDistOpt',
        #     '#SBATCH --output=slurm.out',
        #     'module load ansys/fluent-21.2.0',
        #     'cd $SLURM_SUBMIT_DIR',
        #     'time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out'
        # ]
        # ##### Write Entire Input File #####
        # # useful if input file is short enough
        # x_mid = [1.55, 0.55]
        # outVel = x_mid[1]
        # coflowVel = 0.005  # outVel*(2/100)
        # self.input_lines_rw = [
        #     # IMPORT
        #     f'/file/import ideas-universal {self.meshFile}',
        #     # AUTO-SAVE
        #     '/file/auto-save case-frequency if-case-is-modified',
        #     '/file/auto-save data-frequency 1000',
        #     # MODEL
        #     '/define/models axisymmetric y',
        #     '/define/models/viscous kw-sst y',
        #     # species
        #     '/define/models/species species-transport y mixture-template',
        #     '/define/materials change-create air scalar n n n n n n n n',
        #     '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
        #     # BOUNDARY CONDITIONS
        #     # outlet
        #     '/define/boundary-conditions/modify-zones/zone-type outlet pressure-outlet ;outflow',
        #     # coflow
        #     '/define/boundary-conditions/modify-zones/zone-type coflow velocity-inlet',
        #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel*2} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
        #     # inlet
        #     '/define/boundary-conditions/modify-zones/zone-type inlet velocity-inlet',
        #     f'/define/boundary-conditions velocity-inlet inlet n n y y n {outVel} n 0 n 300 n n y 5 10 n n 1',
        #     # axis
        #     '/define/boundary-conditions/modify-zones/zone-type axis axis',
        #     # INITIALIZE
        #     '/solve/initialize/hyb-initialization',
        #     # CHANGE CONVERGENCE CRITERIA
        #     '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
        #     # SOLVE
        #     '/solve/iterate 1000',
        #     # change convergence, methods and coflow speed
        #     '/solve/set discretization-scheme species-0 6',
        #     '/solve/set discretization-scheme mom 6',
        #     '/solve/set discretization-scheme k 6',
        #     '/solve/set discretization-scheme omega 6',
        #     '/solve/set discretization-scheme temperature 6',
        #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
        #     '/solve/iterate 4000',
        #     # EXPORT
        #     f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
        #     'OK',
        #     f'/file write-case-data {self.datFile}',
        #     'OK',
        #     '/exit',
        #     'OK'
        # ]

    def _postProc(self):
        self.f = self.solnTime


class CompDistOpt(OptRun):
    def __init__(self, algorithm, BaseCase,
                 # *args, **kwargs
                 ):
        super().__init__(algorithm, BaseCase,
                         # optName='CompDistSOO-test',
                         n_opt=20,
                         # base_case_path='base_cases/osc-cyl_base',
                         # optDatDir='cyl-opt_run',
                         # *args, **kwargs
                         )


MyOptRun = CompDistOpt
BaseCase = CompDistSLURM

####################################
#    Genetic Algorithm Criteria    #
####################################
n_gen = 20
pop_size = 40
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
        # out['F'] = np.zeros((BaseCase.n_obj, pop_size))


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

###################
#    ALGORITHM    #
###################
algorithm = NSGA2(
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
optRun = MyOptRun(algorithm, BaseCase)
