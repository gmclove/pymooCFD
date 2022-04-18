from pymooCFD.core.cfdCase import YALES2Case
from pymooCFD.problems.oscill_cyl import OscillCylinder_SLURM
import numpy as np

def get_CompDistYALES2_SLURM(BaseCase):
    global CompDistYALES2_SLURM
    assert YALES2Case in BaseCase.mro()

    class CompDistYALES2_SLURM(BaseCase):
        # base_case_path = os.path.join(os.path.dirname(__file__), 'base_cases', 'osc-cyl_base')
        # inputFile = 'jet_rans-axi_sym.jou'
        # jobFile = 'jobslurm.sh'
        # datFile = 'jet_rans-axi_sym.cgns'

        n_var = 3
        var_labels = ['Number of Tasks', 'Number of CPUs per Task',
                      'Number of Elements per Group']
        var_type = ['int', 'int', 'int']

        n_obj = 2
        obj_labels = ['Solve Time', 'Total Number of CPUs']  # , 'Fidelity']

        n_constr = 1

        externalSolver = True
        solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
        nTasks = 10

        def __init__(self, caseDir, x, meshSF=1.0,
                     # *args, **kwargs
                     ):
            super().__init__(caseDir, x,
                             meshSF=meshSF,
                             # jobFile='jobslurm.sh',
                             # meshFile='2D_cylinder.msh22',
                             # inputFile='2D_cylinder.in'
                             # *args, **kwargs
                             )

        def _preProc(self):
            ntasks = self.x[0]
            c = self.x[1]
            # read job lines
            job_lines = self.job_lines_rw
            if job_lines:
                newLine = f'#SBATCH --cpus-per-task={c}'
                kws = ['#SBATCH --cpus-per-task', '#SBATCH -c']
                job_lines = self.findAndReplaceKeywordLines(
                    job_lines, newLine, kws, insertIndex=1)

                newLine = f'#SBATCH --ntasks={ntasks}'
                kws = ['#SBATCH --ntasks', '#SBATCH -n']
                job_lines = self.findAndReplaceKeywordLines(
                    job_lines, newLine, kws, insertIndex=1)
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
                self.job_lines_rw = job_lines
            elif self.jobFile in self.solverExecCmd:
                self.solverExecCmd.insert(
                    1, '-c').insert(2, str(c)).insert(3, '-n').insert(4, str(ntasks))
            else:
                self.logger.exception('INCOMPLETE: PRE-PROCESSING')
            # number of elements per group parameter
            in_lines = self.input_lines_rw
            # print(in_lines)
            kw_lines = self.findKeywordLines('NELEMENTPERGROUP', in_lines)
            for line_i, _ in kw_lines:
                in_lines[line_i] = f'NELEMENTPERGROUP = {self.x[2]}'
            self.input_lines_rw = in_lines

        def _postProc(self):
            nCPUs = self.x[0] * self.x[1]
            # if self.solnTime < 100:
            #     self.logger.exception(f'{self.solnTime} - too small')
            self.f = [self.wallTime, nCPUs]
            self.g = 500 - self.wallTime

        # def _solveDone(self):
        #     if self.datPath is not None:
        #         if os.path.exists(self.datPath):
        #             return True
        #     else:
        #         return True
    return CompDistYALES2_SLURM


CompDistSLURM = get_CompDistYALES2_SLURM(OscillCylinder_SLURM)


# class CompDistSLURM_YALES2(CompDistSLURM):
#     # base_case_path = 'base_cases/osc-cyl_base'
#     # inputFile = '2D_cylinder.in'
#     # jobFile = 'jobslurm.sh'
#     # datFile = 'FORCES_temporal.txt'
#
#     n_var = 3
#     var_labels = np.append(CompDistSLURM.var_labels,
#                            'Number of Elements per Group')
#     var_type = np.append(CompDistSLURM.var_type, 'real')
#     # xl = np.append(CompDistSLURM.xl, 50)
#     # xu = np.append(CompDistSLURM.xu, 1000)
#
#     # n_constr = 0
#     #
#     # solveExternal = True
#     # solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
#     # def __init__(self, base_case_path, caseDir, x,
#     #              *args, **kwargs):
#     #     super().__init__(base_case_path, caseDir, x,
#     #                      *args, **kwargs)
#
#     def _preProc(self):
#         # ntasks = self.x[0]
#         # c = self.x[1]
#         # # read job lines
#         # job_lines = self.job_lines_rw
#         # if job_lines:
#         #     kw_lines = self.findKeywordLines(
#         #         '#SBATCH --cpus-per-task', job_lines)
#         #     for line_i, line in kw_lines:
#         #         job_lines[line_i] = f'#SBATCH --cpu-per-task={c}'
#         #     kw_lines = self.findKeywordLines('#SBATCH -c', job_lines)
#         #     for line_i, line in kw_lines:
#         #         job_lines[line_i] = f'#SBATCH --cpu-per-task={c}'
#         #     kw_lines = self.findKeywordLines('#SBATCH --ntasks', job_lines)
#         #     for line_i, line in kw_lines:
#         #         job_lines[line_i] = f'#SBATCH --ntasks={ntasks}'
#         #     kw_lines = self.findKeywordLines('#SBATCH -n', job_lines)
#         #     for line_i, line in kw_lines:
#         #         job_lines[line_i] = f'#SBATCH --ntasks={ntasks}'
#         #     # write job lines
#         #     self.job_lines_rw = job_lines
#         # else:
#         #     self.solverExecCmd.insert(
#         #         '-c', 1).insert(str(c), 2).insert('-n', 3).insert(str(ntasks), 4)
#         super()._preProc()
#         in_lines = self.input_lines_rw
#         # print(in_lines)
#         kw_lines = self.findKeywordLines('NELEMENTPERGROUP', in_lines)
#         for line_i, _ in kw_lines:
#             in_lines[line_i] = f'NELEMENTPERGROUP = {self.x[2]}'
#         self.input_lines_rw = in_lines
#         # self.job_lines_rw = [
#         #     '#!/bin/bash',
#         #     "#SBATCH --partition=ib --constraint='ib&haswell_1'",
#         #     f'#SBATCH --cpus-per-task={c}',
#         #     f'#SBATCH --ntasks={ntasks}',
#         #     '#SBATCH --time=03:00:00',
#         #     '#SBATCH --mem-per-cpu=2G',
#         #     '#SBATCH --job-name=compDistOpt',
#         #     '#SBATCH --output=slurm.out',
#         #     'module load ansys/fluent-21.2.0',
#         #     'cd $SLURM_SUBMIT_DIR',
#         #     'time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out'
#         # ]
#         # ##### Write Entire Input File #####
#         # # useful if input file is short enough
#         # x_mid = [1.55, 0.55]
#         # outVel = x_mid[1]
#         # coflowVel = 0.005  # outVel*(2/100)
#         # self.input_lines_rw = [
#         #     # IMPORT
#         #     f'/file/import ideas-universal {self.meshFile}',
#         #     # AUTO-SAVE
#         #     '/file/auto-save case-frequency if-case-is-modified',
#         #     '/file/auto-save data-frequency 1000',
#         #     # MODEL
#         #     '/define/models axisymmetric y',
#         #     '/define/models/viscous kw-sst y',
#         #     # species
#         #     '/define/models/species species-transport y mixture-template',
#         #     '/define/materials change-create air scalar n n n n n n n n',
#         #     '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
#         #     # BOUNDARY CONDITIONS
#         #     # outlet
#         #     '/define/boundary-conditions/modify-zones/zone-type outlet pressure-outlet ;outflow',
#         #     # coflow
#         #     '/define/boundary-conditions/modify-zones/zone-type coflow velocity-inlet',
#         #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel*2} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
#         #     # inlet
#         #     '/define/boundary-conditions/modify-zones/zone-type inlet velocity-inlet',
#         #     f'/define/boundary-conditions velocity-inlet inlet n n y y n {outVel} n 0 n 300 n n y 5 10 n n 1',
#         #     # axis
#         #     '/define/boundary-conditions/modify-zones/zone-type axis axis',
#         #     # INITIALIZE
#         #     '/solve/initialize/hyb-initialization',
#         #     # CHANGE CONVERGENCE CRITERIA
#         #     '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
#         #     # SOLVE
#         #     '/solve/iterate 1000',
#         #     # change convergence, methods and coflow speed
#         #     '/solve/set discretization-scheme species-0 6',
#         #     '/solve/set discretization-scheme mom 6',
#         #     '/solve/set discretization-scheme k 6',
#         #     '/solve/set discretization-scheme omega 6',
#         #     '/solve/set discretization-scheme temperature 6',
#         #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
#         #     '/solve/iterate 4000',
#         #     # EXPORT
#         #     f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
#         #     'OK',
#         #     f'/file write-case-data {self.datFile}',
#         #     'OK',
#         #     '/exit',
#         #     'OK'
#         # ]
#
#     # def _postProc(self):
#     #     self.f = self.solnTime
