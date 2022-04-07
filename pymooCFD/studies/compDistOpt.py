# @Author: glove
# @Date:   2021-12-15T14:59:23-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T17:20:34-05:00

from pymooCFD.core.optStudy import OptStudy

# class LocalCompDistOpt(OptStudy):
#
#     def __init__(self):
#         super().__init__()
#
#     def execute(self):
#         self.singleNodeExec()


from pymooCFD.core.cfdCase import CFDCase


class FluentCompDistSLURM(CFDCase):
    # baseCaseDir = 'base_cases/'
    # datFile = ''

    n_var = 2
    # , 'Time Step']
    var_labels = ['Number of Tasks', 'Number of CPUs per Task']
    var_type = ['int', 'int']
    xl = [1, 1]
    xu = [50, 50]

    n_obj = 1
    obj_labels = ['Wall Time']  # , 'Fidelity']

    n_constr = 0

    solveExternal = True
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']

    def __init__(self, baseCaseDir, caseDir, x,
                 # *args, **kwargs
                 ):
        super().__init__(baseCaseDir, caseDir, x,
                         *args, **kwargs)

    def _preProc(self):
        ntasks = self.x[0]
        c = self.x[1]
        self.jobLines = [
            '#!/bin/bash',
            "#SBATCH --partition=ib --constraint='ib&haswell_1'",
            f'#SBATCH --cpus-per-task={c}',
            f'#SBATCH --ntasks={ntasks}',
            '#SBATCH --time=03:00:00',
            '#SBATCH --mem-per-cpu=2G',
            '#SBATCH --job-name=compDistOpt',
            '#SBATCH --output=slurm.out',
            'module load ansys/fluent-21.2.0',
            'cd $SLURM_SUBMIT_DIR',
            'time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out'
        ]
        ##### Write Entire Input File #####
        # useful if input file is short enough
        x_mid = [1.55, 0.55]
        outVel = x_mid[1]
        coflowVel = 0.005  # outVel*(2/100)
        self.inputLines = [
            # IMPORT
            f'/file/import ideas-universal {self.meshFile}',
            # AUTO-SAVE
            '/file/auto-save case-frequency if-case-is-modified',
            '/file/auto-save data-frequency 1000',
            # MODEL
            '/define/models axisymmetric y',
            '/define/models/viscous kw-sst y',
            # species
            '/define/models/species species-transport y mixture-template',
            '/define/materials change-create air scalar n n n n n n n n',
            '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
            # BOUNDARY CONDITIONS
            # outlet
            '/define/boundary-conditions/modify-zones/zone-type outlet pressure-outlet ;outflow',
            # coflow
            '/define/boundary-conditions/modify-zones/zone-type coflow velocity-inlet',
            f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel*2} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
            # inlet
            '/define/boundary-conditions/modify-zones/zone-type inlet velocity-inlet',
            f'/define/boundary-conditions velocity-inlet inlet n n y y n {outVel} n 0 n 300 n n y 5 10 n n 1',
            # axis
            '/define/boundary-conditions/modify-zones/zone-type axis axis',
            # INITIALIZE
            '/solve/initialize/hyb-initialization',
            # CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
            # SOLVE
            '/solve/iterate 1000',
            # change convergence, methods and coflow speed
            '/solve/set discretization-scheme species-0 6',
            '/solve/set discretization-scheme mom 6',
            '/solve/set discretization-scheme k 6',
            '/solve/set discretization-scheme omega 6',
            '/solve/set discretization-scheme temperature 6',
            f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
            '/solve/iterate 4000',
            # EXPORT
            f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
            'OK',
            f'/file write-case-data {self.datFile}',
            'OK',
            '/exit',
            'OK'
        ]

    def _postProc(self):
        pass

    def _execDone(self):

        return True
