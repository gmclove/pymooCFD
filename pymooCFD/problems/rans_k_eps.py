import numpy as np

from pymooCFD.core.cfdCase import FluentCase


class RANS_k_eps(FluentCase):
    baseCaseDir = 'base_cases/rans_k-eps_3D-room'

    n_var = 2
    var_labels = ['Turbulent Viscosity Constant', 'Number of Iterations']
    var_type = ['real', 'int']
    xl = [0.09 * 0.9, 4_000]
    xu = [0.09 * 1.1, 10_000]

    obj_labels = ['Average of Residuals', 'Wall Time']
    n_obj = 2

    n_constr = 0

    externalSolver = True
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']

    def __init__(self, caseDir, x, *args, **kwargs):
        super().__init__(caseDir, x,
                         jobFile='jobslurm.sh',
                         inputFile='exec.jou',
                         datFile='residuals.dat',
                         *args, **kwargs)

    def _preProc(self):
        self.inputLines = [
            '/file/read rans_k-eps.cas.h5',
            ';DEFINE turbulence solver',
            '/define/models/viscous/ke-standard y',
            f'(rpsetvar\' kecmu {self.x[0]})',
            ';INITIALIZE'
            '/solve/initialize/compute-defaults/velocity-inlet inlet',
            '/solve/initialize/initialize-flow',
            ';SOLVE',
            f'/solve/iterate {int(self.x[1])} n y',
            ';SAVE residuals',
            '(let',
            '((writefile (lambda (p)',
            '(define np (length (residual-history "iteration")))',
            '(let loop ((i 0))',
            '(if (not (= i np))',
            '(begin (define j (+ i 1))',
            '(display (list-ref (residual-history "iteration") (- np j)) p) (display " " p)',
            '(display (list-ref (residual-history "continuity") (- np j)) p) (display " " p)',
            '(display (list-ref (residual-history "x-velocity") (- np j)) p) (display " " p)',
            '(display (list-ref (residual-history "y-velocity") (- np j)) p) (display " " p)',
            '(display (list-ref (residual-history "z-velocity") (- np j)) p) (display " " p)',
            '(display (list-ref (residual-history "k") (- np j)) p) (display " " p)',
            '(display (list-ref (residual-history "epsilon") (- np j)) p)',
            '(newline p)',
            '(loop (+ i 1))',
            ')',
            ')',
            ')',
            ') )',
            '(output-port (open-output-file "residuals.dat")))',
            '(writefile output-port)',
            '(close-output-port output-port))',
            '/exit y'
        ]

        ####### Slurm Job Lines #########
        self.jobLines = \
            ['#!/bin/bash',
             "#SBATCH --partition=ib",
             "#SBATCCH --constraint='ib&sandybridge|haswell_1|haswell_2'",
             '#SBATCH --cpus-per-task=2',
             '#SBATCH --ntasks=10',
             '#SBATCH --time=00:30:00',
             '#SBATCH --mem-per-cpu=2G',
             '#SBATCH --job-name=rans_room',
             '#SBATCH --output=slurm.out',
             'module load ansys/fluent-21.2.0',
             'cd $SLURM_SUBMIT_DIR',
             'time fluent 3ddp -g -pdefault -t$SLURM_NTASKS -slurm -i exec.jou > run.out'
             ]

    def _postProc(self):
        # residuals_dict = self.residuals_file_to_dict(self.datPath)
        dat = np.genfromtxt('residuals.dat')
        if dat[-1, 0] < 2000:
            self.logger.error('LESS THAN 2000 ITERATIONS PREFORMED')
        # avgs = []
        # for col in dat.T[1:]:
        #     avgs.append(np.mean(col[-2000:]))
        # avg = np.mean(avgs)
        avg = np.mean(dat[:, 1:])
        self.f = [avg, self.solnTime]
        return self.f

    @staticmethod
    def residuals_file_to_dict(f_path):
        with open(f_path, 'r') as f:
            dat_lines = f.readlines()
            dat = {}
            for line in dat_lines:
                if line[:2] == '((':
                    label = line.split('"')[1]
                    dat[label] = []
                try:
                    int(line[0])
                except ValueError:
                    continue
                split_line = line.split()
                l = [int(split_line[0]), float(split_line[1])]
                # l = np.array(l)
                dat[label].append(l)
        for key, val in dat.items():
            dat[key] = np.array(val)
        return dat
        # with open() as f:
        #


BaseCase = RANS_k_eps
