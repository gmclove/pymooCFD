import fnmatch
import numpy as np
import os

from pymooCFD.core.cfdCase import FluentCase
from pymooCFD.util.handleData import saveTxt


class RANS_k_eps(FluentCase):
    base_case_path = os.path.join(os.path.dirname(__file__), 'base_cases',
                                  'rans_k-eps_3D-room')

    n_var = 2
    var_labels = ['Turbulent Viscosity Constant', 'Number of Iterations']
    var_type = ['real', 'int']
    # xl = [0.09 * 0.8, 4_000]
    # xu = [0.09 * 1.2, 30_000]

    obj_labels = ['Average of Residuals', 'Wall Time']
    n_obj = 2

    n_constr = 0

    externalSolver = True
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
    nTasks = 4

    def __init__(self, caseDir, x, meshSF=1.0, *args, **kwargs):
        super().__init__(caseDir, x,
                         meshSF=meshSF,
                         jobFile='jobslurm.sh',
                         inputFile='exec.jou',
                         datFile='residuals.dat',
                         *args, **kwargs)

    def _preProc(self):
        self.input_lines_rw = [
            '/file/read rans_k-eps.cas.h5',
            ';DEFINE turbulence solver',
            '/define/models/viscous/ke-standard y',
            f'(rpsetvar\' kecmu {self.x[0]})',
            # CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual check-convergence n n n n n n',
            ';INITIALIZE',
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
        self.job_lines_rw = \
            ['#!/bin/bash',
             "#SBATCH --partition=ib",
             "#SBATCCH --constraint='ib&sandybridge|haswell_1|haswell_2'",
             '#SBATCH --cpus-per-task=2',
             '#SBATCH --ntasks=10',
             '#SBATCH --time=10:00:00',
             '#SBATCH --mem-per-cpu=2G',
             '#SBATCH --job-name=RANSroom',
             '#SBATCH --output=slurm.out',
             'module load ansys/fluent-21.2.0',
             'cd $SLURM_SUBMIT_DIR',
             'time fluent 3ddp -g -pdefault -t$SLURM_NTASKS -slurm -i exec.jou > run.out'
             ]

    def _postProc(self):
        # residuals_dict = self.residuals_file_to_dict(self.datPath)
        dat = np.genfromtxt(self.datPath)
        if dat[-1, 0] < 1000:
            self.logger.error('LESS THAN 1000 ITERATIONS PREFORMED')
        rel_dat = dat[-2000:, 1:]
        saveTxt(self.abs_path, 'residual_avgs.txt', np.mean(rel_dat, axis=0))
        avg = np.mean(rel_dat)
        path = os.path.join(self.abs_path, 'slurm.out')
        with open(path) as f:
            lines = f.readlines()
        matches = fnmatch.filter(lines, '*real*m*s*')
        # t_tot = None
        for match in matches:
            try:
                _, t_str = match.split()
                l_t_min = t_str.split('m')
                t_min = int(l_t_min[0])
                l_t_sec = l_t_min[-1].split('s')
                t_sec = float(l_t_sec[0])
                t_tot = t_min * 60 + t_sec
            except AttributeError as err:
                self.logger.error(err)
        self.f = [avg, t_tot]
        return self.f

    def _solveDone(self):
        print('solve done check')
        if self.f is not None:
            t_tot = self.f[2]
            if t_tot < 100:
                self.logger.error(f'{t_tot} is too small')
                return False
        else:
            return super()._solveDone()

    # @staticmethod
    # def residuals_file_to_dict(f_path):
    #     with open(f_path, 'r') as f:
    #         dat_lines = f.readlines()
    #         dat = {}
    #         for line in dat_lines:
    #             if line[:2] == '((':
    #                 label = line.split('"')[1]
    #                 dat[label] = []
    #             try:
    #                 int(line[0])
    #             except ValueError:
    #                 continue
    #             split_line = line.split()
    #             l = [int(split_line[0]), float(split_line[1])]
    #             # l = np.array(l)
    #             dat[label].append(l)
    #     for key, val in dat.items():
    #         dat[key] = np.array(val)
    #     return dat
    # with open() as f:
    #


BaseCase = RANS_k_eps


class RANS_k_eps_x4(RANS_k_eps):
    n_var = 4
    var_labels = ['Turbulent Viscosity Constant', 'C1 Epsilon', 'C2 Epsilon',
                  'Number of Iterations']
    var_type = ['real', 'real', 'real', 'int']    # 0.09, 1.44, 1.92

    def _preProc(self):
        self.input_lines_rw = [
            '/file/read rans_k-eps.cas.h5',
            ';DEFINE turbulence solver',
            '/define/models/viscous/ke-standard y',
            f'(rpsetvar\' kecmu {self.x[0]})',
            f'(rpsetvar\' kec1 {self.x[1]})',
            f'(rpsetvar\' kec2 {self.x[2]})',
            # CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual check-convergence n n n n n n',
            ';INITIALIZE',
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
        self.job_lines_rw = \
            ['#!/bin/bash',
             "#SBATCH --partition=ib",
             "#SBATCCH --constraint='ib&sandybridge|haswell_1|haswell_2'",
             '#SBATCH --cpus-per-task=2',
             '#SBATCH --ntasks=10',
             '#SBATCH --time=10:00:00',
             '#SBATCH --mem-per-cpu=2G',
             '#SBATCH --job-name=RANSroom',
             '#SBATCH --output=slurm.out',
             'module load ansys/fluent-21.2.0',
             'cd $SLURM_SUBMIT_DIR',
             'time fluent 3ddp -g -pdefault -t$SLURM_NTASKS -slurm -i exec.jou > run.out'
             ]
