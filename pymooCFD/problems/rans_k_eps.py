import fnmatch
import numpy as np
import os
import matplotlib.pyplot as plt

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

    n_constr = 1

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
        if os.path.exists(self.datPath):
            os.remove(self.datPath)
        self.input_lines_rw = [
            '/file/read rans_k-eps.cas.h5',
            ';DEFINE turbulence solver',
            '/define/models/viscous/ke-standard y',
            f'(rpsetvar\' kecmu {self.x[0]})',
            # SAVE RESIDUALS
            '/file/read-macro init.scm',
            '/solve/execute-commands add-edit save-resid 1 "iteration" "file read-journal residual.jou"',
            # CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual check-convergence n n n n n n',
            ';INITIALIZE',
            '/solve/initialize/compute-defaults/velocity-inlet inlet',
            '/solve/initialize/initialize-flow',
            ';SOLVE',
            f'/solve/iterate {int(self.x[1])} n y',
            # ';SAVE residuals',
            # '(let',
            # '((writefile (lambda (p)',
            # '(define np (length (residual-history "iteration")))',
            # '(let loop ((i 0))',
            # '(if (not (= i np))',
            # '(begin (define j (+ i 1))',
            # '(display (list-ref (residual-history "iteration") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "continuity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "x-velocity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "y-velocity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "z-velocity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "k") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "epsilon") (- np j)) p)',
            # '(newline p)',
            # '(loop (+ i 1))',
            # ')',
            # ')',
            # ')',
            # ') )',
            # '(output-port (open-output-file "residuals.dat")))',
            # '(writefile output-port)',
            # '(close-output-port output-port))',
            '/file write-case-data fluent-data.cas.h5',
            '/exit y'
        ]

        ####### Slurm Job Lines #########
        self.job_lines_rw = \
            ['#!/bin/bash',
             "#SBATCH --partition=ib",
             "#SBATCCH --constraint='ib&sandybridge|haswell_1|haswell_2'",
             '#SBATCH --cpus-per-task=1',
             '#SBATCH --ntasks=15',
             '#SBATCH --time=20:00:00',
             '#SBATCH --mem-per-cpu=2G',
             '#SBATCH --job-name=RANSroom',
             '#SBATCH --output=slurm.out',
             'module load ansys/fluent-21.2.0',
             'cd $SLURM_SUBMIT_DIR',
             'time fluent 3ddp -g -pdefault -t$SLURM_NTASKS -slurm -i exec.jou > run.out'
             ]

    def _postProc(self):
        # residuals_dict = self.residuals_file_to_dict(self.datPath)
        dat = np.genfromtxt(self.datPath, skip_header=1)
        # print(len(dat))
        # print(dat)
        # PLOT
        its = np.arange(1, len(dat)+1)
        labels = ['continuity', 'x-velocity', 'y-velocity', 'z-velocity', 'k',
                  'epsilon']
        fig, ax = plt.subplots()
        for i, col in enumerate(dat.T):
            ax.plot(its, col, label=labels[i])
        ax.set_title('Scaled Residuals')
        ax.legend()
        path = os.path.join(self.abs_path, 'residuals.png')
        fig.savefig(path)
        fig, ax = plt.subplots()
        for i, col in enumerate(dat[-2000:].T):
            ax.plot(its[-2000:], col, label=labels[i])
        ax.set_title('Scaled Residuals: Last 2,000 Iterations')
        ax.legend()
        path = os.path.join(self.abs_path, 'residuals-final.png')
        fig.savefig(path)
        fig, ax = plt.subplots()
        for i, col in enumerate(dat[:2000].T):
            ax.plot(its[:2000], col, label=labels[i])
        ax.set_title('Scaled Residuals: First 2,000 Iterations')
        ax.legend()
        path = os.path.join(self.abs_path, 'residuals-initial.png')
        fig.savefig(path)
        # Objective 1: residuals average
        if len(dat) < 1000:
            self.logger.error('LESS THAN 1000 ITERATIONS PREFORMED')
        rel_dat = dat[-2000:]
        saveTxt(self.abs_path, 'residual_avgs.txt', np.mean(rel_dat, axis=0))
        avg = np.mean(rel_dat)
        # Objective 2: wall time
        path = os.path.join(self.abs_path, 'slurm.out')
        with open(path) as f:
            lines = f.readlines()
        matches = fnmatch.filter(lines, '*real*m*s*')
        # t_tot = None
        if matches:
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
        else:
            self.logger.error('NO MATCHES TO TIME STRING FOUND')
        self.f = [avg, t_tot]
        self.g = avg - 1e-5
        return self.f

    # def _solveDone(self):
    #     print('solve done check')
    #     if self.f is not None:
    #         t_tot = self.f[2]
    #         if t_tot < 100:
    #             self.logger.error(f'{t_tot} is too small')
    #             return False
    #     return super()._solveDone()

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
        super()._preProc()
        self.input_lines_rw = [
            '/file/read rans_k-eps.cas.h5',
            ';DEFINE turbulence solver',
            '/define/models/viscous/ke-standard y',
            f'(rpsetvar\' kecmu {self.x[0]})',
            f'(rpsetvar\' kec1 {self.x[1]})',
            f'(rpsetvar\' kec2 {self.x[2]})',
            # SAVE RESIDUALS
            '/file/read-macro init.scm',
            '/solve/execute-commands add-edit save-resid 1 "iteration" "file read-journal residual.jou"',
            # CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual check-convergence n n n n n n',
            ';INITIALIZE',
            '/solve/initialize/compute-defaults/velocity-inlet inlet',
            '/solve/initialize/initialize-flow',
            ';SOLVE',
            f'/solve/iterate {int(self.x[3])} n y',
            # ';SAVE residuals',
            # '(let',
            # '((writefile (lambda (p)',
            # '(define np (length (residual-history "iteration")))',
            # '(let loop ((i 0))',
            # '(if (not (= i np))',
            # '(begin (define j (+ i 1))',
            # '(display (list-ref (residual-history "iteration") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "continuity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "x-velocity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "y-velocity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "z-velocity") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "k") (- np j)) p) (display " " p)',
            # '(display (list-ref (residual-history "epsilon") (- np j)) p)',
            # '(newline p)',
            # '(loop (+ i 1))',
            # ')',
            # ')',
            # ')',
            # ') )',
            # '(output-port (open-output-file "residuals.dat")))',
            # '(writefile output-port)',
            # '(close-output-port output-port))',
            '/file write-case-data fluent-data.cas.h5',
            '/exit y'
        ]

        # ####### Slurm Job Lines #########
        # self.job_lines_rw = \
        #     ['#!/bin/bash',
        #      "#SBATCH --partition=ib",
        #      "#SBATCCH --constraint='ib&sandybridge|haswell_1|haswell_2'",
        #      '#SBATCH --cpus-per-task=1',
        #      '#SBATCH --ntasks=15',
        #      '#SBATCH --time=20:00:00',
        #      '#SBATCH --mem-per-cpu=2G',
        #      '#SBATCH --job-name=RANSroom',
        #      '#SBATCH --output=slurm.out',
        #      'module load ansys/fluent-21.2.0',
        #      'cd $SLURM_SUBMIT_DIR',
        #      'time fluent 3ddp -g -pdefault -t$SLURM_NTASKS -slurm -i exec.jou > run.out'
        #      ]
