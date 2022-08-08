from pymooFEA.core.pymooBase import FEAGeneticProblem, FEAGeneticAlgorithm, get_FEAGeneticAlgorithm
from pymooFEA.core.optRun import OptRun
from pymooFEA.core.picklePath import PicklePath
from pymooFEA.util.sysTools import yes_or_no

from pymoo.core.mixed import MixedVariableGA

import os
import shutil


class MinimizeFEA(PicklePath):
    def __init__(self, FEACase,
                 # xl, xu,
                 FEAGeneticAlgorithm=FEAGeneticAlgorithm,
                 FEAGeneticProblem=FEAGeneticProblem,
                 dir_path=None,
                 # **kwargs
                 ):
        if dir_path is None:
            dir_path = 'optStudy-' + FEACase.__name__
        ##########################
        #    RESET ATTRIBUTES    #
        ##########################
        # OptRun(self.get_algorithm(), self.get_problem(), run_path='run00')]
        self.opt_runs = {}
        self.case_runs = {}
        # self.opt_runs = []
        # self.case_runs = []
        self.FEACase = FEACase
        self.FEAGeneticAlgorithm = FEAGeneticAlgorithm
        self.FEAGeneticProblem = FEAGeneticProblem
        #####################
        #    PICKLE PATH    #
        #####################
        super().__init__(dir_path)
        # for pp_child in self.opt_runs:
        #     if PicklePath in pp_child.__class__.mro():
        #         pp_child.__init__(pp_child.algorithm, pp_child.problem)
        #     print(run.__dict__)
        #     print(run.algorithm.__dict__)
        #     run.__init__(FEACase)

    def get_problem(self, xl, xu, **kwargs):
        return self.FEAGeneticProblem(self.FEACase, xl, xu, **kwargs)

    def get_algorithm(self, **kwargs):
        if 'mixed_vars' in kwargs:
            mixed_vars = kwargs.get('mixed_vars')
        else:
            mixed_vars = False
        if mixed_vars:
            return get_FEAGeneticAlgorithm(MixedVariableGA)
        return self.FEAGeneticAlgorithm(repair=self.FEACase.repair, **kwargs)

    def new_run(self, alg, prob, run_dir=None):  # algorithm=None, problem=None,
        # if problem is None:
        #     problem = self.FEAGeneticProblem(self.FEACase, **kwargs)
        # if algorithm is None:
        #     algorithm = self.algorithm
        if run_dir is None:
            run_dir = 'run' + str(len(self.opt_runs)).zfill(2)
        run_path = os.path.join(self.abs_path, run_dir)
        if run_dir in self.opt_runs.keys():
            if run_dir in os.listdir(self.abs_path):
                question = '\n\tRun directory already exists. Overwrite?'
                yes = yes_or_no(question)
                if yes:
                    shutil.rmtree(run_path)
            else:
                question = '\n\tRun in study.opt_runs but run directory does not exists. Continue?'
                yes = yes_or_no(question)
                if not yes:
                    exit()
        opt_run = OptRun(alg, prob, run_path=run_path)
        self.opt_runs[run_dir] = opt_run
        self.save_self()
        return opt_run

        # self.algorithm = FEAAlgorithm(sampling, crossover, mutation)
    def run_case(self, case_dir, x, **kwargs):
        case_path = os.path.join(self.abs_path, case_dir)
        if case_dir in self.case_runs.keys():
            if case_dir in os.listdir(self.abs_path):
                question = '\n\tRun directory already exists. Overwrite?'
                yes = yes_or_no(question)
                if yes:
                    shutil.rmtree(case_path)
            else:
                question = '\n\tRun in study.opt_runs but run directory does not exists. Continue?'
                yes = yes_or_no(question)
                if not yes:
                    exit()
        case = self.FEACase(case_path, x, **kwargs)
        case.run()
        self.case_runs[case_dir] = case
        self.save_self()
        return case
