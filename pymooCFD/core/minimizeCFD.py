from pymooCFD.core.pymooBase import CFDGeneticProblem, CFDGeneticAlgorithm
from pymooCFD.core.optRun import OptRun
from pymooCFD.core.picklePath import PicklePath
from pymooCFD.util.sysTools import yes_or_no

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

import os
import shutil


class MinimizeCFD(PicklePath):
    def __init__(self, CFDCase,
                 # xl, xu,
                 CFDGeneticAlgorithm=CFDGeneticAlgorithm,
                 CFDGeneticProblem=CFDGeneticProblem,
                 dir_path=None,
                 # **kwargs
                 ):
        if dir_path is None:
            dir_path = 'optStudy-' + CFDCase.__name__
        ##########################
        #    RESET ATTRIBUTES    #
        ##########################
        # OptRun(self.get_algorithm(), self.get_problem(), run_path='run00')]
        self.opt_runs = {}
        self.case_runs = {}
        # self.opt_runs = []
        # self.case_runs = []
        self.CFDCase = CFDCase
        self.CFDGeneticAlgorithm = CFDGeneticAlgorithm
        self.CFDGeneticProblem = CFDGeneticProblem
        #####################
        #    PICKLE PATH    #
        #####################
        super().__init__(dir_path)
        # for pp_child in self.opt_runs:
        #     if PicklePath in pp_child.__class__.mro():
        #         pp_child.__init__(pp_child.algorithm, pp_child.problem)
        #     print(run.__dict__)
        #     print(run.algorithm.__dict__)
        #     run.__init__(CFDCase)

    def get_problem(self, xl, xu, **kwargs):
        return self.CFDGeneticProblem(self.CFDCase, xl, xu, **kwargs)

    def get_algorithm(self, **kwargs):
        if 'sampling' not in kwargs:
            sampling = MixedVariableSampling(self.CFDCase.var_type, {
                "real": get_sampling("real_lhs"),  # "real_random"),
                "int": get_sampling("int_random")
            })
        else:
            sampling = kwargs.pop('sampling')
        if 'crossover' not in kwargs:
            crossover = MixedVariableCrossover(self.CFDCase.var_type, {
                "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
                "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
            })
        else:
            crossover = kwargs.pop('crossover')
        if 'mutation' not in kwargs:
            mutation = MixedVariableMutation(self.CFDCase.var_type, {
                "real": get_mutation("real_pm", eta=3.0),
                "int": get_mutation("int_pm", eta=3.0)
            })
        else:
            mutation = kwargs.pop('mutation')
        return self.CFDGeneticAlgorithm(sampling, crossover, mutation,
                                        repair=self.CFDCase.repair, **kwargs)

    def new_run(self, alg, prob, run_dir=None):  # algorithm=None, problem=None,
        # if problem is None:
        #     problem = self.CFDGeneticProblem(self.CFDCase, **kwargs)
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

        # self.algorithm = CFDAlgorithm(sampling, crossover, mutation)
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
        case = self.CFDCase(case_path, x, **kwargs)
        case.run()
        self.case_runs[case_dir] = case
        self.save_self()
        return case
