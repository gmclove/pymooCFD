from pymooCFD.core.pymooBase import CFDGeneticProblem, CFDGeneticAlgorithm
from pymooCFD.core.optRun import OptRun
from pymooCFD.core.picklePath import PicklePath
from pymooCFD.util.sysTools import yes_or_no

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

import os


class MinimizeCFD(PicklePath):
    def __init__(self, CFDCase,
                 # xl, xu,
                 CFDGeneticAlgorithm=CFDGeneticAlgorithm,
                 CFDGeneticProblem=CFDGeneticProblem,
                 dir_path=None,
                 # **kwargs
                 ):
        if dir_path is None:
            dir_path = 'optStudy-'+CFDCase.__name__
        ##########################
        #    RESET ATTRIBUTES    #
        ##########################
        self.opt_runs = []  # OptRun(self.get_algorithm(), self.get_problem(), run_path='run00')]
        self.case_runs = []
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
        sampling = MixedVariableSampling(self.CFDCase.var_type, {
            "real": get_sampling("real_lhs"),  # "real_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(self.CFDCase.var_type, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })

        mutation = MixedVariableMutation(self.CFDCase.var_type, {
            "real": get_mutation("real_pm", eta=3.0),
            "int": get_mutation("int_pm", eta=3.0)
        })
        return self.CFDGeneticAlgorithm(sampling, crossover, mutation, **kwargs)

    def new_run(self, alg, prob, run_dir=None):  # algorithm=None, problem=None,
        # if problem is None:
        #     problem = self.CFDGeneticProblem(self.CFDCase, **kwargs)
        # if algorithm is None:
        #     algorithm = self.algorithm
        if run_dir is None:
            run_dir = 'run'+str(len(self.opt_runs)).zfill(2)
        run_path = os.path.join(self.abs_path, run_dir)
        if run_path in os.listdir(self.abs_path):
            question = 'Run directory already exists. Overwrite?'
            yes = yes_or_no(question)
            if yes:
                os.rmdir(run_path)
        opt_run = OptRun(alg, prob, run_path=run_path)
        self.opt_runs.append(opt_run)
        self.save_self()
        return opt_run

        # self.algorithm = CFDAlgorithm(sampling, crossover, mutation)
    def run_case(self, case_dir, x, **kwargs):
        path = os.path.join(self.abs_path, case_dir)
        case = self.CFDCase(path, x, **kwargs)
        case.run()
        self.case_runs.append(case)
        self.save_self()
        return case
