from pymooCFD.core.pymooBase import CFDProblem_GA, CFDAlgorithm
from pymooCFD.core.optStudy import OptStudy
from pymooCFD.core.picklePath import PicklePath


class MinimizeCFD(PicklePath):
    def __init__(self, CFDCase,
                 algorithm=CFDAlgorithm(), #Problem,
                 dir_path=None):
        if dir_path is None:
            dir_path = 'optStudy-'+self.__class__.__name__
        super().__init__(dir_path)
        self.CFDCase = CFDCase
        ###################
        #    OPERATORS    #
        ###################
        from pymoo.factory import get_sampling, get_crossover, get_mutation
        from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

        sampling = MixedVariableSampling(CFDCase.var_type, {
            "real": get_sampling("real_lhs"),  # "real_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(CFDCase.var_type, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })

        mutation = MixedVariableMutation(CFDCase.var_type, {
            "real": get_mutation("real_pm", eta=3.0),
            "int": get_mutation("int_pm", eta=3.0)
        })
        algorithm.sampling = sampling
        algorithm.crossover = crossover
        algorithm.mutation = mutation
        self.algorithm = algorithm

        self.problem = CFDProblem_GA(CFDCase)

        self.runs = [OptStudy(algorithm, CFDCase, runDir='run00')]
        # self.save_self()

    def new_problem(self):
        self.logger.info('INITIALIZING NEW OPTIMIZATION PROBLEM')
        self.problem = CFDProblem_GA(CFDCase)
        # self.save_self()
        return self.problem

    def new_run(self, algorithm=None, problem=None, run_dir=None):
        if problem is None:
            problem = self.problem
        if algorithm is None:
            algorithm = self.algorithm
        if run_dir is None:
            run_dir = 'run'+str(len(self.runs)).zfill(2)
        if run_dir in os.listdir(self.abs_path):
            question = 'Run directory already exists. Overwrite?'
            yes = yes_or_no(question)
            if yes:
                os.rmdir(run_dir)
        self.runs.append(OptStudy(algorithm, problem, runDir='run00'))
        self.save_self()

        # self.algorithm = CFDAlgorithm(sampling, crossover, mutation)
