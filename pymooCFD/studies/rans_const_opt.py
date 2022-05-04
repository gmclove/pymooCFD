# @Author: glove
# @Date:   2021-12-14T16:02:45-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:54-05:00
from pymooCFD.problems.rans_k_eps import RANS_k_eps
from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.core.pymooBase import CFDTestProblem


def exec_test():
    study = MinimizeCFD(RANS_k_eps, CFDGeneticProblem=CFDTestProblem)
    # if study.opt_runs:
    #     opt_run = study.opt_runs[0]
    # else:
    # xl = [0.09 * 0.8, 5]
    # xu = [0.09 * 1.2, 10]
    # prob = study.get_problem(xl, xu)
    # alg = study.get_algorithm(n_gen=2, pop_size=3,
    #                           n_offsprings=2)
    #
    # opt_run = study.new_run(alg, prob, run_dir='test-run')
    # opt_run.test_case.run()
    # opt_run.run_bnd_cases()
    # opt_run.run()


def exec_study():
    study = MinimizeCFD(RANS_k_eps)
    # if study.opt_runs:
    #     opt_run = study.opt_runs[0]
    # else:
    # xl = [0.09 * 0.8, 4_000]
    # xu = [0.09 * 1.2, 30_000]
    # prob = study.get_problem(xl, xu)
    # alg = study.get_algorithm(n_gen=20, pop_size=40,
    #                           n_offsprings=8)
    # opt_run = study.new_run(alg, prob)
    # opt_run.test_case.run()
    # opt_run.run_bnd_cases()
    # opt_run.run()
