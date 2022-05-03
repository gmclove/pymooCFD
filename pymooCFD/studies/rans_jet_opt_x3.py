from pymooCFD.problems.rans_jet import RANSJet_x3 as BaseCase
from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.core.pymooBase import CFDTestProblem
import os


def exec_test(test_run_dir='test_run', **kwargs):
    study = MinimizeCFD(BaseCase, CFDGeneticProblem=CFDTestProblem)
    study.run_case('RANS_equiv_BCs', [0.02, 0.2, 1])
    if test_run_dir in study.opt_runs:
        opt_run = study.opt_runs[test_run_dir]
    else:
        alg = study.get_algorithm(n_gen=2, pop_size=3, n_offsprings=2)
        xl = [0.002, 0.1, 0.5]  # lower limits of parameters/variables
        xu = [0.04, 0.6, 1.5]  # upper limits of variables
        prob = study.get_problem(xl, xu, **kwargs)
        opt_run = study.new_run(alg, prob, run_dir=test_run_dir, **kwargs)
    opt_run.test_case.run()
    # opt_run.test_case.mesh_study.run()
    opt_run.run_bnd_cases()
    opt_run.run()


def exec_study(**kwargs):
    study = MinimizeCFD(BaseCase)
    run_dir = 'default_run'
    if run_dir in study.opt_runs:
        opt_run = study.opt_runs[run_dir]
    else:
        xl = [0.002, 0.1, 0.5]  # lower limits of parameters/variables
        xu = [0.04, 0.6, 1.5]  # upper limits of variables
        alg = study.get_algorithm(n_gen=35, pop_size=30, n_offsprings=8)
        prob = study.get_problem(xl, xu)
        opt_run = study.new_run(alg, prob, run_dir=run_dir, **kwargs)
    opt_run.test_case.run()
    # opt_run.test_case.mesh_study.run()
    opt_run.run_bnd_cases()
    opt_run.run()
