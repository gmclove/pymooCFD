from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.oscill_cyl import OscillCylinder as BaseCase
# from pymooCFD.core.pymooBase import CFDTestProblem

def exec_test():
    study = MinimizeCFD(BaseCase)
    run_dir = 'test_run'
    if run_dir in study.opt_runs:
        opt_run = study.opt_runs[run_dir]
    else:
        xl = [0.1, 0.2]  # lower limits of parameters/variables
        xu = [10, 1]  # upper limits of variables
        prob = study.get_problem(xl, xu)
        alg = study.get_algorithm(n_gen=2, pop_size=3,
                                  n_offsprings=2)
        opt_run = study.new_run(alg, prob, run_dir=run_dir)
    opt_run.test_case.run()
    # opt_run.run_bnd_cases()
    opt_run.run()

def exec_study():
    study = MinimizeCFD(BaseCase)
    # study.run_case('oscill_cyl_f-Strouhal')
    run_dir = 'default_run'
    if run_dir in study.opt_runs:
        opt_run = study.opt_runs[run_dir]
    else:
        xl = [0.1, 0.2]  # lower limits of parameters/variables
        xu = [10, 1]  # upper limits of variables
        prob = study.get_problem(xl, xu)
        alg = study.get_algorithm(n_gen=35, pop_size=50,
                                  n_offsprings=15)
        opt_run = study.new_run(alg, prob, run_dir=run_dir)
    opt_run.test_case.run()
    opt_run.test_case.mesh_study.run()
    opt_run.run_bnd_cases(n_pts=3, doMeshStudy=True)
    opt_run.run()
