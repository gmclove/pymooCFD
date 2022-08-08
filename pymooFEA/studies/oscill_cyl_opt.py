from pymooFEA.core.minimizeFEA import MinimizeFEA
from pymooFEA.problems.oscill_cyl import OscillCylinder as BaseCase
# from pymooFEA.core.pymooBase import FEATestProblem

def exec_test():
    study = MinimizeFEA(BaseCase)
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
    study = MinimizeFEA(BaseCase)
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
    opt_run.gen_bnd_cases(n_pts=3, getDiags=True)
    opt_run.run_bnd_cases(do_mesh_study=True)
    opt_run.run()
