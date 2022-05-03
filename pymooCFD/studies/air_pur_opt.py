from pymooCFD.problems.air_purifier import Room2D_AP as BaseCase
from pymooCFD.core.minimizeCFD import MinimizeCFD


def exec_test(**kwargs):
    study = MinimizeCFD(BaseCase)
    test_run_dir = 'test_run'
    if test_run_dir in study.opt_runs:
        opt_run = study.opt_runs[test_run_dir]
    else:
        xl = [0.5, 0.5, 1, 0.5]
        xu = [3.5, 3.5, 4, 6]
        alg = study.get_algorithm(**kwargs)
        prob = study.get_problem(xl, xu, **kwargs)
        opt_run = study.new_run(alg, prob, run_dir=test_run_dir, **kwargs)
    opt_run.test_case.run()
    # opt_run.run_bnd_cases()
    # opt_run.run()


def exec_study(**kwargs):
    study = MinimizeCFD(BaseCase)
    run_dir = 'default_run'
    if run_dir in study.opt_runs:
        opt_run = study.opt_runs[run_dir]
    else:
        xl = [0.5, 0.5, 1, 0.5]
        xu = [3.5, 3.5, 4, 6]
        alg = study.get_algorithm(n_gen=20, pop_size=50, n_offsprings=7, **kwargs)
        prob = study.get_problem(xl, xu, **kwargs)
        opt_run = study.new_run(alg, prob, run_dir=run_dir, **kwargs)
    # opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    opt_run.run()
