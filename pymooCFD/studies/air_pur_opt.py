from pymooCFD.problems.air_purifier import Room2D_AP
from pymooCFD.core.minimizeCFD import MinimizeCFD


def exec_test():
    study = MinimizeCFD(Room2D_AP)
    xl = [0.5, 0.5, 1, 0.5]
    xu = [3.5, 3.5, 4, 6]
    alg = study.get_algorithm()
    prob = study.get_problem(xl, xu)
    opt_run = study.new_run(alg, prob)
    opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    # opt_run.run()


def exec_opt():
    study = MinimizeCFD(Room2D_AP)
    xl = [0.5, 0.5, 1, 0.5]
    xu = [3.5, 3.5, 4, 6]
    alg = study.get_algorithm()
    prob = study.get_problem(xl, xu)
    opt_run = study.new_run(alg, prob)
    # opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    opt_run.run()
