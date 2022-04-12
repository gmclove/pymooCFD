from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.oscill_cyl_x2 import OscillCylinder


def exec_test():
    study = MinimizeCFD(OscillCylinder)

    xl = [0.1, 0.1]  # lower limits of parameters/variables
    xu = [6, 1]  # upper limits of variables
    prob = study.get_problem(xl, xu)
    alg = study.get_algorithm(n_gen=2, pop_size=3,
                              n_offsprings=2)
    opt_run = study.new_run(alg, prob, run_dir='test_run')
    opt_run.test_case.run()
    opt_run.run_bnd_cases()
    opt_run.run()

def exec():
    # if study.opt_runs:
    #     opt_run = study.opt_runs[0]
    # else:
    xl = [0.1, 0.1]  # lower limits of parameters/variables
    xu = [6, 1]  # upper limits of variables
    prob = study.get_problem(xl, xu)
    alg = study.get_algorithm(n_gen=30, pop_size=50,
                              n_offsprings=20)
    opt_run = study.new_run(alg, prob)
    opt_run.test_case.run()
    opt_run.run_bnd_cases()
    opt_run.run()
