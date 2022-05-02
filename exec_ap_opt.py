from pymooCFD.studies.air_pur_opt import exec_test, exec_study
from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.air_purifier import Room2D_AP as BaseCase

def main():
    # exec_test()
    # exec_study()

    study = MinimizeCFD(BaseCase)
    xl = [0.5, 0.5, 1, 0.5]
    xu = [3.5, 3.5, 4, 6]
    alg = study.get_algorithm(n_gen=20, pop_size=50, n_offsprings=7)
    prob = study.get_problem(xl, xu)
    opt_run = study.new_run(alg, prob)
    # opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    opt_run.run()

if __name__ == '__main__':
    main()
