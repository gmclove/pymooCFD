from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.comp_dist import CompDistSLURM_YALES2

def main():
    study = MinimizeCFD(CompDistSLURM_YALES2)
    study.run_case('short-test', [1, 20])
    xl = [0.2, 0.1]  # lower limits of parameters/variables
    xu = [8, 1]  # upper limits of variables
    prob = study.get_problem(xl, xu)
    alg = study.get_algorithm(n_gen=25, pop_size=50,
                              n_offsprings=15)
    opt_run = study.new_run(alg, prob)
    opt_run.test_case.run()
    opt_run.run_bnd_cases()
    opt_run.run()




if __name__ == '__main__':
    main()
