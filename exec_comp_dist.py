from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.comp_dist import CompDistYALES2_SLURM

def main():
    study = MinimizeCFD(CompDistYALES2_SLURM)
    study.run_case('short-test', [1, 20, 100])
    xl = [1, 1, 50]  # lower limits of parameters/variables
    xu = [10, 10, 1_000]  # upper limits of variables
    prob = study.get_problem(xl, xu)
    alg = study.get_algorithm(n_gen=25, pop_size=50,
                              n_offsprings=15)
    opt_run = study.new_run(alg, prob)
    opt_run.test_case.run()
    opt_run.run_bnd_cases()
    opt_run.run()


if __name__ == '__main__':
    main()
