from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.oscill_cyl import OscillCylinder as BaseCase



def main():
    study = MinimizeCFD(BaseCase)
    # path = os.path.join(study.abs_path, 'best_k-e_standard')
    # BaseCase(path, [0.09, 30_000]).run()
  #  xl = [0.2, 0.1]  # lower limits of parameters/variables
 #   xu = [8, 1]  # upper limits of variables
#    prob = study.get_problem(xl, xu)
    #alg = study.get_algorithm(n_gen=25, pop_size=50,
    #n_offsprings=15)
    #opt_run = study.new_run(alg, prob)
    opt_run = study.opt_runs[0]
    opt_run.problem.BaseCase.procLim = 60
    opt_run.run_test_case()
    opt_run.run_bnd_cases()
    opt_run.run()


if __name__ == '__main__':
    main()
