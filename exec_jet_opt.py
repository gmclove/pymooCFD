from pymooCFD.studies.rans_jet_opt import exec_test, exec_study, BaseCase, MinimizeCFD

def main():
    #exec_test()
    #exec_study()
    study = MinimizeCFD(BaseCase)
    xl = [0.005, 0.1]  # lower limits of parameters/variables
    xu = [0.04, 0.4]  # upper limits of variables
    alg = study.get_algorithm(n_gen=20, pop_size=30, n_offsprings=8)
    prob = study.get_problem(xl, xu)
    opt_run = study.new_run(alg, prob)
    # opt_run = study.opt_runs[0]
    # opt_run.test_case.run()
    # opt_run.test_case.mesh_study.run()
    # opt_run.run_bnd_cases()
    opt_run.run()                     


if __name__ == '__main__':
    main()
