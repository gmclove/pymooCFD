from pymooCFD.studies.ap_2_opt import exec_study, exec_test, BaseCase, MinimizeCFD


def main():
    # exec_test()
    # exec_study()
    study = MinimizeCFD(BaseCase)
    # run_dir = 'default_run'
    x_min = y_min = 0.3
    x_max = y_max = 3.7

    xl = [x_min, y_min, x_min, y_min, 1, 1, 0.5]
    xu = [x_max, y_max, x_max, y_max, 4, 4, 6]
    alg = study.get_algorithm(n_gen=20, pop_size=50, n_offsprings=7)
    prob = study.get_problem(xl, xu)
    opt_run = study.new_run(alg, prob)
    # opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    opt_run.run()


if __name__ == '__main__':
    main()
