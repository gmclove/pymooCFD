from pymooCFD.studies.oscill_cyl_Re500_SLURM_opt import exec_test, exec_study, \
                                                        BaseCase, MinimizeCFD


def main():
    # exec_test()
    # exec_study()
    BaseCase.nTasks = 10
    study = MinimizeCFD(BaseCase)
    
    # xl = [0.2, 0.1]  # lower limits of parameters/variables
    # xu = [8, 1.5]  # upper limits of variables
    # prob = study.get_problem(xl, xu)
    # alg = study.get_algorithm(n_gen=35, pop_size=50,
    #                            n_offsprings=15)
    # opt_run = study.new_run(alg, prob)
    
    opt_run = study.opt_runs['run02']

    print(opt_run.algorithm.termination.__dict__)
    
    # opt_run.problem.BaseCase.procLim = 60
    # opt_run.run_test_case()
    # opt_run.gen_bnd_cases(n_pts=3, getDiags=True)
    # opt_run.run_bnd_cases(do_mesh_study=True)
    opt_run.run()


if __name__ == '__main__':
    main()
