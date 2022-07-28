from pymooCFD.studies.oscill_cyl_SLURM_opt import exec_test, exec_study, BaseCase, MinimizeCFD
from pymooCFD.core.meshStudy import MeshStudy
from pymoo.factory import get_termination


def main():
    # exec_test()
    # exec_study()
    study = MinimizeCFD(BaseCase)

    # study.run_case('2.0_0.4', [2.0, 0.4])
    study.test_case.mesh_study.run()

    # xl = [0.2, 0.1]  # lower limits of parameters/variables
    # xu = [8, 1]  # upper limits of variables
    # prob = study.get_problem(xl, xu)
    # alg = study.get_algorithm(n_gen=25, pop_size=50,
    # n_offsprings=15)
    # opt_run = study.new_run(alg, prob)
    # opt_run = study.opt_runs['default_run']
    
    # opt_run.gen_bnd_cases(n_pts=3, getDiags=True)
    # ms = MeshStudy(opt_run.test_case,size_factors=[1.0, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0])
    # opt_run.test_case.mesh_study = ms
    # for case in opt_run.bnd_cases:
    #     ms = MeshStudy(case, size_factors=[1.0, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0])
    #     case.mesh_study = ms
    # print(opt_run.test_case.mesh_study.__dict__)
    # print(opt_run.bnd_cases[0].mesh_study.__dict__)
    # opt_run.test_case.mesh_study.run()
    # opt_run.run_bnd_cases(do_mesh_study=True)
    
    # opt_run.problem.BaseCase.procLim = 60
    # opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    
    # alg = opt_run.algorithm
    # alg.has_terminated = False
    # alg.termination = get_termination('n_gen', 50)
    
    # opt_run.run()


if __name__ == '__main__':
    main()
