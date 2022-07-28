from pymooCFD.studies.air_pur_opt import exec_test, exec_study
from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.air_purifier import Room2D_AP as BaseCase
from pymoo.factory import get_termination

def main():
    # exec_test()
    # exec_study()

    study = MinimizeCFD(BaseCase)
    
    #study.run_case('ap-mid-room_ACH-0.5', [2, 2, 1, 0.5])

    # xl = [0.5, 0.5, 1, 0.5]
    # xu = [3.5, 3.5, 4, 6]
    # alg = study.get_algorithm(n_gen=20, pop_size=50, n_offsprings=10)
    # prob = study.get_problem(xl, xu)
    # opt_run = study.new_run(alg, prob)
    
    opt_run = study.opt_runs['run05']

    opt_run.test_case.mesh_study.run()

    for case in opt_run.bnd_cases:
        case.mesh_study.run()
    
    # opt_run.algorithm.has_terminated = False
    # opt_run.algorithm.termination = get_termination('n_gen', 80)
    
    # opt_run.run_test_case()
    # opt_run.run_bnd_cases()
    # opt_run.run()


if __name__ == '__main__':
    main()
