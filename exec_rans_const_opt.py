from pymooCFD.problems.rans_k_eps import RANS_k_eps
from pymooCFD.core.minimizeCFD import MinimizeCFD
import os

def main():
    study = MinimizeCFD(RANS_k_eps)
    path = os.path.join(study.abs_path, 'best_k-e_standard')
    RANS_k_eps(path, [0.09, 30_000])
    xl = [0.09*0.8, 4_000]
    xu = [0.09*1.2, 30_000]
    prob = study.get_problem(xl, xu)
    alg = study.get_algorithm(n_gen=20, pop_size=50, n_offsprings=10)
    opt_run = study.new_run(alg, prob)
    # print(study.get_logger())
    # print(study.opt_runs)
    # print(study.__dict__)
    opt_runs = study.opt_runs
    print(opt_runs)
    print(len(opt_runs))
    # opt_run = opt_runs[0]
    print(opt_run)
    opt_run.test_case.run()
    opt_run.run_bnd_cases()
    opt_run.run()
