from pymooCFD.problems.rans_jet import RANSJet
from pymooCFD.core.minimizeCFD import MinimizeCFD

def exec():
    study = MinimizeCFD(RANSJet)
    xl = [0.005, 0.1]  # lower limits of parameters/variables
    xu = [0.04, 0.4]  # upper limits of variables
    if not study.opt_runs:
        alg = study.get_algorithm()
        prob = study.get_problem(xl, xu)
        opt_run = study.new_run(alg, prob)
    else:
        opt_run = study.opt_runs[0]
    opt_run.test_case.run()
    opt_run.test_case.mesh_study.run()
    opt_run.run_bnd_cases()
    opt_run.run()
