from pymooCFD.problems.rans_jet import RANSJet
from pymooCFD.core.minimizeCFD import MinimizeCFD

def main():
    study = MinimizeCFD(RANSJet)
    xl = [0.005, 0.1]  # lower limits of parameters/variables
    xu = [0.04, 0.4]  # upper limits of variables
    opt_run = study.new_run(xl, xu)
    opt_run.test_case.run()
    opt_run.test_case.mesh_study.run()
    opt_run.run_bnd_cases()
    opt_run.run()


if __name__=='__main__':
    main()
