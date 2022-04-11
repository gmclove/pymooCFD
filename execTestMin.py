from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.oscillCyl_x2 import OscillCylinder

def main():
    rans_const_study = MinimizeCFD(OscillCylinder)
    xl = [0.1, 1]
    xu = [2.0, 8]
    prob = rans_const_study.get_problem(xl, xu)
    alg = rans_const_study.get_algorithm()
    # opt_run = rans_const_study.new_run(alg, prob)
    # print(rans_const_study.get_logger())
    # print(rans_const_study.opt_runs)
    opt_run = rans_const_study.opt_runs[0]
    # print(opt_run)
    # print(opt_run.get_logger())

    opt_run.testCase.run()
    opt_run.runBndCases()
    opt_run.run()


if __name__ == '__main__':
    main()
