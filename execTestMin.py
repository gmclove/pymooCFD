from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.oscillCyl_x2 import OscillCylinder

def main():
    rans_const_study = MinimizeCFD(OscillCylinder)
    xl = [0.09 * 0.9, 4_000]
    xu = [0.09 * 1.1, 30_000]
    prob = rans_const_study.get_problem(xl, xu)
    alg = rans_const_study.get_algorithm()
    rans_const_study.new_run(alg, prob)
    rans_const_study.opt_runs[0].testCase.run()


if __name__ == '__main__':
    main()
