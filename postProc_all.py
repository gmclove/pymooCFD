import os

from pymooCFD.core.minimizeCFD import MinimizeCFD
from pymooCFD.problems.rans_k_eps import RANS_k_eps as BaseCase

def main():
    study = MinimizeCFD(BaseCase)
    print(study.opt_runs)
    opt_run = study.opt_runs[0]
    opt_run.test_case.postProc()
    postProc_bnd_cases()
    postProc_gen1(opt_run)

def postProc_cases_in_directory(opt_run, path):
    cases = opt_run.loadCases(path)
    print('Number of Cases:', len(cases))
    postProc_cases(cases)

def postProc_cases(cases):
    for case in cases:
        case.postProc()
        print('Previous:', case.x, '->', case.f)
        print('New:', case.x, '->', case.f)

def postProc_bnd_cases(opt_run):
    cases = opt_run.bnd_cases
    print('Number of Boundary Cases:', len(cases))
    postProc_cases(cases)

def postProc_gen1(opt_run):
    print('Generation 1')
    path = os.path.join(opt_run.abs_path, 'gen1')
    postProc_cases_in_directory(opt_run, path)


if __name__ == '__main__':
    main()
