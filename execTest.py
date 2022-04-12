# from pymooCFD.studies.oscillCyl_x2 import optStudy
from pymooCFD.studies.rans_const_opt import optStudy

def main():
    case = optStudy.loadCases.loadCase('/users/g/l/glove1/gitrepo/pymooCFD/optStudy-RANSConstOpt/run/gen1/ind3')
    case.postProc()
    optStudy.test_case.preProc()
    optStudy.test_case.solve()
    optStudy.test_case.postProc()
    # db = optStudy.problem.BaseCase.database
    # print(db.database)
    # print(db.database.location)
    # print(db.database.PyClass)
    # optStudy.test_case.externalSolver = True
    # optStudy.test_case.run()
    # optStudy.test_case.validated = True
    # case = optStudy.test_case
    # case.f = None
    # case.postProc()

    optStudy.test_case.mesh_study.run()
    optStudy.run()


if __name__=='__main__':
    main()
