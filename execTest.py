# from pymooCFD.studies.oscillCyl_x2 import optStudy
from pymooCFD.studies.rans_const_opt import optStudy

def main():
    optStudy.testCase.preProc()
    optStudy.testCase.solve()
    optStudy.testCase.postProc()
    # db = optStudy.problem.BaseCase.database
    # print(db.database)
    # print(db.database.location)
    # print(db.database.PyClass)
    # optStudy.testCase.externalSolver = True
    # optStudy.testCase.run()
    # optStudy.testCase.validated = True
    # case = optStudy.testCase
    # case.f = None
    # case.postProc()

    optStudy.testCase.meshStudy.run()
    optStudy.run()


if __name__=='__main__':
    main()
