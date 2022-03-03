# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

<<<<<<< HEAD
from pymooCFD.studies.oscillCyl_x2 import optStudy
=======
#from setupOpt import optStudy
>>>>>>> a6d66a2165c828f45319bd0b44b55e542c6f9ad3
import numpy as np
#from pymooCFD.studies.oscillCyl import BaseCase
#from pymoo.factory import get_termination
import os
<<<<<<< HEAD
#from pymooCFD.studies.oscillCyl_soo import optStudy

=======
from pymooCFD.studies.oscillCyl_soo import optStudy
>>>>>>> a6d66a2165c828f45319bd0b44b55e542c6f9ad3

def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
<<<<<<< HEAD
    # print(optStudy.algorithm.__dict__)
    # print(optStudy.problem.__dict__)
    # print(optStudy.BaseCase.__dict__)
    # print(optStudy.__dict__)

    noOscCase = optStudy.BaseCase('no_osc_case', [0, 0])
    # noOscCase.genMesh()
    # noOscCase.f = None
    # noOscCase.postProc()
    noOscCase.run()
    # noOscCase.msCases = None
    # noOscCase.meshStudy()
=======
    print(optStudy.algorithm.__dict__)
    print(optStudy.problem.__dict__)
    print(optStudy.BaseCase.__dict__)
    print(optStudy.__dict__)
    optStudy.run()
    #noOscCase = optStudy.BaseCase('no_osc_case', [0, 0])
    #noOscCase.run()
>>>>>>> a6d66a2165c828f45319bd0b44b55e542c6f9ad3

    #optStudy.runDir = os.path.join(optStudy.optDatDir, 'run')
    # optStudy.saveCP()
    # optStudy.testCase.loadCP()
    # print(optStudy.testCase.__dict__)
    #optStudy.algorithm.termination = get_termination("n_gen", 25)
    # print(optStudy.algorithm.callback.gen)
    ### Pre-Proccess ###
    # optStudy.preProc()
    # optStudy.runTestCase()
    # optStudy.testCase.meshSFs = np.around(
    #                        np.arange(0.3, 1.6, 0.1), decimals=2)
    #optStudy.BaseCase = BaseCase
    # optStudy.saveCP()
    #optStudy.testCase.caseDir = os.path.join(optStudy.runDir, 'test_case')
    # optStudy.saveCP()

    # optStudy.testCase.meshStudy()
    # optStudy.genBndCases()
    # for case in optStudy.bndCases:
    #   case.meshSFs = np.around(
    #                         np.arange(0.3, 1.6, 0.1), decimals=2)
    # optStudy.saveCP()
    #optStudy.runBndCases(n_pts=3, getDiags=True, doMeshStudy=True)
    # for case in optStudy.gen1Pop:
    #     case.logger = case.getLogger()
    # optStudy.runGen1()
    # optStudy.run()
    # optStudy.meshStudy(optStudy.gen1Pop)

    ### Execute Study ###
    # optStudy.run(restart = False)

    ### Post-Process ###
    # optStudy.archive()


if __name__ == '__main__':
    main()
