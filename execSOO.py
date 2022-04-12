# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

# from setupOpt import optStudy
import numpy as np
#from pymooCFD.studies.oscillCyl import BaseCase
#from pymoo.factory import get_termination
import os
from pymooCFD.studies.oscillCyl_soo import optStudy

def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
    #print(optStudy.algorithm.__dict__)
    #print(optStudy.problem.__dict__)
    #print(optStudy.BaseCase.__dict__)
    #print(optStudy.__dict__)
    #noOscCase = optStudy.BaseCase('no_osc_case', [0, 0])
    #noOscCase.run()

    #optStudy.runDir = os.path.join(optStudy.optDatDir, 'run')
    #optStudy.saveCP()
    #optStudy.test_case.loadCP()
    #print(optStudy.test_case.__dict__)
    #optStudy.algorithm.termination = get_termination("n_gen", 25)
    #print(optStudy.algorithm.callback.gen)
    ### Pre-Proccess ###
    # optStudy.preProc()
    # optStudy.runtest_case()
    #optStudy.test_case.meshSFs = np.around(
    #                        np.arange(0.3, 1.6, 0.1), decimals=2)
    #optStudy.BaseCase = BaseCase
    #optStudy.saveCP()
    #optStudy.test_case.abs_path = os.path.join(optStudy.runDir, 'test_case')
    #optStudy.saveCP()

    # optStudy.test_case.mesh_study()
    #optStudy.genBndCases()
    #for case in optStudy.bndCases:
     #   case.meshSFs = np.around(
    #                         np.arange(0.3, 1.6, 0.1), decimals=2)
    #optStudy.saveCP()
    #optStudy.run_bnd_cases(n_pts=3, getDiags=True, do_mesh_study=True)
    # for case in optStudy.gen1Pop:
    #     case.logger = case.getLogger()
    # optStudy.runGen1()
    optStudy.run()
    # optStudy.mesh_study(optStudy.gen1Pop)

    ### Execute Study ###
    # optStudy.run(restart = False)

    ### Post-Process ###
    # optStudy.archive()


if __name__ == '__main__':
    main()
