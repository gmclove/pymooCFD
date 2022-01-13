# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

from setupOpt import optStudy
import numpy as np


def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
    # optStudy.testCase.msCases[0].genMesh()

    ### Pre-Proccess ###
    # optStudy.preProc()
    # optStudy.runTestCase()
    # print(np.around(np.arange(0.2, 0.6, 0.05), decimals=2))
    # optStudy.testCase.meshSFs = np.around(
    #     np.arange(0.2, 0.6, 0.05), decimals=2)
    # optStudy.testCase.meshStudy(restart=False)
    optStudy.runBndCases(n_pts=1, getDiags=False, doMeshStudy=True)
    # # optStudy.runGen1()
    # optStudy.run(restart=False)
    # optStudy.meshStudy(optStudy.gen1Pop)

    ### Execute Study ###
    # optStudy.run(restart = False)

    ### Post-Process ###
    # optStudy.archive()


if __name__ == '__main__':
    main()
