# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

from setupOpt import optStudy


def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
    ### Pre-Proccess ###
    # optStudy.preProc()
    optStudy.runTestCase()
    optStudy.testCase.meshStudy()
    optStudy.runBndCases(1, getDiags=False, doMeshStudy=True)
    # optStudy.runGen1()
    optStudy.run(restart=False)
    # optStudy.meshStudy(optStudy.gen1Pop)

    ### Execute Study ###
    # optStudy.run(restart = False)

    ### Post-Process ###
    # optStudy.archive()


if __name__ == '__main__':
    main()
