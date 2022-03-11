# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

from pymooCFD.studies.oscillCyl_x2 import optStudy, algorithm, problem, BaseCase
import numpy as np
# from pymooCFD.studies.oscillCyl import BaseCase
# from pymoo.factory import get_termination
import os
# from pymooCFD.studies.oscillCyl_soo import optStudy
import types


def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
    # print(optStudy.algorithm.__dict__)
    # print(optStudy.problem.__dict__)
    # print(optStudy.BaseCase.__dict__)
    # print(optStudy.__dict__)

    # optStudy.BaseCase = BaseCase
    # optStudy.algorithm = algorithm
    # optStudy.problem = problem
    # optStudy.newAlg()
    # noOscCase = optStudy.BaseCase('no_osc_case', [0, 0])
    # noOscCase.genMesh()
    # noOscCase.f = None
    # noOscCase.postProc()
    # noOscCase.run()
    # noOscCase.msCases = None
    # noOscCase.meshStudy()

    # optStudy.runDir = os.path.join(optStudy.optDatDir, 'run')
    # optStudy.saveCP()
    # optStudy.testCase.loadCP()
    # print(optStudy.testCase.__dict__)
    # optStudy.algorithm.termination = get_termination("n_gen", 25)
    # print(optStudy.algorithm.callback.gen)
    ### Pre-Proccess ###
    # optStudy.preProc()
    # optStudy.runTestCase()

    # meshSFs = [0.6, 0.7, 0.9, 1.0, 1.3, 1.5, 2, 3, 4, 5]
    # meshSFs = [1.0, 1.3, 1.5, 2, 3, 4, 5]
    # noOscCase = optStudy.BaseCase('no_osc_case', [0, 0])
    # # noOscCase.nProc = 30
    # # noOscCase.run()
    # # noOscCase.nProc = 10
    # noOscCase.meshSFs = meshSFs
    # noOscCase.meshStudy()
    #
    # optStudy.testCase.meshSFs = meshSFs
    # optStudy.testCase.meshStudy()
    #
    # optStudy.genBndCases()

    # def _postProc(self):
    #     ####### EXTRACT VAR ########
    #     # Extract parameters for each individual
    #     A_omega = self.x[0]
    #     freq = self.x[1]
    #     ######## Compute Objectives ##########
    #     ### Objective 1: Drag on Cylinder ###
    #     U = 1
    #     rho = 1
    #     D = 1  # [m] cylinder diameter
    #     data = np.genfromtxt(self.datPath, skip_header=1)
    #     # collect data after 100 seconds of simulation time
    #     mask = np.where(data[:, 3] > 100)
    #     # Surface integrals of Cp and Cf
    #     # DRAG: x-direction integrals
    #     F_P1, = data[mask, 4]
    #     F_S1, = data[mask, 6]
    #     F_drag = np.mean(F_P1 - F_S1)
    #     C_drag = F_drag / ((1 / 2) * rho * U**2 * D**2)
    #     # C_drag_noOsc = 1.363317903314267276
    #     # prec_change = -((C_drag_noOsc - C_drag) / C_drag_noOsc) * 100
    #
    #     ### Objective 2 ###
    #     # Objective 2: Power consumed by rotating cylinder
    #     res_torque = data[mask, 9]
    #     abs_mean_res_torque = np.mean(abs(res_torque))
    #     F_res = abs_mean_res_torque * D / 2
    #     self.f = [C_drag, F_res]

    # for case in optStudy.bndCases:
    #     print(case)
    #     print(case.msCases)
    #     case.meshSFs = meshSFs
    #     case.meshStudy()
    #     # optStudy.saveCP()
    #     # print(case.msCases)
    #     for msCase in case.msCases:
    #         msCase._postProc = types.MethodType(_postProc, msCase)
    #         # msCase._postProc = _postProc
    #
    #         msCase.f = None
    #         msCase.postProc()
    #     case.obj_labels = ['Coefficient of Drag', 'Resistive Torque [N m]']
    #     case.plotMeshStudy()
    #
    # for case in optStudy.bndCases:
    #     case.f = None
    # optStudy.BaseCase.parallelize(optStudy.bndCases)
    # optStudy.runBndCases(n_pts=3, getDiags=True, doMeshStudy=True)
    # for case in optStudy.gen1Pop:
    #     case.logger = case.getLogger()
    # optStudy.runGen1()
    optStudy.run()
    # optStudy.meshStudy(optStudy.gen1Pop)

    ### Execute Study ###
    # optStudy.run(restart = False)

    ### Post-Process ###
    # optStudy.archive()


if __name__ == '__main__':
    main()
