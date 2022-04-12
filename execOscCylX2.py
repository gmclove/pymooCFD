# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

from pymooCFD.studies.oscillCyl_x2 import optRun, algorithm, problem, BaseCase
import numpy as np
# from pymooCFD.studies.oscillCyl import BaseCase
# from pymoo.factory import get_termination
import os
# from pymooCFD.studies.oscillCyl_soo import optRun
import types


def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
    # print(optRun.algorithm.__dict__)
    # print(optRun.problem.__dict__)
    # print(optRun.BaseCase.__dict__)
    # print(optRun.__dict__)

    # optRun.BaseCase = BaseCase
    # optRun.algorithm = algorithm
    # optRun.problem = problem
    # optRun.newAlg()
    # noOscCase = optRun.BaseCase('no_osc_case', [0, 0])
    # noOscCase.genMesh()
    # noOscCase.f = None
    # noOscCase.postProc()
    # noOscCase.run()
    # noOscCase.msCases = None
    # noOscCase.mesh_study()

    # optRun.runDir = os.path.join(optRun.optDatDir, 'run')
    # optRun.saveCP()
    # optRun.test_case.loadCP()
    # print(optRun.test_case.__dict__)
    # optRun.algorithm.termination = get_termination("n_gen", 25)
    # print(optRun.algorithm.callback.gen)
    ### Pre-Proccess ###
    # optRun.preProc()
    # optRun.runtest_case()

    # meshSFs = [0.6, 0.7, 0.9, 1.0, 1.3, 1.5, 2, 3, 4, 5]
    # meshSFs = [1.0, 1.3, 1.5, 2, 3, 4, 5]
    # noOscCase = optRun.BaseCase('no_osc_case', [0, 0])
    # # noOscCase.nProc = 30
    # # noOscCase.run()
    # # noOscCase.nProc = 10
    # noOscCase.meshSFs = meshSFs
    # noOscCase.mesh_study()
    #
    # optRun.test_case.meshSFs = meshSFs
    # optRun.test_case.mesh_study()
    # optRun.run_bnd_cases(n_pts=3, getDiags=True, do_mesh_study=True)
    #
    # optRun.genBndCases()
    #
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
    #
    # optRun.test_case.obj_labels = [
    #     'Coefficient of Drag', 'Resistive Torque [N m]']
    # optRun.test_case._postProc = types.MethodType(
    #     _postProc, optRun.test_case)
    # for case in optRun.test_case.msCases:
    #     # print(case)
    #     # print(case.msCases)
    #     # case.meshSFs = meshSFs
    #     # case.mesh_study()
    #     # optRun.saveCP()
    #     # print(case.msCases)
    #     # for msCase in case.msCases:
    #     case._postProc = types.MethodType(_postProc, case)
    #     # msCase._postProc = _postProc
    #
    #     case.f = None
    #     case.postProc()
    #     case.obj_labels = ['Coefficient of Drag', 'Resistive Torque [N m]']
    #optRun.test_case.plotmesh_study()
    #optRun.plotBndPts()
    #optRun.plotBndPtsObj()    
#
    # for case in optRun.bndCases:
    #     case.f = None
    # optRun.BaseCase.parallelize(optRun.bndCases)
    # optRun.run_bnd_cases(n_pts=3, getDiags=True, do_mesh_study=True)
 #   for case in optRun.loadCases(os.path.join(optRun.runDir, 'gen21')):
#         case.postProc()
    # optRun.runGen1()
    optRun.run()
    # optRun.mesh_study(optRun.gen1Pop)

    ### Execute Study ###
    # optRun.run(restart = False)

    ### Post-Process ###
    # optRun.archive()


if __name__ == '__main__':
    main()
