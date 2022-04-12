# @Author: glove
# @Date:   2021-12-10T10:34:04-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:30-05:00

from pymooCFD.studies.oscillCyl_x3 import optRun, BaseCase
import numpy as np
#from pymooCFD.studies.oscillCyl import BaseCase
#from pymoo.factory import get_termination
import os
import types
#from pymooCFD.studies.oscillCyl_soo import optRun


def main():
    '''
    Execute Optimization Study

    import your optimization study object/instance from setupOpt.py and execute
    '''
    # print(optRun.algorithm.__dict__)
    # print(optRun.problem.__dict__)
    # print(optRun.BaseCase.__dict__)
    # print(optRun.__dict__)
#     optRun.BaseCase = BaseCase
#     # , meshFile='2D_cylinder.msh22')
#     noOscCase_Re100 = optRun.BaseCase('no_osc_x3_Re100_case', [0, 0, 1])
#     # , meshFile='2D_cylinder.msh22')
#     noOscCase_Re200 = optRun.BaseCase('no_osc_x3_Re200_case', [0, 0, 2])
#     # , meshFile='2D_cylinder.msh22')
#     noOscCase_Re300 = optRun.BaseCase('no_osc_x3_Re300_case', [0, 0, 3])
#     # , meshFile='2D_cylinder.msh22')
#     noOscCase_Re400 = optRun.BaseCase('no_osc_x3_Re400_case', [0, 0, 4])
#     # , meshFile='2D_cylinder.msh22')
#     noOscCase_Re500 = optRun.BaseCase('no_osc_x3_Re500_case', [0, 0, 5])
# #    del noOscCase_Re200.msCases[0]
#     cases = [noOscCase_Re100, noOscCase_Re200,
#              noOscCase_Re300, noOscCase_Re400, noOscCase_Re500]
#
#     def _postProc(self):
#         ####### EXTRACT VAR ########
#         # Extract parameters for each individual
#         A_omega = self.x[0]
#         freq = self.x[1]
#         ######## Compute Objectives ##########
#         ### Objective 1: Drag on Cylinder ###
#         U = 1
#         rho = 1
#         D = 1  # [m] cylinder diameter
#         data = np.genfromtxt(self.datPath, skip_header=1)
#         # collect data after 100 seconds of simulation time
#         mask = np.where(data[:, 3] > 100)
#         # Surface integrals of Cp and Cf
#         # DRAG: x-direction integrals
#         F_P1, = data[mask, 4]
#         F_S1, = data[mask, 6]
#         F_drag = np.mean(F_P1 - F_S1)
#         C_drag = F_drag / ((1 / 2) * rho * U**2 * D**2)
#         # C_drag_noOsc = 1.363317903314267276
#         # prec_change = -((C_drag_noOsc - C_drag) / C_drag_noOsc) * 100
#
#         ### Objective 2 ###
#         # Objective 2: Power consumed by rotating cylinder
#         res_torque = data[mask, 9]
#         abs_mean_res_torque = np.mean(abs(res_torque))
#         F_res = abs_mean_res_torque * D / 2
#         self.f = [C_drag, F_res]
#
#     for case in cases:
#         case.meshSFs = [1.0, 1.3, 1.5, 2, 3, 4, 5]
#         case._postProc = types.MethodType(_postProc, case)
#         case.obj_labels = ['Coefficient of Drag', 'Resistive Force [N]']
#         case.f = None
#         case.postProc()
#         case.genmesh_study()
#         for msCase in case.msCases:
#             msCase._postProc = types.MethodType(_postProc, msCase)
#             msCase.obj_labels = [
#                 'Coefficient of Drag', 'Resistive Force [N]']
#             # msCase.f = None
#             # msCase.postProc()
#         case.mesh_study()
#         # case.plotmesh_study()
#         # case.saveCP()

    # optRun.BaseCase.parallelize(cases)

    # for case in cases:
    # case.mesh_study()
    # case.meshFile = '2D_cylinder.msh22'
    # case.meshPath = os.path.join(case.abs_path, case.meshFile)
    # case.saveCP()
    # for msCase in case.msCases:
    #   print(msCase.f)
    #  path = os.path.join(msCase.abs_path, 'obj.txt')
    # msCase.f = np.loadtxt(path)
    # case.genmesh_study()
    # case.mesh_study()
    # exit()
    # noOscCase.genMesh()
    # noOscCase.f = None
    # noOscCase.postProc()
    # noOscCase.run()
    # noOscCase.msCases = None
    # noOscCase.mesh_study()

    #optRun.runDir = os.path.join(optRun.optDatDir, 'run')
    # optRun.saveCP()
    # optRun.test_case.loadCP()
    # print(optRun.test_case.__dict__)
    #optRun.algorithm.termination = get_termination("n_gen", 25)
    # print(optRun.algorithm.callback.gen)
    ### Pre-Proccess ###
    # optRun.preProc()
#    optRun.runtest_case()
    # optRun.test_case.meshSFs = np.around(
    #                        np.arange(0.3, 1.6, 0.1), decimals=2)
    #optRun.BaseCase = BaseCase
    # optRun.saveCP()
    #optRun.test_case.abs_path = os.path.join(optRun.runDir, 'test_case')
    # optRun.saveCP()

#    optRun.test_case.mesh_study()
    # optRun.genBndCases()
    # for case in optRun.bndCases:
    #   case.meshSFs = np.around(
    #                         np.arange(0.3, 1.6, 0.1), decimals=2)
    # optRun.saveCP()
    #optRun.run_bnd_cases(n_pts=3, getDiags=True, do_mesh_study=True)
    # for case in optRun.gen1Pop:
    #     case.logger = case.getLogger()
    # optRun.runGen1()
    optRun.run()
    # optRun.mesh_study(optRun.gen1Pop)

    ### Execute Study ###
    # optRun.run(restart = False)

    ### Post-Process ###
    # optRun.archive()


if __name__ == '__main__':
    main()
