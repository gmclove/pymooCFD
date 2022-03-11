# @Author: glove
# @Date:   2021-12-14T16:02:45-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:54-05:00

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.core.problem import Problem
import os
import gmsh
import numpy as np
from glob import glob
# import scipy
# from scipy.integrate import quad

from pymooCFD.core.optStudy import OptStudy
from pymooCFD.core.cfdCase import YALES2Case  # CFDCase
# from pymooCFD.util.yales2Tools import getLatestXMF


class OscillCylinder(YALES2Case):
    baseCaseDir = 'base_cases/osc-cyl_base'
    ####### Define Design Space #########
    n_var = 2
    var_labels = ['Amplitude [radians/s]', 'Frequency [cycles/s]']
    varType = ["real", "real"]  # options: 'int' or 'real'
    xl = [0.1, 0.1]  # lower limits of parameters/variables
    xu = [3.5, 1]  # upper limits of variables

    ####### Define Objective Space ########
    obj_labels = ['Change in Coefficient of Drag [%]',
                  'Resistive Torque [N m]']
    n_obj = 2
    ####### Define Constraints ########
    n_constr = 0
    ##### Local Execution Command #####
    externalSolver = True
    onlyParallelizeSolve = True
    nProc = 10
    procLim = 40
    solverExecCmd = ['mpirun', '-n', str(nProc), '2D_cylinder']

    def __init__(self, caseDir, x, meshSF=1.0):  # , *args, **kwargs):
        super().__init__(caseDir, x,
                         meshFile='2D_cylinder.msh22',
                         datFile='FORCES_temporal.txt',
                         jobFile='jobslurm.sh',
                         inputFile='2D_cylinder.in',
                         meshSF=meshSF,
                         meshSFs=[1.0, 1.3, 1.5, 2, 3, 4, 5]
                         # meshSFs=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8],
                         # meshSFs=np.append(
                         #     np.around(
                         #         np.arange(0.3, 1.6, 0.1), decimals=2),
                         #     [0.25, 0.35, 0.45]),
                         # *args, **kwargs
                         )

        # fPath = os.path.join(self.caseDir, 'solver01_rank00.log')
        # if os.path.exists(fPath):
        #     with open(fPath, 'rb') as f:
        #         print(f)
        #         try:  # catch OSError in case of a one line file
        #             f.seek(-2, os.SEEK_END)
        #             while f.read(1) != b'\n':
        #                 f.seek(-2, os.SEEK_CUR)
        #         except OSError:
        #             f.seek(0)
        #         last_line = f.readline().decode()
        #     return bool('in destroy_mpi' in last_line)
        # dumpDir = os.path.join(self.caseDir, 'dump')
        # finalSolnPath = os.path.join(dumpDir, '2D_cyl.sol000400.xmf')
        # if os.path.isfile(finalSolnPath) and os.path.isfile(self.datPath):
        #     return True
#            with open(self.datPath) as f:
#                lines = f.readlines()
#                if lines[-1][:5] == ' 2000':
#                    return True
#        return False
    #
    # def _preProc_restart(self):
    #     self._preProc()
    #     # read input lines
    #     in_lines = self.inputLines
    #     in_lines = self.commentKeywordLines('RESTART', in_lines)
    #
    #     # in_lines = self.commentKeywordLines('GMSH', in_lines)
    #
    #     # kw = 'GMSH'
    #     # kw_lines = self.findKeywordLines(kw, in_lines)
    #     # for kw_line, kw_line_i in kw_lines:
    #     #     if  kw_line[-1] != '#':
    #     #         in_lines[kw_line_i] = '#' + kw_line
    #
    #     # kw = 'RESTART_TYPE = GMSH'
    #     # kw_lines = self.findKeywordLines(kw, in_lines, exact=True)
    #     # for kw_line, kw_line_i in kw_lines:
    #     #     in_lines[kw_line_i] = '#' + kw_line
    #     # kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
    #     # kw_lines = self.findKeywordLines(kw, in_lines, exact=True)
    #     # for kw_line, kw_line_i in kw_lines:
    #     #     in_lines[kw_line_i] = '#' + kw_line
    #     # kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
    #     # kw_lines = self.findKeywordLines(kw, in_lines, exact=True)
    #     # for kw_line, kw_line_i in kw_lines:
    #     #     in_lines[kw_line_i] = '#' + kw_line
    #
    #     # latestSolnFile = self.getLatestSoln()
    #     # latestMeshFile = self.getLatestMesh()
    #     # if latestMeshFile is None:
    #     #     self.logger.error('Latest HDF mesh file not found')
    #     #     return
    #     # if latestSolnFile is None:
    #     #     self.logger.error('Latest HDF solution file not found')
    #     #     return
    #
    #     latestXMF = self.getLatestXMF()
    #     kw_lines = self.findKeywordLines('XMF', in_lines)
    #     for line_i, line in kw_lines:
    #         if 'RESTART' in line:
    #             del in_lines[line_i]
    #     in_lines.append('RESTART_TYPE = XMF')
    #     path = os.path.join('dump', latestXMF)
    #     in_lines.append('RESTART_XMF_SOLUTION = ' + path)
    #
    #     # write input lines
    #     self.inputLines = in_lines

    def _preProc(self):
        self.genMesh()
        ### EXTRACT VAR ###
        # Extract parameters for each individual
        omega = self.x[0]
        freq = self.x[1]
        ### SIMULATION INPUT PARAMETERS ###
        # open and read YALES2 input file to array of strings for each line
        in_lines = self.inputLines
        # find line that must change using a keyword
        keyword = 'CYL_ROTATION_PROP'
        kw_lines = self.findKeywordLines(keyword, in_lines)
        for kw_line_i, kw_line in kw_lines:
            # create new string to replace line
            newLine = f'{kw_line[:kw_line.index("=")]}= {omega} {freq} \n'
            in_lines[kw_line_i] = newLine
        # REPEAT FOR EACH LINE THAT MUST BE CHANGED
        self.inputLines = in_lines

    def _postProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        A_omega = self.x[0]
        freq = self.x[1]
        ######## Compute Objectives ##########
        ### Objective 1: Drag on Cylinder ###
        U = 1
        rho = 1
        D = 1  # [m] cylinder diameter
        data = np.genfromtxt(self.datPath, skip_header=1)
        # collect data after 100 seconds of simulation time
        mask = np.where(data[:, 3] > 100)
        # Surface integrals of Cp and Cf
        # DRAG: x-direction integrals
        F_P1, = data[mask, 4]
        F_S1, = data[mask, 6]
        F_drag = np.mean(F_P1 - F_S1)
        C_drag = F_drag / ((1 / 2) * rho * U**2 * D**2)
        C_drag_noOsc = 1.363317903314267276
        prec_change = -((C_drag_noOsc - C_drag) / C_drag_noOsc) * 100

        ### Objective 2 ###
        # Objective 2: Power consumed by rotating cylinder
        res_torque = data[mask, 9]
        abs_mean_res_torque = np.mean(abs(res_torque))
        F_res = abs_mean_res_torque * D / 2
        # t = 1  # [sec]
        # D = 1  # [m] cylinder diameter
        # th = 0.1  # [m] thickness of cylinder wall
        # r_o = D / 2  # [m] outer radius
        # r_i = r_o - th  # [m] inner radius
        # d = 2700  # [kg/m^3] density of aluminum
        # L = 1  # [m] length of cylindrical tube
        # V = L * np.pi * (r_o**2 - r_i**2)  # [m^3] volume of cylinder
        # m = d * V  # [kg] mass of cylinder
        # # [kg m^2] moment of inertia of a hollow cylinder
        # I = 0.5 * m * (r_i**2 + r_o**2)
        # KE_consu = 0.5 * I * omega**2 * 4 * np.pi * freq * \
        #     quad(lambda t: abs(np.sin(2 * np.pi * freq * t)
        #                        * np.cos(2 * np.pi * freq * t)), 0, 1)[0]
        # A_star = (A_omega*D)/(2*U)
        # f_star = freq/0.17001699830023273
        # KE_over_I = (A_star**2/(8*np.pi*f_star))*(2*np.pi*f_star - 0.5*np.sin(4*np.pi*f_star))
        # print(A_star)
        # print(f_star)
        # print(KE_over_I)
        self.f = [prec_change, F_res]

    def _genMesh(self):
        projName = '2D_cylinder'
        cylD = 1
        cylR = cylD / 2
        cyl_cx, cyl_cy, cyl_cz = 0, 0, 0
        dom_dx, dom_dy, dom_dz = cylD * 60, cylD * 30, 1
        dom_ox, dom_oy, dom_oz = -dom_dx / 6 + cyl_cx, - \
            dom_dy / 2 + cyl_cy, 0 + cyl_cz  # -dom_dz/2
        meshSizeMin, meshSizeMax = 0.01 * self.meshSF, 0.4
        #################################
        #          Initialize           #
        #################################
        gmsh.initialize()
        # By default Gmsh will not print out any messages: in order to output messages
        # on the terminal, just set the "General.Terminal" option to 1:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add(projName)
        # gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)
        #################################
        #      YALES2 Requirements      #
        #################################
        # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
        gmsh.option.setNumber('Mesh.RecombineAll', 0)
        # Only save entities that are assigned to a physical group
        gmsh.option.setNumber('Mesh.SaveAll', 0)
        #################################
        #           Geometry            #
        #################################
        rect = gmsh.model.occ.addRectangle(
            dom_ox, dom_oy, dom_oz, dom_dx, dom_dy)
        # add circle to rectangular domain to represent cylinder
        cir = gmsh.model.occ.addCircle(0, 0, 0, cylR)
        # use 1-D circle to create curve loop entity
        cir_loop = gmsh.model.occ.addCurveLoop([cir])
        cir_plane = gmsh.model.occ.addPlaneSurface(
            [cir_loop])  # creates 2-D entity
        # cut circle out of a rectangle
        # print(cylLoop)
        # print(rect)
        domDimTags, domDimTagsMap = gmsh.model.occ.cut(
            [(2, rect)], [(2, cir_plane)])
        # divide domain into 4 regions
        p_top_cyl = gmsh.model.occ.addPoint(0, cyl_cy + cylR, 0)
        p_bot_cyl = gmsh.model.occ.addPoint(0, cyl_cy - cylR, 0)
        p_left_cyl = gmsh.model.occ.addPoint(cyl_cx - cylR, 0, 0)
        p_right_cyl = gmsh.model.occ.addPoint(cyl_cx + cylR, 0, 0)

        p_top_dom = gmsh.model.occ.addPoint(0, dom_oy + dom_dy, 0)
        p_bot_dom = gmsh.model.occ.addPoint(0, dom_oy, 0)
        p_left_dom = gmsh.model.occ.addPoint(dom_ox, 0, 0)
        p_right_dom = gmsh.model.occ.addPoint(dom_ox + dom_dx, 0, 0)

        l_top = gmsh.model.occ.addLine(p_top_cyl, p_top_dom)
        l_bot = gmsh.model.occ.addLine(p_bot_cyl, p_bot_dom)
        l_left = gmsh.model.occ.addLine(p_left_cyl, p_left_dom)
        l_right = gmsh.model.occ.addLine(p_right_cyl, p_right_dom)

        domDimTags, domDimTagsMap = gmsh.model.occ.fragment(
            domDimTags, [(1, l_top), (1, l_bot), (1, l_left), (1, l_right)])
        # We finish by synchronizing the data from OpenCASCADE CAD kernel with
        # the Gmsh model:
        gmsh.model.occ.synchronize()
        #################################
        #    Physical Group Naming      #
        #################################
        dim = 2
        grpTag = gmsh.model.addPhysicalGroup(dim, range(1, 4 + 1))
        gmsh.model.setPhysicalName(dim, grpTag, 'dom')

        dim = 1
        grpTag = gmsh.model.addPhysicalGroup(dim, [14, 20])
        gmsh.model.setPhysicalName(dim, grpTag, 'x0')
        grpTag = gmsh.model.addPhysicalGroup(dim, [25, 18])
        gmsh.model.setPhysicalName(dim, grpTag, 'x1')
        grpTag = gmsh.model.addPhysicalGroup(dim, [16, 19])
        gmsh.model.setPhysicalName(dim, grpTag, 'y0')
        grpTag = gmsh.model.addPhysicalGroup(dim, [21, 24])
        gmsh.model.setPhysicalName(dim, grpTag, 'y1')
        grpTag = gmsh.model.addPhysicalGroup(dim, [15, 17, 22, 23])
        gmsh.model.setPhysicalName(dim, grpTag, 'cyl')
        #################################
        #           MESHING             #
        #################################
        # TRANSFINITE CURVE
        bnds_right = gmsh.model.getParametrizationBounds(1, l_right)
        len_right = abs(bnds_right[1][0] - bnds_right[0][0])
        bnds_left = gmsh.model.getParametrizationBounds(1, l_left)
        len_left = abs(bnds_left[1][0] - bnds_left[0][0])
        bnds_top = gmsh.model.getParametrizationBounds(1, l_top)
        len_top = abs(bnds_top[1][0] - bnds_top[0][0])
        bnds_bot = gmsh.model.getParametrizationBounds(1, l_bot)
        len_bot = abs(bnds_bot[1][0] - bnds_bot[0][0])

        def get_coeff_and_NN(x_min, x_max, x_tot, NN_init=100, coef_init=1.001):
            max_it = 40
            it = 0
            thresh = 1e-6
            err = np.inf

            x_0 = x_min
            x_f = x_max
            NN = NN_init
            coef = coef_init
            while it < max_it and err > thresh:
                coef_prev = coef
                NN = int(np.log(1 + x_tot / x_0 * (coef - 1)) / np.log(coef) + 3)
                coef = np.e**(np.log(x_f / x_0) / (NN - 3))
                err = abs(coef_prev - coef)
                it += 1
                # print(it, err, coef, NN)
            return coef, NN

        x_min = meshSizeMin
        x_max = meshSizeMax
        coef_left, NN_left = get_coeff_and_NN(x_min, x_max, len_left)
        coef_right, NN_right = get_coeff_and_NN(x_min, x_max, len_right)
        coef_bot, NN_bot = get_coeff_and_NN(x_min, x_max, len_bot)
        coef_top, NN_top = get_coeff_and_NN(x_min, x_max, len_top)

        gmsh.model.mesh.setTransfiniteCurve(l_bot, NN_bot, coef=coef_bot)
        gmsh.model.mesh.setTransfiniteCurve(l_top, NN_top, coef=coef_top)
        gmsh.model.mesh.setTransfiniteCurve(l_left, NN_left, coef=coef_left)
        gmsh.model.mesh.setTransfiniteCurve(l_right, NN_right, coef=coef_right)

        len_quarter_cyl = 2 * np.pi * cylR / 4
        NN_quarter_cyl = int(len_quarter_cyl / x_min)
        cyl_tags = [15, 17, 22, 23]
        for tag in cyl_tags:
            gmsh.model.mesh.setTransfiniteCurve(tag, NN_quarter_cyl)
        # Set minimum and maximum mesh size
        # gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
        gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)

        # Set number of nodes along cylinder wall
        # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 200)
        # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)

        # Set size of mesh at every point in model
        # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)

        # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
        # We can then generate a 2D mesh...
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.generate(2)
        # extract elements
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
        # count number of elements
        self.numElem = sum(len(i) for i in elemTags)
        # print('Number of Elements:', numElem)
        ##################
        #    FINALIZE    #
        ##################
        # ... and save it to disk
        gmsh.write(self.meshPath)
        # To visualize the model we can run the graphical user interface with
        # `gmsh.fltk.run()'.
        # gmsh.fltk.run()
        # This should be called when you are done using the Gmsh Python API:
        gmsh.finalize()
        #######################################################################
        # projName = '2D_cylinder'
        # # dom_dx, dom_dy = 60, 25
        # centX, centY, centZ = 0, 0, 0
        # cylD = 1
        # domD = cylD * 20
        # meshSizeMax = 0.5
        # #################################
        # #          Initialize           #
        # #################################
        # gmsh.initialize()
        # # By default Gmsh will not print out any messages: in order to output messages
        # # on the terminal, just set the "General.Terminal" option to 1:
        # gmsh.option.setNumber("General.Terminal", 0)
        # gmsh.clear()
        # gmsh.model.add(projName)
        # gmsh.option.setNumber('Mesh.MeshSizeFactor', self.meshSF)
        # #################################
        # #      YALES2 Requirements      #
        # #################################
        # # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
        # gmsh.option.setNumber('Mesh.RecombineAll', 0)
        # # Only save entities that are assigned to a physical group
        # gmsh.option.setNumber('Mesh.SaveAll', 0)
        # #################################
        # #           Geometry            #
        # #################################
        # cyl = gmsh.model.occ.addCircle(centX, centY, centZ, cylD/2)
        # topPt = gmsh.model.occ.addPoint(centX, centY + domD, centZ)
        # botPt = gmsh.model.occ.addPoint(centX, centY - domD, centZ)
        # centPt = gmsh.model.occ.addPoint(centX, centY, centZ)
        # inlet = gmsh.model.occ.addCircleArc(botPt, centPt, topPt)
        # outlet = gmsh.model.occ.addCircleArc(topPt, centPt, botPt)
        # outerLoop = gmsh.model.occ.addCurveLoop([inlet, outlet])
        # innerLoop = gmsh.model.occ.addCurveLoop([cyl])
        # dom = gmsh.model.occ.addPlaneSurface([innerLoop, outerLoop])
        # # We finish by synchronizing the data from OpenCASCADE CAD kernel with
        # # the Gmsh model:
        # gmsh.model.occ.synchronize()
        # #################################
        # #    Physical Group Naming      #
        # #################################
        # dim = 2
        # grpTag = gmsh.model.addPhysicalGroup(dim, [dom])
        # gmsh.model.setPhysicalName(dim, grpTag, 'dom')
        # dim = 1
        # grpTag = gmsh.model.addPhysicalGroup(dim, [inlet])
        # gmsh.model.setPhysicalName(dim, grpTag, 'x0')
        # grpTag = gmsh.model.addPhysicalGroup(dim, [outlet])
        # gmsh.model.setPhysicalName(dim, grpTag, 'x1')
        # grpTag = gmsh.model.addPhysicalGroup(dim, [cyl])
        # gmsh.model.setPhysicalName(dim, grpTag, 'cyl')
        # #################################
        # #           MESHING             #
        # #################################
        # # We could also use a `Box' field to impose a step change in element
        # # sizes inside a box
        # # boxF = gmsh.model.mesh.field.add("Box")
        # # gmsh.model.mesh.field.setNumber(boxF, "VIn", meshSizeMax/10)
        # # gmsh.model.mesh.field.setNumber(boxF, "VOut", meshSizeMax)
        # # gmsh.model.mesh.field.setNumber(boxF, "XMin", cylD/3)
        # # gmsh.model.mesh.field.setNumber(boxF, "XMax", cylD/3+cylD*10)
        # # gmsh.model.mesh.field.setNumber(boxF, "YMin", -cylD)
        # # gmsh.model.mesh.field.setNumber(boxF, "YMax", cylD)
        # # # Finally, let's use the minimum of all the fields as the background mesh field:
        # # minF = gmsh.model.mesh.field.add("Min")
        # # gmsh.model.mesh.field.setNumbers(minF, "FieldsList", [boxF])
        # # gmsh.model.mesh.field.setAsBackgroundMesh(minF)
        #
        # # Set minimum and maximum mesh size
        # #gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
        # gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)
        #
        # # Set number of nodes along cylinder wall
        # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 200)
        # # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)
        #
        # # Set size of mesh at every point in model
        # # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)
        #
        # # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
        # # We can then generate a 2D mesh...
        # gmsh.model.mesh.generate(1)
        # gmsh.model.mesh.generate(2)
        # # extract number of elements
        # # get all elementary entities in the model
        # entities = gmsh.model.getEntities()
        # ##################
        # #    FINALIZE    #
        # ##################
        # e = entities[-1]
        # # get the mesh elements for each elementary entity
        # elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
        #     e[0], e[1])
        # # count number of elements
        # self.numElem = sum(len(i) for i in elemTags)
        # # ... and save it to disk
        # gmsh.write(self.meshPath)
        # # To visualize the model we can run the graphical user interface with
        # # `gmsh.fltk.run()'.
        # # gmsh.fltk.run()
        # # This should be called when you are done using the Gmsh Python API:
        # gmsh.finalize()

    # def _genMesh(self):
    #     projName = '2D_cylinder'
    #     dom_dx, dom_dy = 60, 25
    #     cylD = 1
    #     meshSizeMax = 0.5
    #     #################################
    #     #          Initialize           #
    #     #################################
    #     gmsh.initialize()
    #     # By default Gmsh will not print out any messages: in order to output messages
    #     # on the terminal, just set the "General.Terminal" option to 1:
    #     gmsh.option.setNumber("General.Terminal", 0)
    #     gmsh.clear()
    #     gmsh.model.add(projName)
    #     gmsh.option.setNumber('Mesh.MeshSizeFactor', self.meshSF)
    #     #################################
    #     #      YALES2 Requirements      #
    #     #################################
    #     # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
    #     gmsh.option.setNumber('Mesh.RecombineAll', 0)
    #     # Only save entities that are assigned to a physical group
    #     gmsh.option.setNumber('Mesh.SaveAll', 0)
    #     #################################
    #     #           Geometry            #
    #     #################################
    #     rect = gmsh.model.occ.addRectangle(0, 0, 0, dom_dx, dom_dy)
    #     # add circle to rectangular domain to represent cylinder
    #     cir = gmsh.model.occ.addCircle(dom_dx / 4, dom_dy / 2, 0, cylD)
    #     # use 1-D circle to create curve loop entity
    #     cir_loop = gmsh.model.occ.addCurveLoop([cir])
    #     cir_plane = gmsh.model.occ.addPlaneSurface(
    #         [cir_loop])  # creates 2-D entity
    #     # cut circle out of a rectangle
    #     domDimTags, domDimTagsMap = gmsh.model.occ.cut(
    #         [(2, rect)], [(2, cir_plane)])
    #     # dom = domDimTags
    #     # We finish by synchronizing the data from OpenCASCADE CAD kernel with
    #     # the Gmsh model:
    #     gmsh.model.occ.synchronize()
    #     #################################
    #     #    Physical Group Naming      #
    #     #################################
    #     dim = 2
    #     grpTag = gmsh.model.addPhysicalGroup(dim, [1])
    #     gmsh.model.setPhysicalName(dim, grpTag, 'dom')
    #     dim = 1
    #     grpTag = gmsh.model.addPhysicalGroup(dim, [7])
    #     gmsh.model.setPhysicalName(dim, grpTag, 'x0')
    #     grpTag = gmsh.model.addPhysicalGroup(dim, [8])
    #     gmsh.model.setPhysicalName(dim, grpTag, 'x1')
    #     grpTag = gmsh.model.addPhysicalGroup(dim, [6])
    #     gmsh.model.setPhysicalName(dim, grpTag, 'y0')
    #     grpTag = gmsh.model.addPhysicalGroup(dim, [9])
    #     gmsh.model.setPhysicalName(dim, grpTag, 'y1')
    #     grpTag = gmsh.model.addPhysicalGroup(dim, [5])
    #     gmsh.model.setPhysicalName(dim, grpTag, 'cyl')
    #     #################################
    #     #           MESHING             #
    #     #################################
    #     # We could also use a `Box' field to impose a step change in element
    #     # sizes inside a box
    #     # boxF = gmsh.model.mesh.field.add("Box")
    #     # gmsh.model.mesh.field.setNumber(boxF, "VIn", meshSizeMax/10)
    #     # gmsh.model.mesh.field.setNumber(boxF, "VOut", meshSizeMax)
    #     # gmsh.model.mesh.field.setNumber(boxF, "XMin", cylD/3)
    #     # gmsh.model.mesh.field.setNumber(boxF, "XMax", cylD/3+cylD*10)
    #     # gmsh.model.mesh.field.setNumber(boxF, "YMin", -cylD)
    #     # gmsh.model.mesh.field.setNumber(boxF, "YMax", cylD)
    #     # # Finally, let's use the minimum of all the fields as the background mesh field:
    #     # minF = gmsh.model.mesh.field.add("Min")
    #     # gmsh.model.mesh.field.setNumbers(minF, "FieldsList", [boxF])
    #     # gmsh.model.mesh.field.setAsBackgroundMesh(minF)
    #
    #     # Set minimum and maximum mesh size
    #     #gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
    #     gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)
    #
    #     # Set number of nodes along cylinder wall
    #     gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 200)
    #     # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)
    #
    #     # Set size of mesh at every point in model
    #     # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)


class OscillCylinderOpt(OptStudy):
    def __init__(self, algorithm, problem, BaseCase,
                 # *args, **kwargs
                 ):
        super().__init__(algorithm, problem, BaseCase,
                         optName='OscCylX2',
                         runDir='run'
                         # n_opt = 20,
                         # baseCaseDir='base_cases/osc-cyl_base',
                         # optDatDir='cyl-opt_run',
                         # *args, **kwargs
                         )


MyOptStudy = OscillCylinderOpt
BaseCase = OscillCylinder

#####################################
#### Genetic Algorithm Criteria #####
#####################################
n_gen = 25
pop_size = 50
n_offsprings = int(pop_size * (2 / 3))  # = num. of evaluations each generation

#################
#    PROBLEM    #
#################
# from pymooCFD.core.pymooBase import GA_CFD
# from pymoo.core.problem import ElementwiseProblem


class GA_CFD(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(n_var=BaseCase.n_var,
                         n_obj=BaseCase.n_obj,
                         n_constr=BaseCase.n_constr,
                         xl=np.array(BaseCase.xl),
                         xu=np.array(BaseCase.xu),
                         *args,
                         **kwargs
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        out = optStudy.runGen(X, out)
        # out['F'] = np.zeros((BaseCase.n_obj, pop_size))


problem = GA_CFD()


#################
#    DISPLAY    #
#################


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(problem.n_obj):
            self.output.append(
                f'mean obj.{obj + 1}', np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append(
                f"best obj.{obj+1}", algorithm.pop.get('F')[:, obj].min())
        self.output.header()


display = MyDisplay()


##################
#    CALLBACK    #
##################


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.gen = 1
        self.data['best'] = []

    def notify(self, alg):
        # save checkpoint
        optStudy.saveCP()
        # increment generation
        self.gen += 1
        self.data["best"].append(alg.pop.get("F").min())
        # For longer runs to save memory may want to use callback.data
        # instead of using algorithm.save_history=True which stores deep
        # copy of algorithm object every generation.
        ## Example: self.data['var'].append(alg.pop.get('X'))


callback = MyCallback()

###################
#    OPERATORS    #
###################

sampling = MixedVariableSampling(BaseCase.varType, {
    "real": get_sampling("real_lhs"),  # "real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(BaseCase.varType, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(BaseCase.varType, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})


###############################
#    TERMINATION CRITERION    #
###############################
# https://pymoo.org/interface/termination.html
termination = get_termination("n_gen", n_gen)

# from pymoo.util.termination.default import MultiObjectiveDefaultTermination
# termination = MultiObjectiveDefaultTermination(
#     x_tol=1e-8,
#     cv_tol=1e-6,
#     f_tol=0.0025,
#     nth_gen=5,
#     n_last=30,
#     n_max_gen=1000,
#     n_max_evals=100000
# )

###################
#    ALGORITHM    #
###################
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# initialize algorithm here
# will be overwritten in runOpt() if checkpoint already exists
algorithm = NSGA2(pop_size=pop_size,
                  n_offsprings=n_offsprings,
                  eliminate_duplicates=False,

                  termination=termination,

                  sampling=sampling,
                  crossover=crossover,
                  mutation=mutation,

                  display=display,
                  callback=callback,
                  )
# setup run specific criteria
algorithm.save_history = True
algorithm.seed = 1
algorithm.return_least_infeasible = True
algorithm.verbose = True

################################################################################
########  Optimization Study Object Initialization ##########
optStudy = MyOptStudy(algorithm, problem, BaseCase)
