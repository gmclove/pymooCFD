# @Author: glove
# @Date:   2021-12-14T16:02:45-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:54-05:00

from pymooCFD.core.optStudy import OptStudy
import os
import gmsh
import numpy as np
# import scipy
from scipy.integrate import quad

from pymooCFD.util.yales2Tools import getLatestXMF
from pymooCFD.core.cfdCase import CFDCase


class OscillCylinder(CFDCase):
    ####### Define Design Space #########
    n_var = 2
    var_labels = ['Amplitude [1/s]', 'Frequency [1/s]']
    varType = ["real", "real"]  # options: 'int' or 'real'
    xl = [0.1, 0.1]  # lower limits of parameters/variables
    xu = [3.0, 1]  # upper limits of variables
    if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
        raise Exception("Design Space Definition Incorrect")
    ####### Define Objective Space ########
    obj_labels = ['Drag on Cylinder [N]', 'Energy Consumption [J/s]']
    n_obj = 2
    ####### Define Constraints ########
    n_constr = 0
    ##### Local Execution Command #####
    externalSolver = True
    nProc = 10
    procLim = 60
    solverExecCmd = ['mpirun', '-n', str(nProc), '2D_cylinder']

    def __init__(self, baseCaseDir, caseDir, x):
        super().__init__(baseCaseDir, caseDir, x,
                         meshFile='2D_cylinder.msh22',
                         datFile='ics_temporals.txt',
                         jobFile='jobslurm.sh',
                         inputFile='2D_cylinder.in',
                         meshSFs=np.around(
                             np.arange(0.2, 2.0, 0.1), decimals=2)
                         )

    def _preProc_restart(self):
        self._preProc()
        dumpDir = os.path.join(self.caseDir, 'dump')
        latestXMF = getLatestXMF(dumpDir)
        in_lines = self.inputLines
        kw = 'RESTART_TYPE = GMSH'
        kw_line, kw_line_i = self.findKeywordLine(kw, in_lines)
        in_lines[kw_line_i] = '#' + kw + '\n'
        kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
        kw_line, kw_line_i = self.findKeywordLine(kw, in_lines)
        in_lines[kw_line_i] = '#' + kw + '\n'
        kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
        kw_line, kw_line_i = self.findKeywordLine(kw, in_lines)
        in_lines[kw_line_i] = '#' + kw + '\n'
        in_lines.append('RESTART_TYPE = XMF' + '\n')
        in_lines.append('RESTART_XMF_SOLUTION = dump/' + latestXMF + '\n')
        self.inputLines = in_lines

    def _preProc(self):
        ### EXTRACT VAR ###
        # Extract parameters for each individual
        omega = self.x[0]
        freq = self.x[1]
        ### SIMULATION INPUT PARAMETERS ###
        # open and read YALES2 input file to array of strings for each line
        in_lines = self.inputLines
        # find line that must change using a keyword
        keyword = 'CYL_ROTATION_PROP'
        keyword_line, keyword_line_i = self.findKeywordLine(keyword, in_lines)
        # create new string to replace line
        newLine = f'{keyword_line[:keyword_line.index("=")]}= {omega} {freq} \n'
        in_lines[keyword_line_i] = newLine
        # REPEAT FOR EACH LINE THAT MUST BE CHANGED
        self.inputLines = in_lines

    def _postProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        omega = self.x[0]
        freq = self.x[1]
        ######## Compute Objectives ##########
        ### Objective 1: Drag on Cylinder ###
        U = 1
        rho = 1
        D = 1
        # create string for directory of individual's data file
        data = np.genfromtxt(self.datPath, skip_header=1)
        # try:
        #     data = np.genfromtxt(dataDir, skip_header=1)
        # except IOError as err:
        #     print(err)
        #     print('ics_temporals.txt does not exist')
        #     obj = [None] * n_obj
        #     return obj
        # collect data after 100 seconds of simulation time
        mask = np.where(data[:, 1] > 100)
        # Surface integrals of Cp and Cf
        # DRAG: x-direction integrals
        # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
        p_over_rho_intgrl_1 = data[mask, 4]
        tau_intgrl_1 = data[mask, 6]
        F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
        C_drag = F_drag / ((1 / 2) * rho * U**2 * D**2)

        ### Objective 2 ###
        # Objective 2: Power consumed by rotating cylinder
        # t = 1  # [sec]
        D = 1  # [m] cylinder diameter
        th = 0.1  # [m] thickness of cylinder wall
        r_o = D / 2  # [m] outer radius
        r_i = r_o - th  # [m] inner radius
        d = 2700  # [kg/m^3] density of aluminum
        L = 1  # [m] length of cylindrical tube
        V = L * np.pi * (r_o**2 - r_i**2)  # [m^3] volume of cylinder
        m = d * V  # [kg] mass of cylinder
        # [kg m^2] moment of inertia of a hollow cylinder
        I = 0.5 * m * (r_i**2 + r_o**2)
        KE_consu = 0.5 * I * omega**2 * 4 * np.pi * freq * \
            quad(lambda t: abs(np.sin(2 * np.pi * freq * t)
                               * np.cos(2 * np.pi * freq * t)), 0, 1)[0]
        obj = [C_drag, KE_consu]
        self.f = obj

    def _genMesh(self):
        projName = '2D_cylinder'
        # dom_dx, dom_dy = 60, 25
        centX, centY, centZ = 0, 0, 0
        cylD = 1
        domD = cylD * 20
        meshSizeMax = 0.5
        #################################
        #          Initialize           #
        #################################
        gmsh.initialize()
        # By default Gmsh will not print out any messages: in order to output messages
        # on the terminal, just set the "General.Terminal" option to 1:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add(projName)
        gmsh.option.setNumber('Mesh.MeshSizeFactor', self.meshSF)
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
        cyl = gmsh.model.occ.addCircle(centX, centY, centZ, cylD)
        topPt = gmsh.model.occ.addPoint(centX, centY + domD, centZ)
        botPt = gmsh.model.occ.addPoint(centX, centY - domD, centZ)
        centPt = gmsh.model.occ.addPoint(centX, centY, centZ)
        inlet = gmsh.model.occ.addCircleArc(botPt, centPt, topPt)
        outlet = gmsh.model.occ.addCircleArc(topPt, centPt, botPt)
        outerLoop = gmsh.model.occ.addCurveLoop([inlet, outlet])
        innerLoop = gmsh.model.occ.addCurveLoop([cyl])
        dom = gmsh.model.occ.addPlaneSurface([innerLoop, outerLoop])
        # We finish by synchronizing the data from OpenCASCADE CAD kernel with
        # the Gmsh model:
        gmsh.model.occ.synchronize()
        #################################
        #    Physical Group Naming      #
        #################################
        dim = 2
        grpTag = gmsh.model.addPhysicalGroup(dim, [dom])
        gmsh.model.setPhysicalName(dim, grpTag, 'dom')
        dim = 1
        grpTag = gmsh.model.addPhysicalGroup(dim, [inlet])
        gmsh.model.setPhysicalName(dim, grpTag, 'x0')
        grpTag = gmsh.model.addPhysicalGroup(dim, [outlet])
        gmsh.model.setPhysicalName(dim, grpTag, 'x1')
        grpTag = gmsh.model.addPhysicalGroup(dim, [cyl])
        gmsh.model.setPhysicalName(dim, grpTag, 'cyl')
        #################################
        #           MESHING             #
        #################################
        # We could also use a `Box' field to impose a step change in element
        # sizes inside a box
        # boxF = gmsh.model.mesh.field.add("Box")
        # gmsh.model.mesh.field.setNumber(boxF, "VIn", meshSizeMax/10)
        # gmsh.model.mesh.field.setNumber(boxF, "VOut", meshSizeMax)
        # gmsh.model.mesh.field.setNumber(boxF, "XMin", cylD/3)
        # gmsh.model.mesh.field.setNumber(boxF, "XMax", cylD/3+cylD*10)
        # gmsh.model.mesh.field.setNumber(boxF, "YMin", -cylD)
        # gmsh.model.mesh.field.setNumber(boxF, "YMax", cylD)
        # # Finally, let's use the minimum of all the fields as the background mesh field:
        # minF = gmsh.model.mesh.field.add("Min")
        # gmsh.model.mesh.field.setNumbers(minF, "FieldsList", [boxF])
        # gmsh.model.mesh.field.setAsBackgroundMesh(minF)

        # Set minimum and maximum mesh size
        #gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
        gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)

        # Set number of nodes along cylinder wall
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 200)
        # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)

        # Set size of mesh at every point in model
        # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)

        # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
        # We can then generate a 2D mesh...
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.generate(2)
        # extract number of elements
        # get all elementary entities in the model
        entities = gmsh.model.getEntities()
        ##################
        #    FINALIZE    #
        ##################
        e = entities[-1]
        # get the mesh elements for each elementary entity
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
            e[0], e[1])
        # count number of elements
        self.numElem = sum(len(i) for i in elemTags)
        # ... and save it to disk
        gmsh.write(self.meshPath)
        # To visualize the model we can run the graphical user interface with
        # `gmsh.fltk.run()'.
        # gmsh.fltk.run()
        # This should be called when you are done using the Gmsh Python API:
        gmsh.finalize()

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
    #     # print(cylLoop)
    #     # print(rect)
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
    #
    #     # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
    #     # We can then generate a 2D mesh...
    #     gmsh.model.mesh.generate(1)
    #     gmsh.model.mesh.generate(2)
    #     # extract number of elements
    #     # get all elementary entities in the model
    #     entities = gmsh.model.getEntities()
    #     e = entities[-1]
    #     # get the mesh elements for each elementary entity
    #     elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
    #         e[0], e[1])
    #     # count number of elements
    #     self.numElem = sum(len(i) for i in elemTags)
    #     # ... and save it to disk
    #     gmsh.write(self.meshPath)
    #     # To visualize the model we can run the graphical user interface with
    #     # `gmsh.fltk.run()'.
    #     # gmsh.fltk.run()
    #     # This should be called when you are done using the Gmsh Python API:
    #     gmsh.finalize()


class OscillCylinderOpt(OptStudy):
    def __init__(self, algorithm, problem, baseCase,
                 *args, **kwargs):
        super().__init__(algorithm, problem, baseCase,
                         baseCaseDir='base_cases/osc-cyl_base',
                         optDatDir='cyl-opt_run',
                         *args, **kwargs)

    # def execute(self, cases):
    #     self.singleNodeExec(cases)


MyOptStudy = OscillCylinderOpt
BaseCase = OscillCylinder
