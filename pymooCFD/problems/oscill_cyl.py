import gmsh
import numpy as np
import os

from pymooCFD.core.cfdCase import YALES2Case
from pymooCFD.core.meshStudy import MeshStudy


class OscillCylinder(YALES2Case):
    base_case_path = os.path.join(os.path.dirname(__file__), 'base_cases',
                                  'osc-cyl_base')
    ####### Define Design Space #########
    n_var = 2
    var_labels = ['Amplitude [radians/s]', 'Frequency [cycles/s]']
    var_type = ["real", "real"]  # options: 'int' or 'real'
    # xl = [0.1, 0.1]  # lower limits of parameters/variables
    # xu = [6, 1]  # upper limits of variables

    ####### Define Objective Space ########
    obj_labels = ['Change in Coefficient of Drag [%]',
                  'Resistive Force [N]']
    n_obj = 2
    ####### Define Constraints ########
    n_constr = 0
    ##### Local Execution Command #####
    externalSolver = True
    parallelize_preProc = False
    onlyParallelizeSolve = True
    nProc = 10
    procLim = 60
    solverExecCmd = ['mpirun', '-n', str(nProc), '2D_cylinder']

    def __init__(self, caseDir, x, meshSF=1.0, **kwargs):
        super().__init__(caseDir, x,
                         # optName='OscCylX2',
                         meshFile='2D_cylinder.msh22',
                         datFile='FORCES_temporal.txt',
                         jobFile='jobslurm.sh',
                         inputFile='2D_cylinder.in',
                         meshSF=meshSF,
                         # mesh_study=MeshStudy(self,
                         #     size_factors=[1.0, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0]),
                         **kwargs
                         )

    @classmethod
    def parallelize(cls, cases):
        cls.setup_parallelize(cls.externalSolver)
        for case in cases:
            case.preProc()
        print('PARALLELIZING . . .')
        with cls.Pool(cls.nTasks) as pool:
            for case in cases:
                pool.apply_async(case.solveAndPostProc, ())
            pool.close()
            pool.join()

    def solveAndPostProc(self):
        self.solve()
        self.postProc()

    def _preProc(self):
        self.genMesh()
        ### EXTRACT VAR ###
        # Extract parameters for each individual
        omega = self.x[0]
        freq = self.x[1]
        ### SIMULATION INPUT PARAMETERS ###
        # open and read YALES2 input file to array of strings for each line
        in_lines = self.input_lines_rw
        # find line that must change using a keyword
        keyword = 'CYL_ROTATION_PROP'
        kw_lines = self.findKeywordLines(keyword, in_lines)
        for kw_line_i, kw_line in kw_lines:
            # create new string to replace line
            newLine = f'{kw_line[:kw_line.index("=")]}= {omega} {freq} \n'
            in_lines[kw_line_i] = newLine
        # REPEAT FOR EACH LINE THAT MUST BE CHANGED
        self.input_lines_rw = in_lines

    def _postProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        # A_omega = self.x[0]
        # freq = self.x[1]
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
        prec_change = ((C_drag - C_drag_noOsc) / C_drag_noOsc) * 100

        ### Objective 2 ###
        # Objective 2: Power consumed by rotating cylinder
        res_torque = data[mask, 9]
        abs_mean_res_torque = np.mean(abs(res_torque))
        F_res = abs_mean_res_torque * D / 2

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


BaseCase = OscillCylinder


class OscillCylinder_SLURM(OscillCylinder):
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
    nTasks = 20


class OscillCylinder_Re500(OscillCylinder):
    base_case_path = os.path.join(os.path.dirname(__file__), 'base_cases',
                                  'osc-cyl_base-Re500')

class OscillCylinder_SLURM_Re500(OscillCylinder_SLURM):
    base_case_path = os.path.join(os.path.dirname(__file__), 'base_cases',
                                  'osc-cyl_base-Re500')
