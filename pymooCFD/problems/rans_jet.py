# @Author: glove
# @Date:   2021-12-14T16:17:44-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:59-05:00


# from pymooCFD.core.parallelProc import ParellelProc
# class MyParallelProc(ParallelProc):
#     def __init__(self):
#         super().__init__(externalSolver)


import gmsh
import math
from pymooCFD.core.cfdCase import FluentCase
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py

from pymooCFD.util.gridInterp import GridInterp2D, GridInterp3D, radialAvg


class RANSJet(FluentCase):
    base_case_path = 'base_cases/rans_jet-base'
    ###################################################
    #      High Quality Simulation Interpolation      #
    ###################################################
    # Define grid interpolation parameters and perform interpolation on high
    # quality simulation. These results can be compared to lower quality
    # simulations on a universal grid.
    hqSim_dir = os.path.join('rans_jet_opt', 'hq_sim')
    hqSim_y2DatDir = os.path.join(hqSim_dir, 'dump')
    y2DumpPrefix = 'pipe_expansion.sol'
    xmin, xmax = 1.0, 2.0
    ymin, ymax = -0.5, 0.5
    zmin, zmax = -0.5, 0.5
    t_begin, t_end = 80, 100
    t_resol = t_end - t_begin  # highest quality
    gridInterp3D = GridInterp3D(y2DumpPrefix, xmin, xmax, ymin, ymax, zmin, zmax,
                                t_begin, t_end, t_resol,
                                x_resol=200j)
    # SPECIAL CASE: Radial Averaging
    # initialize 2D gridInterp object for interpolating onto after getting
    # radial average
    gridInterp2D = GridInterp2D(gridInterp3D.y2DumpPrefix,
                                gridInterp3D.xmin, gridInterp3D.xmax,
                                0.01, gridInterp3D.ymax,
                                gridInterp3D.t_begin, gridInterp3D.t_end,
                                gridInterp3D.t_resol,
                                x_resol=gridInterp3D.x_resol)
    # stop expensive spatiotemporal interoplation from being performed each time
    # by checking if path exists already
    hqGrid_uMag_path_3D = os.path.join(hqSim_dir, 'hqGrid_uMag-3D.npy')
    if not os.path.exists(hqGrid_uMag_path_3D):
        with h5py.File(os.path.join(hqSim_dir, 'merged-mesh.h5')) as h5f:
            coor = h5f['XYZ'][:]
            # print(coor.shape)
        with h5py.File(os.path.join(hqSim_dir, 'merged-u_mean.h5')) as h5f:
            dset1 = list(h5f.keys())[0]
            uMean = h5f[dset1][:]
        mag_uMean = np.linalg.norm(uMean, axis=1)
        hqGrid_mag_uMean = gridInterp3D.getInterpGrid(coor, mag_uMean)
        np.save(hqGrid_uMag_path_3D, hqGrid_mag_uMean)
        gridInterp3D.plot3DGrid(hqGrid_mag_uMean, 'hqGrid_uMag-3D')
    else:
        hqGrid_uMag_3D = np.load(hqGrid_uMag_path_3D)

    hqGrid_uMag_path = os.path.join(hqSim_dir, 'hqGrid_uMag.npy')
    if not os.path.exists(hqGrid_uMag_path):
        # radial average
        hqGrid_uMag = radialAvg(hqGrid_uMag_3D, gridInterp3D, gridInterp2D)
        gridInterp2D.plot2DGrid(hqGrid_uMag, 'hqGrid_uMag_radAvg')
        # save binary
        np.save(hqGrid_uMag_path, hqGrid_uMag)
    else:
        hqGrid_uMag = np.load(hqGrid_uMag_path)

    hqGrid_phi_path_3D = os.path.join(hqSim_dir, 'hqGrid_phi-3D.npy')
    if not os.path.exists(hqGrid_phi_path_3D):
        with h5py.File(os.path.join(hqSim_dir, 'merged-mesh.h5')) as h5f:
            coor = h5f['XYZ'][:]
        with h5py.File(os.path.join(hqSim_dir, 'merged-phi_mean.h5')) as h5f:
            dset1 = list(h5f.keys())[0]
            phiMean = h5f[dset1][:]
        hqGrid_phi_3D = gridInterp3D.getInterpGrid(coor, phiMean)
        np.save(hqGrid_phi_path_3D, hqGrid_phi_3D)
        gridInterp3D.plot3DGrid(hqGrid_phi_3D, 'hqGrid_phi-3D')
    else:
        hqGrid_phi_3D = np.load(hqGrid_phi_path_3D)

    hqGrid_phi_path = os.path.join(hqSim_dir, 'hqGrid_phi.npy')
    if not os.path.exists(hqGrid_phi_path):
        # radial average
        hqGrid_phi = radialAvg(hqGrid_phi_3D, gridInterp3D, gridInterp2D)
        gridInterp2D.plot2DGrid(hqGrid_phi, 'hqGrid_phi_radAvg')
        # save binary
        np.save(hqGrid_phi_path, hqGrid_phi)
    else:
        hqGrid_phi = np.load(hqGrid_phi_path)

    ####### Define Design Space #########
    n_var = 2
    var_labels = ['Mouth Diameter [m]', 'Breath Velocity [m/s]']
    var_type = ["real", "real"]  # OPTIONS: 'int' or 'real'
    # xl = [0.005, 0.1]  # lower limits of parameters/variables
    # xu = [0.04, 0.4]  # upper limits of variables

    ####### Define Objective Space ########
    n_obj = 2
    obj_labels = ['Velocity Field Error [m/s]', 'Scalar Concentration Error']
    n_constr = 0
    ##### Execution Command #####
    externalSolver = True
    onlyParallelizeSolve = True
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
    # nProc = 10
    # procLim = 60
    nTasks = 4

    def __init__(self, case_path, x):
        super().__init__(case_path, x,
                         meshSF=0.4,
                         meshSFs=np.append(
                             np.around(np.arange(0.3, 1.6, 0.1), decimals=2),
                             [0.25, 0.35, 0.45]),
                         meshFile='jet_rans-axi_sym.unv',
                         datFile='jet_rans-axi_sym.cgns',
                         jobFile='jobslurm.sh',
                         inputFile='jet_rans-axi_sym.jou'
                         )

    def _execDone(self):
        if os.path.exists(self.datPath):
            return True

    def _genMesh(self):
        mouthD = self.x[0]

        projName = 'jet_cone_axi-sym'

        cyl_r = mouthD / 2  # 0.02/2

        sph_r = (0.15 / 0.02) * cyl_r
        sph_xc = -sph_r
        sph_yc = 0

        cone_dx = 3.5
        cone_r1 = 0.5 / 2
        cone_r2 = cone_dx * (0.7 / 2.5) + cone_r1
        cone_xc = -0.5

        # Meshing Scheme
        NN_inlet = int((20 / 0.01) * cyl_r)
        sph_wall_len = sph_r * math.pi - \
            math.asin(cyl_r / (2 * math.pi)) * sph_r
        NN_sph_wall = int((100 / 0.2355) * sph_wall_len)
        NN_axis_r = 500
        axis_l_len = (sph_xc - sph_r) + 0.5
        NN_axis_l = int((40 / 0.35) * axis_l_len)
        NN_cyl_axis = int((70 / 0.075) * sph_r)
        #################################
        #          Initialize           #
        #################################
        gmsh.initialize()
        gmsh.model.add(projName)
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFactor', self.meshSF)
        gmsh.option.setNumber('Mesh.FlexibleTransfinite', 1)
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
        ##### Points #####
        # Cone
        cone_ll = gmsh.model.occ.addPoint(cone_xc, 0, 0)
        cone_lr = gmsh.model.occ.addPoint(cone_dx + cone_xc, 0, 0)
        cone_ur = gmsh.model.occ.addPoint(cone_dx + cone_xc, cone_r2, 0)
        cone_ul = gmsh.model.occ.addPoint(cone_xc, cone_r1, 0)
        # sphere
        sph_start = gmsh.model.occ.addPoint(sph_xc - sph_r, sph_yc, 0)
        sph_cent = gmsh.model.occ.addPoint(sph_xc, sph_yc, 0)
        sph_end_x = sph_xc + math.sqrt(sph_r**2 - cyl_r**2)
        sph_end = gmsh.model.occ.addPoint(sph_end_x, sph_yc + cyl_r, 0)
        # cylinder
        cyl_ll = sph_cent
        cyl_ul = gmsh.model.occ.addPoint(sph_xc, sph_yc + cyl_r, 0)
        cyl_ur = sph_end
        cyl_mid_outlet = gmsh.model.occ.addPoint(sph_xc + sph_r, 0, 0)
        ##### Curves #####
        # sphere wall
        sph_wall = gmsh.model.occ.addCircleArc(sph_start, sph_cent, sph_end)
        # cylinder wall
        cyl_wall = gmsh.model.occ.addLine(cyl_ul, cyl_ur)
        # cylinder axis (for meshing control)
        axis_cyl = cyl_axis = gmsh.model.occ.addLine(cyl_ll, cyl_mid_outlet)
        # inlet (cylinder back wall)
        inlet = gmsh.model.occ.addLine(cyl_ul, cyl_ll)
        # axis
        axis_l = gmsh.model.occ.addLine(sph_start, cone_ll)
        axis_r = gmsh.model.occ.addLine(cyl_mid_outlet, cone_lr)
        axis = [axis_l, axis_cyl, axis_r]
        # cone
        back_wall = gmsh.model.occ.addLine(cone_ll, cone_ul)
        cone_wall = gmsh.model.occ.addLine(cone_ul, cone_ur)
        outlet = gmsh.model.occ.addLine(cone_ur, cone_lr)
        # mesh field line
        # field_line = gmsh.model.occ.addLine(sph_end, cone_ur)

        ##### Surfaces #####
        # domain surface
        curv_loop_tags = [sph_wall, cyl_wall, inlet, axis_cyl, axis_r, outlet,
                          cone_wall, back_wall, axis_l]
        dom_loop = gmsh.model.occ.addCurveLoop(curv_loop_tags)
        dom = gmsh.model.occ.addPlaneSurface([dom_loop])

        gmsh.model.occ.synchronize()
        #################################
        #           MESHING             #
        #################################
        gmsh.model.mesh.setTransfiniteCurve(sph_wall, NN_sph_wall)
        gmsh.model.mesh.setTransfiniteCurve(cyl_wall, NN_cyl_axis)
        gmsh.model.mesh.setTransfiniteCurve(cyl_axis, NN_cyl_axis)
        gmsh.model.mesh.setTransfiniteCurve(inlet, NN_inlet)
        gmsh.model.mesh.setTransfiniteCurve(
            axis_r, NN_axis_r, meshType='Progression', coef=1.005)
        gmsh.model.mesh.setTransfiniteCurve(axis_l, NN_axis_l)
        gmsh.model.mesh.setTransfiniteCurve(back_wall, 15)
        gmsh.model.mesh.setTransfiniteCurve(
            outlet, 50, meshType='Progression', coef=0.99)
        gmsh.model.mesh.setTransfiniteCurve(cone_wall, 100)
        ##### Execute Meshing ######
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.generate(2)
        # extract number of elements
        # get all elementary entities in the model
        entities = gmsh.model.getEntities()
        e = entities[-1]
        # get the mesh elements for each elementary entity
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
            e[0], e[1])
        # count number of elements
        self.numElem = sum(len(i) for i in elemTags)
        #################################
        #    Physical Group Naming      #
        #################################
        # 1-D physical groups
        dim = 1
        # inlet
        grpTag = gmsh.model.addPhysicalGroup(dim, [inlet])
        gmsh.model.setPhysicalName(dim, grpTag, 'inlet')
        # outlet
        grpTag = gmsh.model.addPhysicalGroup(dim, [outlet])
        gmsh.model.setPhysicalName(dim, grpTag, 'outlet')
        # axis of symmetry
        grpTag = gmsh.model.addPhysicalGroup(dim, axis)
        gmsh.model.setPhysicalName(dim, grpTag, 'axis')
        # walls
        grpTag = gmsh.model.addPhysicalGroup(dim, [sph_wall, cyl_wall])
        gmsh.model.setPhysicalName(dim, grpTag, 'walls')
        # coflow
        grpTag = gmsh.model.addPhysicalGroup(dim, [back_wall, cone_wall])
        gmsh.model.setPhysicalName(dim, grpTag, 'coflow')

        # 2-D physical groups
        dim = 2
        grpTag = gmsh.model.addPhysicalGroup(dim, [dom])
        gmsh.model.setPhysicalName(dim, grpTag, 'dom')

        #################################
        #        Write Mesh File        #
        #################################
        gmsh.write(self.meshPath)
        # if '-nopopup' not in sys.argv:
        #     gmsh.fltk.run()
        gmsh.finalize()

    def _preProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        outVel = self.x[self.var_labels.index('Breath Velocity [m/s]')]
        # outD = var[var_labels.index('Mouth Diameter')]
        ##### Generate New Mesh ######
        self.genMesh()
        ##### Write Entire Input File #####
        # useful if input file is short enough
        coflowVel = 0.005  # outVel*(2/100)
        lines = [
            # IMPORT
            f'/file/import ideas-universal {self.meshFile}',
            # AUTO-SAVE
            '/file/auto-save case-frequency if-case-is-modified',
            '/file/auto-save data-frequency 1000',
            # MODEL
            '/define/models axisymmetric y',
            '/define/models/viscous kw-sst y',
            # species
            '/define/models/species species-transport y mixture-template',
            '/define/materials change-create air scalar n n n n n n n n',
            '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
            # BOUNDARY CONDITIONS
            # outlet
            '/define/boundary-conditions/modify-zones/zone-type outlet pressure-outlet ;outflow',
            # coflow
            '/define/boundary-conditions/modify-zones/zone-type coflow velocity-inlet',
            f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel*2} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
            # inlet
            '/define/boundary-conditions/modify-zones/zone-type inlet velocity-inlet',
            f'/define/boundary-conditions velocity-inlet inlet n n y y n {outVel} n 0 n 300 n n y 5 10 n n 1',
            # axis
            '/define/boundary-conditions/modify-zones/zone-type axis axis',
            # INITIALIZE
            '/solve/initialize/hyb-initialization',
            # CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
            # SOLVE
            '/solve/iterate 1000',
            # change convergence, methods and coflow speed
            '/solve/set discretization-scheme species-0 6',
            '/solve/set discretization-scheme mom 6',
            '/solve/set discretization-scheme k 6',
            '/solve/set discretization-scheme omega 6',
            '/solve/set discretization-scheme temperature 6',
            f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
            '/solve/iterate 4000',
            # EXPORT
            f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
            'OK',
            f'/file write-case-data {self.datFile}',
            'OK',
            '/exit',
            'OK'
        ]
        self.inputLines = lines
        ####### Slurm Job Lines #########
        lines = ['#!/bin/bash',
                 "#SBATCH --partition=ib --constraint='ib&haswell_1'",
                 '#SBATCH --cpus-per-task=20',
                 '#SBATCH --ntasks=10',
                 '#SBATCH --time=00:30:00',
                 '#SBATCH --mem-per-cpu=2G',
                 '#SBATCH --job-name=jet_rans',
                 '#SBATCH --output=slurm.out',
                 'module load ansys/fluent-21.2.0',
                 'cd $SLURM_SUBMIT_DIR',
                 'time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out'
                 ]
        self.jobLines = lines

    # def _preProc_restart(self):
        # # get latest autosave and load this into fluent
        # inLines = [
        #     # # IMPORT
        #     # f'/file/import ideas-universal {self.meshFile}',
        #     # # AUTO-SAVE
        #     # '/file/auto-save case-frequency if-case-is-modified',
        #     # '/file/auto-save data-frequency 1000',
        #     # # MODEL
        #     # '/define/models axisymmetric y',
        #     # '/define/models/viscous kw-sst y',
        #     # # species
        #     # '/define/models/species species-transport y mixture-template',
        #     # '/define/materials change-create air scalar n n n n n n n n',
        #     # '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
        #     # # BOUNDARY CONDITIONS
        #     # # outlet
        #     # '/define/boundary-conditions/modify-zones/zone-type outlet pressure-outlet ;outflow',
        #     # # coflow
        #     # '/define/boundary-conditions/modify-zones/zone-type coflow velocity-inlet',
        #     # f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel*2} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
        #     # # inlet
        #     # '/define/boundary-conditions/modify-zones/zone-type inlet velocity-inlet',
        #     # f'/define/boundary-conditions velocity-inlet inlet n n y y n {outVel} n 0 n 300 n n y 5 10 n n 1',
        #     # # axis
        #     # '/define/boundary-conditions/modify-zones/zone-type axis axis',
        #     # # INITIALIZE
        #     # '/solve/initialize/hyb-initialization',
        #     # # CHANGE CONVERGENCE CRITERIA
        #     # '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
        #     # # SOLVE
        #     # '/solve/iterate 1000',
        #     # Load
        #     '/file/load'
        #     # change convergence, methods and coflow speed
        #     '/solve/set discretization-scheme species-0 6',
        #     '/solve/set discretization-scheme mom 6',
        #     '/solve/set discretization-scheme k 6',
        #     '/solve/set discretization-scheme omega 6',
        #     '/solve/set discretization-scheme temperature 6',
        #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
        #     '/solve/iterate 4000',
        #     # EXPORT
        #     f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
        #     'OK',
        #     f'/file write-case-data {self.datFile}',
        #     'OK',
        #     '/exit',
        #     'OK'
        #     ]
        # pass
        # self._preProc()
        # outVel = self.x[self.var_labels.index('Breath Velocity')]
        # ##### Write Entire Input File #####
        # # useful if input file is short enough
        # coflowVel=0.005  # outVel*(2/100)
        # lines=[
        #     # IMPORT
        #     f'/file/import ideas-universal {self.meshFile}',
        #     # AUTO-SAVE
        #     '/file/auto-save case-frequency if-case-is-modified',
        #     '/file/auto-save data-frequency 1000',
        #     # MODEL
        #     '/define/models axisymmetric y',
        #     '/define/models/viscous kw-sst y',
        #     # species
        #     '/define/models/species species-transport y mixture-template',
        #     '/define/materials change-create air scalar n n n n n n n n',
        #     '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
        #     # BOUNDARY CONDITIONS
        #     # outlet
        #     '/define/boundary-conditions/modify-zones/zone-type outlet pressure-outlet ;outflow',
        #     # coflow
        #     '/define/boundary-conditions/modify-zones/zone-type coflow velocity-inlet',
        #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel*2} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
        #     # inlet
        #     '/define/boundary-conditions/modify-zones/zone-type inlet velocity-inlet',
        #     f'/define/boundary-conditions velocity-inlet inlet n n y y n {outVel} n 0 n 300 n n y 5 10 n n 1',
        #     # axis
        #     '/define/boundary-conditions/modify-zones/zone-type axis axis',
        #     # INITIALIZE
        #     '/solve/initialize/hyb-initialization',
        #     # CHANGE CONVERGENCE CRITERIA
        #     '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
        #     # SOLVE
        #     '/solve/iterate 1000',
        #     # change convergence, methods and coflow speed
        #     '/solve/set discretization-scheme species-0 6',
        #     '/solve/set discretization-scheme mom 6',
        #     '/solve/set discretization-scheme k 6',
        #     '/solve/set discretization-scheme omega 6',
        #     '/solve/set discretization-scheme temperature 6',
        #     f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
        #     '/solve/iterate 4000',
        #     # EXPORT
        #     f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
        #     'OK',
        #     f'/file write-case-data {self.datFile}',
        #     'OK',
        #     '/exit',
        #     'OK'
        #     ]
        # self.inputLines=lines

    def _postProc(self):
        ####### EXTRACT VAR ########
        # OPTIONAL: Extract parameters for each individual
        # sometimes variables are used in the computation of the objectives

        ######## Compute Objectives ##########
        ######## Objective 1: Mean Difference in Scalar Distribution #########
        coor, dat = self.gridInterp2D.getCGNSData(
            self.datPath, 'Mass_fraction_of_scalar')
        ransGrid_phi = self.gridInterp2D.getInterpGrid(coor, dat)
        dnsGrid_phi = self.hqGrid_phi
        phi_meanDiff = np.mean(abs(ransGrid_phi - dnsGrid_phi))

        ######## Objective 2: Mean Difference in Velocity Magnitude #########
        coor, dat = self.gridInterp2D.getCGNSData(
            self.datPath, 'VelocityMagnitude')
        ransGrid_uMag = self.gridInterp2D.getInterpGrid(coor, dat)
        dnsGrid_uMag = self.hqGrid_uMag
        uMag_meanDiff = np.mean(abs(ransGrid_uMag - dnsGrid_uMag))

        obj = [phi_meanDiff, uMag_meanDiff]
        self.f = obj

        ##### SAVE DATA VISUALIZATION ######
        # phi grid plot
        plt.imshow(ransGrid_phi.T, extent=(self.gridInterp2D.xmin, self.gridInterp2D.xmax,
                   self.gridInterp2D.ymin, self.gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS - Mass Fraction of Scalar')
        path = os.path.join(self.abs_path, 'RANS-phi-grid.png')
        plt.savefig(path)
        plt.clf()
        # phi difference plot
        phiDiffGrid = ransGrid_phi - dnsGrid_phi
        plt.imshow(phiDiffGrid.T, extent=(self.gridInterp2D.xmin, self.gridInterp2D.xmax,
                   self.gridInterp2D.ymin, self.gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS DNS Difference - Mass Fraction of Scalar')
        path = os.path.join(self.abs_path, 'diff-phi-grid.png')
        plt.savefig(path)
        plt.clf()
        # uMag grid plot
        plt.imshow(ransGrid_uMag.T, extent=(self.gridInterp2D.xmin, self.gridInterp2D.xmax,
                   self.gridInterp2D.ymin, self.gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS - Velocity Magnitude')
        path = os.path.join(self.abs_path, 'RANS-uMag-grid.png')
        plt.savefig(path)
        plt.clf()
        # uMag difference plot
        uMagDiffGrid = ransGrid_uMag - dnsGrid_uMag
        plt.imshow(uMagDiffGrid.T, extent=(self.gridInterp2D.xmin, self.gridInterp2D.xmax,
                   self.gridInterp2D.ymin, self.gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS DNS Difference - Velocity Magnitude')
        path = os.path.join(self.abs_path, 'diff-uMag-grid.png')
        plt.savefig(path)
        plt.clf()

BaseCase = RANSJet
