# @Author: glove
# @Date:   2021-12-10T10:32:00-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-14T16:29:52-05:00
import gmsh
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from pymooCFD.myStudies import RANSJetOpt
from pymooCFD.core.cfdCase import CFDCase
class RANSJet(CFDCase):
    ####### Define Design Space #########
    n_var = 2
    var_labels = ['Mouth Diameter', 'Breath Velocity']
    varType =    ["real", "real"]  ## OPTIONS: 'int' or 'real'
    xl =         [0.005, 0.1] ## lower limits of parameters/variables
    xu =         [0.04, 0.4]  ## upper limits of variables
    if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
        raise Exception("Design Space Definition Incorrect")
    ####### Define Objective Space ########
    obj_labels = ['Velocity Field Error', 'Particle Concentration Error']
    n_obj = 2
    n_constr = 0

    n_obj = 2
    obj_labels = ['Velocity Field Error', 'Particle Concentration Error']
    n_constr = 0
    ##### Local Execution Command #####
    nProc = 8
    solverExecCmd = ['C:\"Program Files"\"Ansys Inc"\v211\fluent\ntbin\win64\fluent.exe',
    '2ddp', f'-t{nProc}', '-g', '-i', 'jet_rans-axi_sym.jou', '>', 'run.out']

    def __init__(self, baseCaseDir, caseDir, x):
        super().__init__(baseCaseDir, caseDir, x,
                        # var_labels = var_labels,
                        # obj_labels = obj_labels,
                        # n_var = n_var,
                        # n_obj = n_obj,
                        meshFile = 'jet_rans-axi_sym.unv',
                        datFile = 'jet_rans-axi_sym.cgns',
                        jobFile = 'jobslurm.sh',
                        inputFile = 'jet_rans-axi_sym.jou',
                        # solverExecCmd = solverExecCmd,
                        # nProc = nProc
                        )
    def _genMesh(self):
        mouthD = self.x[0]

        projName = 'jet_cone_axi-sym'

        cyl_r = mouthD/2 #0.02/2

        sph_r = (0.15/0.02)*cyl_r
        sph_xc = -sph_r
        sph_yc = 0

        cone_dx = 3.5
        cone_r1 = 0.5/2
        cone_r2 = cone_dx*(0.7/2.5)+cone_r1
        cone_xc = -0.5

        ### Meshing Scheme
        NN_inlet = int((20/0.01)*cyl_r)
        sph_wall_len = sph_r*math.pi - math.asin(cyl_r/(2*math.pi))*sph_r
        NN_sph_wall = int((100/0.2355)*sph_wall_len)
        NN_axis_r = 500
        axis_l_len = (sph_xc-sph_r)+0.5
        NN_axis_l = int((40/0.35)*axis_l_len)
        NN_cyl_axis = int((70/0.075)*sph_r)
        #################################
        #          Initialize           #
        #################################
        gmsh.initialize()
        gmsh.model.add(projName)
        gmsh.option.setNumber('General.Terminal', 0)
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
        ##### Points #####
        # Cone
        cone_ll = gmsh.model.occ.addPoint(cone_xc, 0, 0)
        cone_lr = gmsh.model.occ.addPoint(cone_dx+cone_xc, 0, 0)
        cone_ur = gmsh.model.occ.addPoint(cone_dx+cone_xc, cone_r2, 0)
        cone_ul = gmsh.model.occ.addPoint(cone_xc, cone_r1, 0)
        # sphere
        sph_start = gmsh.model.occ.addPoint(sph_xc-sph_r, sph_yc, 0)
        sph_cent = gmsh.model.occ.addPoint(sph_xc, sph_yc, 0)
        sph_end_x = sph_xc + math.sqrt(sph_r**2 - cyl_r**2)
        sph_end = gmsh.model.occ.addPoint(sph_end_x, sph_yc+cyl_r, 0)
        # cylinder
        cyl_ll = sph_cent
        cyl_ul = gmsh.model.occ.addPoint(sph_xc, sph_yc+cyl_r, 0)
        cyl_ur = sph_end
        cyl_mid_outlet = gmsh.model.occ.addPoint(sph_xc+sph_r, 0, 0)
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
        gmsh.model.mesh.setTransfiniteCurve(axis_r, NN_axis_r, meshType='Progression', coef=1.005)
        gmsh.model.mesh.setTransfiniteCurve(axis_l, NN_axis_l)
        gmsh.model.mesh.setTransfiniteCurve(back_wall, 15)
        gmsh.model.mesh.setTransfiniteCurve(outlet, 50, meshType='Progression', coef=0.99)
        gmsh.model.mesh.setTransfiniteCurve(cone_wall, 100)
        ##### Execute Meshing ######
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.generate(2)
        ### extract number of elements
        # get all elementary entities in the model
        entities = gmsh.model.getEntities()
        e = entities[-1]
        # get the mesh elements for each elementary entity
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(e[0], e[1])
        # count number of elements
        numElem = sum(len(i) for i in elemTags)
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
        return numElem

    def _preProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        outVel = self.x[self.var_labels.index('Breath Velocity')]
        # outD = var[var_labels.index('Mouth Diameter')]
        ##### Generate New Mesh ######
        self.genMesh()
        ##### Write Entire Input File #####
        ### useful if input file is short enough
        coflowVel = 0.005 # outVel*(2/100)
        lines = [
            ## IMPORT
            f'/file/import ideas-universal {self.meshFile}',
            ## AUTO-SAVE
            '/file/auto-save case-frequency if-case-is-modified',
    	    '/file/auto-save data-frequency 1000',
            ## MODEL
            '/define/models axisymmetric y',
            '/define/models/viscous kw-sst y',
            # species
            '/define/models/species species-transport y mixture-template',
            '/define/materials change-create air scalar n n n n n n n n',
            '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
            ## BOUNDARY CONDITIONS
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
            ## INITIALIZE
            '/solve/initialize/hyb-initialization',
            ## CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
            ## SOLVE
            '/solve/iterate 1000',
            # change convergence, methods and coflow speed
            '/solve/set discretization-scheme species-0 6',
            '/solve/set discretization-scheme mom 6',
            '/solve/set discretization-scheme k 6',
            '/solve/set discretization-scheme omega 6',
            '/solve/set discretization-scheme temperature 6',
            f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
            '/solve/iterate 4000',
            ## EXPORT
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
                '#SBATCH --nodes=1',
                '#SBATCH --ntasks=20',
                '#SBATCH --time=00:20:00',
                '#SBATCH --mem-per-cpu=2G',
                '#SBATCH --job-name=jet_rans',
                '#SBATCH --output=slurm.out',
                'module load ansys/fluent-21.2.0',
                'cd $SLURM_SUBMIT_DIR',
                'time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out'
                ]
        self.jobLines = lines

    def _preProc_restart(self):
        self._preProc()
        ##### Write Entire Input File #####
        ### useful if input file is short enough
        coflowVel = 0.005 # outVel*(2/100)
        lines = [
            ## IMPORT
            f'/file/import ideas-universal {self.meshFile}',
            ## AUTO-SAVE
            '/file/auto-save case-frequency if-case-is-modified',
    	    '/file/auto-save data-frequency 1000',
            ## MODEL
            '/define/models axisymmetric y',
            '/define/models/viscous kw-sst y',
            # species
            '/define/models/species species-transport y mixture-template',
            '/define/materials change-create air scalar n n n n n n n n',
            '/define/materials change-create mixture-template mixture-template y 2 scalar air 0 0 n n n n n n',
            ## BOUNDARY CONDITIONS
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
            ## INITIALIZE
            '/solve/initialize/hyb-initialization',
            ## CHANGE CONVERGENCE CRITERIA
            '/solve/monitors/residual convergence-criteria 1e-5 1e-6 1e-6 1e-6 1e-6 1e-5 1e-6',
            ## SOLVE
            '/solve/iterate 1000',
            # change convergence, methods and coflow speed
            '/solve/set discretization-scheme species-0 6',
            '/solve/set discretization-scheme mom 6',
            '/solve/set discretization-scheme k 6',
            '/solve/set discretization-scheme omega 6',
            '/solve/set discretization-scheme temperature 6',
            f'/define/boundary-conditions velocity-inlet coflow y y n {coflowVel} n 0 n 1 n 0 n 300 n n y 5 10 n n 0',
            '/solve/iterate 4000',
            ## EXPORT
            f'/file/export cgns {self.datFile} n y velocity-mag scalar q',
            'OK',
            f'/file write-case-data {self.datFile}',
            'OK',
            '/exit',
            'OK'
            ]
        self.inputLines = lines


    def _postProc(self):
        gridInterp2D = RANSJetOpt.gridInterp2D
        gridInterp3D = RANSJetOpt.gridInterp3D
        dnsGrid_uMag = RANSJetOpt.hqGrid_uMag
        dnsGrid_phi = RANSJetOpt.hqGrid_phi
        ######## Objective 1: Mean Difference in Scalar Distribution #########
        coor, dat = gridInterp2D.getCGNSData(self.datPath, 'Mass_fraction_of_scalar')
        ransGrid_phi = gridInterp2D.getInterpGrid(coor, dat)
        phi_meanDiff = np.mean(abs(ransGrid_phi - dnsGrid_phi))
        ######## Objective 2: Mean Difference in Velocity Magnitude #########
        coor, dat = gridInterp2D.getCGNSData(self.datPath, 'VelocityMagnitude')
        ransGrid_uMag = gridInterp2D.getInterpGrid(coor, dat)
        uMag_meanDiff = np.mean(abs(ransGrid_uMag - dnsGrid_uMag))

        obj = [phi_meanDiff, uMag_meanDiff]

        ##### SAVE DATA VISUALIZATION ######
        # phi grid plot
        plt.imshow(ransGrid_phi.T, extent=(gridInterp2D.xmin, gridInterp2D.xmax,
                    gridInterp2D.ymin, gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS - Mass Fraction of Scalar')
        path = os.path.join(self.caseDir, 'RANS-phi-grid.png')
        plt.savefig(path)
        plt.clf()
        # phi difference plot
        phiDiffGrid = ransGrid_phi - dnsGrid_phi
        plt.imshow(phiDiffGrid.T, extent=(gridInterp2D.xmin, gridInterp2D.xmax,
                    gridInterp2D.ymin, gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS DNS Difference - Mass Fraction of Scalar')
        path = os.path.join(self.caseDir, 'diff-phi-grid.png')
        plt.savefig(path)
        plt.clf()
        # uMag grid plot
        plt.imshow(ransGrid_uMag.T, extent=(gridInterp2D.xmin, gridInterp2D.xmax,
                    gridInterp2D.ymin, gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS - Velocity Magnitude')
        path = os.path.join(self.caseDir, 'RANS-uMag-grid.png')
        plt.savefig(path)
        plt.clf()
        # uMag difference plot
        uMagDiffGrid = ransGrid_uMag - dnsGrid_uMag
        plt.imshow(uMagDiffGrid.T, extent=(gridInterp2D.xmin, gridInterp2D.xmax,
                    gridInterp2D.ymin, gridInterp2D.ymax), origin='lower')
        plt.colorbar()
        plt.title('RANS DNS Difference - Velocity Magnitude')
        path = os.path.join(self.caseDir, 'diff-uMag-grid.png')
        plt.savefig(path)
        plt.clf()

        return obj

    def _execComplete(self):
        complete = False
        if os.path.exists(self.datPath):
            complete = True
        return complete



BaseCase = RANSJet
