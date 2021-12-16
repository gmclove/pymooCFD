# @Author: glove
# @Date:   2021-12-10T10:32:00-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-14T13:23:23-05:00



import os
import h5py
import numpy as np

# from config import hqSim_dir

import os
from pymooCFD.util.gridInterp import GridInterp2D, GridInterp3D
from pymooCFD.core.optStudy import OptStudy
class RANSJetOpt(OptStudy):
    ###################################################
    #      High Quality Simulation Interpolation      #
    ###################################################
    ### Define grid interpolation parameters and perform interpolation on high
    ## quality simulation. These results can be compared to lower quality
    ## simulations on a universal grid.
    hqSim_dir = 'hq_sim'
    hqSim_y2DatDir = os.path.join(hqSim_dir, 'dump')
    y2DumpPrefix = 'pipe_expansion.sol'
    xmin, xmax = 1.0, 2.0
    ymin, ymax = -0.5, 0.5
    zmin, zmax = -0.5, 0.5
    t_begin, t_end = 80, 100
    t_resol = t_end - t_begin # highest quality
    gridInterp3D = GridInterp3D(y2DumpPrefix, xmin, xmax, ymin, ymax, zmin, zmax,
                                t_begin, t_end, t_resol,
                                x_resol = 200j)
    ## SPECIAL CASE: Radial Averaging
    ## initialize 2D gridInterp object for interpolating onto after getting
    ## radial average
    gridInterp2D = GridInterp2D(gridInterp3D.y2DumpPrefix,
                                gridInterp3D.xmin, gridInterp3D.xmax,
                                0.01, gridInterp3D.ymax,
                                gridInterp3D.t_begin, gridInterp3D.t_end,
                                gridInterp3D.t_resol,
                                x_resol = gridInterp3D.x_resol)
    ## stop expensive spatiotemporal interoplation from being performed each time
    ## by checking if path exists already
    hqGrid_uMag_path_3D = os.path.join(hqSim_dir, 'hqGrid_uMag-3D.npy')
    if not os.path.exists(hqGrid_uMag_path_3D):
        with h5py.File('hq_sim/merged-mesh.h5') as h5f:
            coor = h5f['XYZ'][:]
            print(coor.shape)
        with h5py.File('hq_sim/merged-u_mean.h5') as h5f:
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
        #save binary
        np.save(hqGrid_uMag_path, hqGrid_uMag)
    else:
        hqGrid_uMag = np.load(hqGrid_uMag_path)


    hqGrid_phi_path_3D = os.path.join(hqSim_dir, 'hqGrid_phi-3D.npy')
    if not os.path.exists(hqGrid_phi_path_3D):
        with h5py.File('hq_sim/merged-mesh.h5') as h5f:
            coor = h5f['XYZ'][:]
        with h5py.File('hq_sim/merged-phi_mean.h5') as h5f:
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
        #save binary
        np.save(hqGrid_phi_path, hqGrid_phi)
    else:
        hqGrid_phi = np.load(hqGrid_phi_path)

    def __init__(self, algorithm, problem, baseCase,
                *args, **kwargs):
        super().__init__(algorithm, problem, baseCase,
                        *args, **kwargs)

    def execute(self, cases):
        self.slurmExec(cases)

    def preProc(self):
        pass


MyOptStudy = RANSJetOpt
