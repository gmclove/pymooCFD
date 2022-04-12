import os
import numpy as np
import h5py
from pymooCFD.util.gridInterp import GridInterp2D, GridInterp3D, radialAvg

###################################################
#      High Quality Simulation Interpolation      #
###################################################
# Define grid interpolation parameters and perform interpolation on high
# quality simulation. These results can be compared to lower quality
# simulations on a universal grid.


def main():
    # hqInterp('hq_sim/PulsedJetIrregular/mesh_study/meshSF-0.7')
    # hqInterp('hq_sim/PulsedJetIrregular/mesh_study/meshSF-0.8')
    hqInterp(
        'git_ignore_folder/rans_jet_opt/hq_sim/PulsedJetIrregular/mesh_study/meshSF-0.9')
    # hqInterp('hq_sim/PulsedJetIrregular/mesh_study/meshSF-1.1')
    # hqInterp('hq_sim/PulsedJetIrregular/mesh_study/meshSF-1.2')


def hqInterp(caseDir):
    hqSim_y2DatDir = os.path.join(caseDir, 'dump')
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
    hqGrid_uMag_path_3D = os.path.join(caseDir, 'hqGrid_uMag-3D.npy')
    if not os.path.exists(hqGrid_uMag_path_3D):
        with h5py.File(os.path.join(caseDir, 'merged-mesh.h5')) as h5f:
            coor = h5f['XYZ'][:]
            print(coor.shape)
        with h5py.File(os.path.join(caseDir, 'merged-u_mean.h5')) as h5f:
            dset1 = list(h5f.keys())[0]
            uMean = h5f[dset1][:]
        mag_uMean = np.linalg.norm(uMean, axis=1)
        hqGrid_uMag_3D = gridInterp3D.getInterpGrid(coor, mag_uMean)
        np.save(hqGrid_uMag_path_3D, hqGrid_uMag_3D)
        gridInterp3D.plot3DGrid(hqGrid_mag_uMean, 'hqGrid_uMag-3D')
    else:
        hqGrid_uMag_3D = np.load(hqGrid_uMag_path_3D)

    hqGrid_uMag_path = os.path.join(caseDir, 'hqGrid_uMag.npy')
    if not os.path.exists(hqGrid_uMag_path):
        # radial average
        hqGrid_uMag = radialAvg(hqGrid_uMag_3D, gridInterp3D, gridInterp2D)
        gridInterp2D.plot2DGrid(hqGrid_uMag, 'hqGrid_uMag_radAvg')
        # save binary
        np.save(hqGrid_uMag_path, hqGrid_uMag)
    else:
        hqGrid_uMag = np.load(hqGrid_uMag_path)

    hqGrid_phi_path_3D = os.path.join(caseDir, 'hqGrid_phi-3D.npy')
    if not os.path.exists(hqGrid_phi_path_3D):
        with h5py.File(os.path.join(caseDir, 'merged-mesh.h5')) as h5f:
            coor = h5f['XYZ'][:]
        with h5py.File(os.path.join(caseDir, 'merged-phi_mean.h5')) as h5f:
            dset1 = list(h5f.keys())[0]
            phiMean = h5f[dset1][:]
        hqGrid_phi_3D = gridInterp3D.getInterpGrid(coor, phiMean)
        np.save(hqGrid_phi_path_3D, hqGrid_phi_3D)
        gridInterp3D.plot3DGrid(hqGrid_phi_3D, 'hqGrid_phi-3D')
    else:
        hqGrid_phi_3D = np.load(hqGrid_phi_path_3D)

    hqGrid_phi_path = os.path.join(caseDir, 'hqGrid_phi.npy')
    if not os.path.exists(hqGrid_phi_path):
        # radial average
        hqGrid_phi = radialAvg(hqGrid_phi_3D, gridInterp3D, gridInterp2D)
        gridInterp2D.plot2DGrid(hqGrid_phi, 'hqGrid_phi_radAvg')
        # save binary
        np.save(hqGrid_phi_path, hqGrid_phi)
    else:
        hqGrid_phi = np.load(hqGrid_phi_path)


if __name__ == '__main__':
    main()
