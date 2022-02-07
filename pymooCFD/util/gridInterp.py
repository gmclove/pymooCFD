import numpy as np
from scipy.interpolate import griddata
import os
import time
import h5py
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def radialAvg(grid3D, gridInterp3D, gridInterp2D, num_x_slices=None):
    if num_x_slices is None:
        num_x_slices = grid3D.shape[0]
    x_step = int(len(grid3D)/num_x_slices)
    xGrid = gridInterp3D.grid_x[::x_step]
    yGrid = gridInterp3D.grid_y[::x_step]
    zGrid = gridInterp3D.grid_z[::x_step]
    grid3D = grid3D[::x_step]

    cylDat = []
    for x_i, x_plane in enumerate(grid3D):
        xx = xGrid[x_i]
        assert np.all(xx == xx[0,0])
        yy = yGrid[x_i]
        zz = zGrid[x_i]

        R = np.sqrt(yy**2+zz**2)
        R = R.round(decimals=10)
        R_unq = np.unique(R)

        for r in R_unq:
            bool_mat = (R == r)
            radial_avg = grid3D[x_i][bool_mat].mean()
            # print('num. points used for calc.:', grid3D[x_i][bool_mat].shape[0])
            cylDat.append([xx[0,0], r, radial_avg])

    cylDat = np.array(cylDat)
    dat = cylDat[:,2]
    cylCoor = np.delete(cylDat, 2, axis=1)

    # gridInterp2D boundary warning system
    rmin, rmax = min(cylCoor[:,1]), max(cylCoor[:,1])
    xmin, xmax = min(cylCoor[:,0]), max(cylCoor[:,0])
    if gridInterp2D.ymin < rmin:
        warnings.warn(f'gridInterp2D.ymin < rmin : {gridInterp2D.ymin} < {rmin}')
    if gridInterp2D.ymax > rmax:
        warnings.warn(f'gridInterp2D.ymax > rmax : {gridInterp2D.ymax} > {rmax}')
    if gridInterp2D.xmin < xmin:
        warnings.warn(f'gridInterp2D.xmin < xmin : {gridInterp2D.xmin} > {xmin}')
    if gridInterp2D.xmax > xmax:
        warnings.warn(f'gridInterp2D.xmax > xmax : {gridInterp2D.xmax} > {xmax}')
    # interoplate cylinderical coordinates onto universal grid
    radAvg_grid = gridInterp2D.getInterpGrid(cylCoor, dat)
    # gridInterp2D.plotGrid(radAvg_grid, 'Grid-rad_avg')
    return radAvg_grid



class GridInterp2D:
    def __init__(self, y2DumpPrefix, xmin, xmax, ymin, ymax,
                 t_begin, t_end, t_resol, x_resol = 100j):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        # new grid to interpolate onto
        self.x_resol = x_resol
        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)
        y_resol = x_resol*(dy/dx)
        self.grid_x, self.grid_y = np.mgrid[xmin:xmax:x_resol,
                                            ymin:ymax:y_resol]
        self.y2DumpPrefix = y2DumpPrefix
        self.t_begin = t_begin
        self.t_end = t_end
        self.t_resol = t_resol

    def getInterpGrid(self, coor, val): # vals):
        print('     2D Interoplating...')
        start = time.time()
        # only linear interpolation available for 3D domains
        try:
            grid = griddata(coor, val, (self.grid_x, self.grid_y),
                            method='cubic')
        except ValueError as err:
            print(err)
            grid = []
        # make NaN values zeros
        grid[np.isnan(grid)] = 0
        end = time.time()
        print('          interpolation time: %d seconds' %(end - start))
        return grid

    def getCGNSData(self, cgnsPath, datID): # datIDs):
        print(f'     Getting CGNS data... {cgnsPath} {datID}')
        try:
            with h5py.File(cgnsPath, 'r') as f:
                xCoor = f['Base']['Zone']['GridCoordinates']['CoordinateX'][' data'][:]
                yCoor = f['Base']['Zone']['GridCoordinates']['CoordinateY'][' data'][:]

                dat = f['Base']['Zone']['FlowSolution.N:1'][datID][' data'][:]
                # dat = []
                # for datID in datIDs:
                #     dat.append(f['Base']['Zone']['FlowSolution.N:1'][datID][' data'][:]) #'VelocityMagnitude'
            coor = np.array([xCoor, yCoor]).T
            # print('CGNS coor.shape', coor.shape)
            return coor, dat
        except FileNotFoundError as err:
            print(f'\t\t{err}')
            return None, None

    def getY2SolnPaths(self, y2DatDir):
        '''
        Same as 3D no need to overwrite if class inheritence in place.
        Returns
        -------
        solnPaths : <list>
            Each item structured as [solnDatPath, solnMeshPath] evenly spaced
            through time based on self.t_start, self.t_end and self.t_resol.
        '''
        # sort data files to find latest mesh and solution files
        ents = os.listdir(y2DatDir)
        ents.sort()

        t_begin_str = self.getTimeStr(self.t_begin)
        t_end_str = self.getTimeStr(self.t_end)
        # t_begin_str = self.getTimeStr(t_begin)
        # t_end_str = self.getTimeStr(t_end)
        i_init = ents.index(f'{self.y2DumpPrefix}{t_begin_str}.xmf')
        i_final = ents.index(f'{self.y2DumpPrefix}{t_end_str}.xmf')
        ents_bnd = ents[i_init:i_final]
        # print(f'Interpreting from {ents[i_init]} to {ents[i_final]}')

        # store latest mesh from dump before t_begin
        for ent in ents[:i_init]:
            if ent.endswith('.mesh.h5'):
                latestMesh = ent

        solnFiles = []
        for ent in ents_bnd:
            if ent.endswith('.mesh.h5'):
                latestMesh = ent
            if ent.endswith('.sol.h5'):
                latestSoln = ent
                solnFiles.append([latestSoln, latestMesh])

        n_solns = len(solnFiles)
        # use t_resol to reduce number of solution files
        if self.t_resol > n_solns:
            warnings.warn(f't_resol too high, make t_resol <= {n_solns}')
        t_indices = np.linspace(0, n_solns-1, self.t_resol)
        t_indices = [int(t_i) for t_i in t_indices]
        solnFiles = [solnFiles[i] for i in t_indices]
        solnPaths = [[os.path.join(y2DatDir, solnFile[0]), os.path.join(y2DatDir, solnFile[1])]
                      for solnFile in solnFiles]
        return solnPaths

    # def getY2Grids(self, y2DatDir, datID): #, t_resol):
    #     '''
    #     Same as 3D no need to overwrite if class inheritence in place
    #     '''
    #     solnPaths = self.getY2SolnPaths(y2DatDir)
    #     # print(f'Interpreting grid with solution files: {solnFiles}')
    #     grids = []
    #     for soln in solnPaths:
    #         solnDatPath = soln[0]
    #         solnMeshPath = soln[1]
    #         grid, t = self.getY2Grid(solnDatPath, solnMeshPath, datID)
    #         grids.append([grid, t])

    #     grids = np.array(grids, dtype=object)
    #     return grids

    def getY2Data(self, solnPaths, datID):
        '''
        Requires overwrite for 3D
        '''
        print(f'     Getting YALES2 data... {datID}')
        print(f'          {solnPaths}')
        coor = []
        dat = []
        t = []
        for soln in solnPaths:
            solnDatPath = soln[0]
            solnMeshPath = soln[1]
            with h5py.File(solnDatPath, 'r') as f:
                dat.append(f['Data'][datID][:])
                # t = f['Data']['TIME'][:]
                t.append(f['Data']['TOTAL_TIME'][:])
            with h5py.File(solnMeshPath, 'r') as f:
                coor.append(f['Coordinates']['XYZ'][:][:, :2])
        # coor = np.array(coor, dtype=object) # must be dtype=object cause grid could be different sizes when AMR is used
        # dat = np.array(dat)
        # t = np.array(t)
        return coor, dat, t


    # def getY2Data(self, solnDatPath, solnMeshPath, datID):
    #     '''
    #     Requires overwrite for 3D
    #     '''
    #     print(f'Interoplating... {solnDatPath}, {solnMeshPath}')
    #     with h5py.File(solnDatPath, 'r') as f:
    #         dat = f['Data'][datID][:]
    #         # t = f['Data']['TIME'][:]
    #         t = f['Data']['TOTAL_TIME'][:]
    #     with h5py.File(solnMeshPath, 'r') as f:
    #         coor = f['Coordinates']['XYZ'][:][:, :2]
    #     return coor, dat, t

    def getY2Grids(self, coor, dat, t):
        # coor, dat, t = self.getY2Data(solnPaths, datID)
        assert len(coor) == len(dat) == len(t)
        y2Grids = []
        for t_i in range(len(t)):
            val = dat[t_i]
            coord = coor[t_i]
            y2Grids.append(self.getInterpGrid(coord, val))
        return y2Grids, t
    # def getY2Grid(self, solnDatPath, solnMeshPath, datID):
    #     coor, dat, t = self.getY2Data(solnDatPath, solnMeshPath, datID)
    #     y2Grid = getInterpGrid(coor,dat)
    #     return y2Grid, t

    def getGridsMeanDiff(self, grids1, grids2):
        '''
        same as 3D
        '''
        mean_diffs = []
        for i, grid1 in enumerate(grids1):
            grid2 = grids2[i]
            t1 = grid1[1][0]
            t2 = grid2[1][0]
            if t1 != t2:
                t_diff = abs(t1-t2)
                if t_diff > 1e-10:
                    print(f'GRID TIMES DO NOT MATCH: {t1}, {t2} off by {t_diff} seconds')
                # warnings.warn(f'GRID TIMES DO NOT MATCH: off by {t_diff} seconds')
                # raise Exception('GRID COMPARISON FAILED: grid times do not match')
            grid2 = grid2[0]
            grid1 = grid1[0]
            mean_diff = np.mean(abs(grid1 - grid2))
            mean_diffs.append([mean_diff, t1])
        mean_diff_all = np.mean(mean_diffs)
        mean_diffs = np.array(mean_diffs)
        return mean_diff_all

    def getTimeStr(self, t):
        t_str = str(int(t*1)) # need attribute or parameter to adjust
        t_str = t_str.zfill(6)
        return t_str

    def plot2DGrid(self, grid, title):
        plt.imshow(grid.T, extent=(self.xmin, self.xmax, self.ymin, self.ymax), origin='lower')
        plt.colorbar()
        plt.savefig(f'interp_grid-{title.replace(" ", "_")}.png')
        plt.clf()


class GridInterp3D(GridInterp2D):
    def __init__(self, y2DumpPrefix, xmin, xmax, ymin, ymax, zmin, zmax,
                 t_begin, t_end, t_resol, x_resol = 100j):
        super().__init__(y2DumpPrefix, xmin, xmax, ymin, ymax,
                         t_begin, t_end, t_resol, x_resol = x_resol)
        # self.xmin = xmin
        # self.xmax = xmax
        # self.ymin = ymin
        # self.ymax = ymax
        ## new attributes
        self.zmin = zmin
        self.zmax = zmax
        # new grid to interpolate onto
        ## overwite GridInterp2D attributes
        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)
        dz = abs(zmax - zmin)
        y_resol = self.x_resol*(dy/dx)
        z_resol = self.x_resol*(dz/dx)
        self.grid_x, self.grid_y, self.grid_z = np.mgrid[xmin:xmax:x_resol,
                                                         ymin:ymax:y_resol,
                                                         zmin:zmax:z_resol]
        # self.y2DumpPrefix = y2DumpPrefix
        # self.t_begin = t_begin
        # self.t_end = t_end
        # self.t_resol = t_resols

    def getInterpGrid(self, coor, val): # vals):
        '''
        Overwite of GridInterp2D.getInterpGrid()
        '''
        print('     3D Interoplating...')
        start = time.time()
        # only linear interpolation available for 3D domains
        grid = griddata(coor, val, (self.grid_x, self.grid_y, self.grid_z),
                        method='linear')
        # make NaN values zeros
        grid[np.isnan(grid)] = 0
        end = time.time()
        print('          interpolation time: %d seconds' %(end - start))
        return grid

    def getCGNSData(cgnsPath, datID): # datIDs):
        '''
        Overwite of GridInterp2D.getCGNSData()
        '''
        print(f'     Getting CGNS data... {cgnsPath} {datID}')
        with h5py.File(cgnsPath, 'r') as f:
            xCoor = f['Base']['Zone']['GridCoordinates']['CoordinateX'][' data'][:]
            yCoor = f['Base']['Zone']['GridCoordinates']['CoordinateY'][' data'][:]
            zCoor = f['Base']['Zone']['GridCoordinates']['CoordinateZ'][' data'][:]

            dat = f['Base']['Zone']['FlowSolution.N:1'][datID][' data'][:]
            # dat = []
            # for datID in datIDs:
            #     dat.append(f['Base']['Zone']['FlowSolution.N:1'][datID][' data'][:]) #'VelocityMagnitude'
        coor = np.array([xCoor, yCoor, zCoor]).T
        return coor, dat

    # def getY2Data(self, solnDatPath, solnMeshPath, datID):
        # '''
        # Overwrite of GridInterp2D.getY2Data()

        # Parameters
        # ----------
        # solnDatPath : str
        #     system path to yales2 solution data file.
        # solnMeshPath : str
        #     system path to yales2 solution mesh file that accompanies solution data file.
        # datID : str
        #     ID used in yales2 HDF5 solution data file.

        # Returns
        # -------
        # coor : <numpy.array>
        # dat : <numpy.array>
        # t : <numpy.array>
        # '''
    #     with h5py.File(os.path.join(y2DatDir, solnDat), 'r') as f:
    #         dat = f['Data'][datID][:]
    #         # t = f['Data']['TIME'][:]
    #         t = f['Data']['TOTAL_TIME'][:]
    #     with h5py.File(os.path.join(y2DatDir, solnMesh), 'r') as f:
    #         coor = f['Coordinates']['XYZ'][:][:, :3]
    #     return coor, dat, t
    def getY2Data(self, solnPaths, datID):
        '''
        Overwrite of GridInterp2D.getY2Data()

        Parameters
        ----------
        solnDatPaths : <list>
            Each item structured as [solnDatPath, solnMeshPath] evenly spaced
            through time based on self.t_start, self.t_end and self.t_resol.
        datID : str
            ID used in yales2 HDF5 solution data file.

        Returns
        -------
        coor : <list>
        dat : <list>
        t : <list>
        '''
        coor = []
        dat = []
        t = []
        for soln in solnPaths:
            solnDatPath = soln[0]
            solnMeshPath = soln[1]
            with h5py.File(solnDatPath, 'r') as f:
                dat.append(f['Data'][datID][:])
                # t = f['Data']['TIME'][:]
                t.append(f['Data']['TOTAL_TIME'][:])
            with h5py.File(solnMeshPath, 'r') as f:
                coor.append(f['Coordinates']['XYZ'][:][:, :3])
        return coor, dat, t

    def plot3DGrid(self, grid, title):
        xmid = int(grid.shape[0]/2)
        ymid = int(grid.shape[1]/2)
        zmid = int(grid.shape[2]/2)
        self.plot2DGrid(grid[xmid,:,:], f'{title}-mid_yz')
        self.plot2DGrid(grid[:,ymid,:], f'{title}-mid_xz')
        self.plot2DGrid(grid[:,:,zmid], f'{title}-mid_xy')
    # def meanDiff(self, grids1, grids2):
    #     mean_diffs = []
    #     for i, grid1 in enumerate(grids1):
    #         grid2 = grids2[i]
    #         t1 = grid1[1][0]
    #         t2 = grid2[1][0]
    #         if t1 != t2:
    #             t_diff = abs(t1-t2)
    #             if t_diff > 1e-10:
    #                 print(f'GRID TIMES DO NOT MATCH: {t1}, {t2} off by {t_diff} seconds')
    #             # warnings.warn(f'GRID TIMES DO NOT MATCH: off by {t_diff} seconds')
    #             # raise Exception('GRID COMPARISON FAILED: grid times do not match')
    #         grid2 = grid2[0]
    #         grid1 = grid1[0]
    #         mean_diff = np.mean(abs(grid1 - grid2))
    #         mean_diffs.append([mean_diff, t1])
    #     mean_diff_all = np.mean(mean_diffs)
    #     mean_diffs = np.array(mean_diffs)
    #     return mean_diff_all

    # def getTimeStr(self, t):
    #     t_str = str(int(t*1))
    #     t_str = t_str.zfill(6)
    #     return t_str

    # def plotGrid(self, grid, title):
    #     plt.imshow(grid.T) #, extent=(self.xmin, self.xmax, self.ymin, self.ymax), origin='lower')
    #     plt.colorbar()
    #     plt.savefig(f'interp_grid-{title.replace(" ", "_")}.png')
    #     plt.clf()

# import time
# #### TEST #####
# xmin = -0.5
# xmax = 0.5
# ymin = -0.5
# ymax = 1.5

# t_begin = 0.5
# t_end = 1
# t_resol = 2 # evaluate t_resol time steps

# gridInterp = GridInterp('droplet_convection',
#                         xmin, xmax, ymin, ymax,
#                         t_begin, t_end, t_resol
#                         )
# #################################
# ####### HQ DATA EXTRACT #########
# caseDir = '.'
# start = time.time()
# hq_grids = gridInterp.getGrids(caseDir) #, t_begin, t_end)
# print('hq_grids calc time: ', time.time()-start)
# ###################################
# ####### AMR DATA EXTRACT ##########
# caseDir = '../../../base_case'
# # initSolnFile = 'droplet_convection.sol000000_1.sol.h5'
# # with h5py.File(os.path.join(caseDir, 'dump', initSolnFile), 'r') as f:
# #     t_init = f['Data']['TOTAL_TIME'][:]
# start = time.time()
# amr_grids = gridInterp.getGrids(caseDir) #, t_begin, t_end)
# print('amr_grids calc time: ', time.time()-start)

# # print(amr_grids)
# ########################################
# ######## MEAN DIFFERENCE CALC ##########
# print('amr first time step: ', amr_grids[0,1])
# print('hq first time step: ', hq_grids[0,1])
# start = time.time()
# meanDiff, meanDiffs = gridInterp.meanDiff(hq_grids, amr_grids)
# print('meanDiff calc time: ', time.time()-start)

# ###################################
# ######### PLOTS ###################
# gridInterp.plotGrid(amr_grids[-1, 0], 'AMR Simulation')
# gridInterp.plotGrid(hq_grids[-1, 0], 'High Quality Simulation')
# gridInterp.plotGrid(hq_grids[-1, 0] - amr_grids[-1, 0], 'HQ-AMR Differrence')
# gridInterp.plotGrid(abs(hq_grids[-1, 0] - amr_grids[-1, 0]), 'HQ-AMR Absolute Difference')
# # gridInterp plotGrid(abs(hq_grids[-1] - amr_grids[-1]), 'HQ-AMR Absolute Difference')
# print('mean differences through sim. time: ', meanDiffs)

# plt.plot(meanDiffs[:, 1], meanDiffs[:, 0])
# plt.title('Mean Difference between HQ and AMR Simulations')
# plt.xlabel('Sim. Time')
# plt.ylabel('Mean Difference')
# plt.savefig('mean_diff-vs-sim_time.png')
