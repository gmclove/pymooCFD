import os
import numpy as np
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import time

from pymooCFD.setupOpt import dataDir, preProcDir, client, \
                                studyDir, meshSF, baseCaseMS
from pymooCFD.util.handleData import loadCP, archive, findKeywordLine, loadTxt
from pymooCFD.util.sysTools import removeDir, makeDir
from pymooCFD.util.gridInterp import GridInterp

# import os

# from pymoo.optimize import minimize

# def runGen1(restart=True, remove_prev=False):
#     if restart:
#         checkpointFile = os.path.join(dataDir, 'checkpoint-gen1.npy')
#         algorithm = loadCP(checkpointFile=checkpointFile, hasTerminated=True)
#         print("Loaded Checkpoint:", algorithm)
#         print(f'Last checkpoint at generation {algorithm.n_gen}')
#         setAlgorithm(algorithm)
#     else:
#         from pymooCFD.setupOpt import algorithm
#         if remove_prev:
#             # remove previous runs data directory
#             removeDir(dataDir)
#         else:
#             # archive previous runs data directory
#             archive(dataDir)
#     ###########################################################################
#     ######    RUN GENERATION 1   #######
#     from pymooCFD.setupOpt import problem, callback, display
#     from dask.distributed import Client
    
#     client = Client(n_workers = n_workers)
#     problem = problem(client)
    
#     res = minimize(problem,
#                    algorithm,
#                    termination=('n_gen', 1),
#                    callback=callback,
#                    seed=1,
#                    copy_algorithm=False,
#                    # pf=problem.pareto_front(use_cache=False),
#                    save_history=True,
#                    display=display,
#                    verbose=True
#                    )

#     # np.save("checkpoint", algorithm)
#     print("EXEC TIME: %.3f seconds" % res.exec_time)
    
    
    
def runCornerCases(xl, xu):  # alg):
    '''
    Finds binary permutations of limits and runs these cases. 
    
    runCornerCases(alg)
    -------------------
    Running these simulations during the optimization studies pre-processing 
    provides insight into what is happening at the most extreme limits of the 
    parameter space. 
    
    These simulations may provide insight into whether the user should expand 
    or restrict the parameter space. They may reveal that the model becomes 
    non-physical in these extreme cases. 
    
    Exploring the entire parameter space becomes important expecially in
    optimization studies with geometric parameters. This is due to the 
    automated meshing introducting more variablility into the CFD case 
    workflow. 
    '''

    # xl = alg.problem.xl
    # xu = alg.problem.xu
    
    from itertools import product
    n_var = len(xl)  # alg.pop.get('X'))
    n_perm = 2**n_var
    # find every binary permutations of length n_var
    bin_perms = [list(i) for i in product([0, 1], repeat=n_var)]
    lims = np.column_stack((xl, xu))
    # limit permutations
    lim_perms = np.zeros((n_perm, n_var))
    for perm_i, bin_perm in enumerate(bin_perms):
        for var, lim_i in enumerate(bin_perm):
            lim_perms[perm_i, var] = lims[var][lim_i]
            
    print('Limit Permutations: ')
    print(lim_perms)
    
    
    # dirs = []
    # for i in range(len(lim_perms)):
    #     dirs.append(os.path.join(preProcDir, 'corner-cases', f'corner-{i}'))
    ccDir = os.path.join(preProcDir, 'runCC')
    makeDir(ccDir)

    from pymooCFD.setupCFD import runPop
    runPop(ccDir, 'cc', lim_perms, 'local')
    

        
###############################################################################
# from pymooCFD.setupOpt import client 
# from paraview.simple import *

def meshStudy(restart=True):
    # overwrite meshSF
    # meshSF = [0.25, 0.5, 1, 1.5, 2.0, 2.5, 3.0]
    if not restart:
        for sf in meshSF:
            removeDir(os.path.join(studyDir, f'meshSF-{sf}'))
            
    # execute using dask.distribute 
    def fun(sf):
        caseDir = os.path.join(studyDir, f'meshSF-{sf}') 
        f = meshStudyCase(caseDir, sf)
        return f
    jobs = [client.submit(fun, sf) for sf in meshSF]
    jobs.wait()
    # obj = np.row_stack([job.result() for job in jobs])

def meshStudyPlot(coor):
    ##### PLOT #####
    # vMax_perErr = abs((cart_vMaxCoor - amr_vMaxCoor) / cart_vMaxCoor) * 100
    plt.plot(suptitle='Mesh Sensitivity Study', title='Mean Difference in Droplet Boundary Coordinates')
    for i in range(1,len(coor)):
        mean_diff = np.mean(abs(coor[i]-coor[i-1]))
        # n_elem = round(nx*sf) * round(ny*sf)
        plt.add(mean_diff, n_elem[i])
    
    plt.savefig(os.path.join(studyDir, 'mesh-study-plot.png'))


def meshStudyCase(caseDir, sf):
    if meshStudyCompleted(caseDir, sf):
        return 
    else:
        n_elem = meshStudyPreProc(caseDir, sf)
        cmd = cmd = f'cd {caseDir} && mpirun {solverExec} > output.dat'
        os.system(cmd)
        coor = meshStudyPostProc(caseDir)   
    
    
def meshStudyPreProc(caseDir, sf):
    inputFile = 'droplet_convection.in'
    inputPath = os.path.join(baseCaseMS, inputFile)
    
    with open(inputPath, 'r') as f:
        in_lines = f.readlines()
        
    nx_new = round(nx*sf)
    nx_newLine = nx_line[:nx_line.find('=')+2] + str(nx_new) + '\n'
    in_lines[nx_line_i] = nx_newLine

    ny_new = round(ny*sf)
    ny_newLine = ny_line[:ny_line.find('=')+2] + str(ny_new) + '\n'
    in_lines[ny_line_i] = ny_newLine

    sfDir = os.path.join(studyDir, f'meshSF-{sf}')
    copy_tree(baseCaseMS, caseDir)
    with open(os.path.join(caseDir, inputFile), 'w') as f_new:
        f_new.writelines(in_lines)
        
    n_elem = nx_new*ny_new
    return n_elem
    
    
def meshStudyPostProc(caseDir):
     # trace generated using paraview version 5.9.0
    ######## Objective 2: Match Shape and Location of Droplet #########
    timestepFile = 'droplet_convection.sol000400_1.xmf'
    outDatPath = os.path.join(caseDir, 'ls_phi-0.5.txt')
    inDatPath = os.path.join(caseDir, 'dump', timestepFile)
    # trace generated using paraview version 5.9.0
    
    # PARAVIEW SCRIPT
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    
    # create a new 'Xdmf3ReaderS'
    droplet_convectionsol000400_1xmf = Xdmf3ReaderS(
        registrationName=timestepFile, 
        FileName=[inDatPath]
        )
    
    # Properties modified on droplet_convectionsol000400_1xmf
    droplet_convectionsol000400_1xmf.PointArrays = ['LS_PHI']
    droplet_convectionsol000400_1xmf.CellArrays = []
    
    UpdatePipeline(time=0.0, proxy=droplet_convectionsol000400_1xmf)
    
    # create a new 'Contour'
    contour1 = Contour(registrationName='Contour1', Input=droplet_convectionsol000400_1xmf)
    
    UpdatePipeline(time=0.0, proxy=contour1)
    
    # save data
    SaveData(outDatPath, proxy=contour1, ChooseArraysToWrite=1,
        PointDataArrays=['LS_PHI'])

    coor = np.loadtxt(outDatPath)
    x = coor[:,0]
    y = coor[:,1]
    
    from scipy.interpolate import interp1d
    interp = interp1d(x, y, kind='nearest')
    n_pts = 1000
    x_new = np.linspace(min(x),max(x), n_pts)
    y_new = interp(x_new)
    
    coor_new = np.column_stack((x_new, y_new))
    np.savetxt(os.path.join(caseDir, 'interp_drop_coor.txt'), coor_new)
    
    return coor_new


def meshStudyCompleted(caseDir, dat):
    ###### Check Completion ######
    # global varFile, objFile
    # load in previous variable file if it exist and
    # check if it is equal to current variables
    compFile = os.path.join(caseDir, 'interp_drop_coor.txt')
    if os.path.exists(compFile):
        # try:
        #     prev_dat = np.loadtxt(compFile)
        #     if np.array_equal(prev_dat, dat):
        #     print(f'{caseDir} already complete')
        #     return True
        # except OSError as err:
        #     print(err)
        #     return False
        print(f'{caseDir} already complete')
        return True
    else:
        return False


def interpHighFidelitySim(caseDir):
    global gridInterp
    dom_xmin = -0.5
    dom_xmax = 0.5
    dom_ymin = -0.5
    dom_ymax = 1.5
    
    t_begin = 0.5
    t_end = 1
    t_resol = 2 # evaluate t_resol time steps
    
    gridInterp = GridInterp('droplet_convection', 
                            dom_xmin, dom_xmax, dom_ymin, dom_ymax,
                            t_begin, t_end, t_resol
                            )
    ####### HQ DATA EXTRACT #########
    start = time.time()
    hq_grids = gridInterp.getGrids(caseDir, t_begin, t_end)
    print('high quality simulation interpolation time: ', time.time()-start)
        
    np.save('hq_grids', hq_grids)
    return hq_grids
        
# def mapGen0():
#     preProcDir = os.path.join(plotsDir, 'preProc')
#     try:
#         os.mkdir(preProcDir)
#     except OSError as err:
#         print(err)
#         print(f'{preProcDir} already exists')
#     ########################################################################################################################
#     algorithm = loadCP()
#     X = algorithm.pop.get('X')
#     F = algorithm.pop.get('F')
#     ########################################################################################################################
#     print('VARS')
#     print(X)
#     print('OBJ')
#     print(F)
#     ########################################################################################################################
#     ##### SCATTER PLOTS #######
#     ###########################
#     X = np.array(X)
#     F = np.array(F)
#     from pymoo.visualization.scatter import Scatter
#     # https://pymoo.org/visualization/scatter.html
#     ##### Function Space ######
#     f_space = Scatter(title = 'Objective Space',
#                         labels = obj_labels)
#     f_space.add(F)
#     # if pf is not None:
#     #     f_space.add(pf)
#     f_space.save(f'{preProcDir}/obj-space.png')
#     ##### Variable Space ######
#     f_space = Scatter(title = 'Design Space',
#                         labels = obj_labels)
#     f_space.add(X)
#     # if pf is not None:
#     #     f_space.add(pf)
#     f_space.save(f'{preProcDir}/var-space.png')

#     ##### Variable vs. Objective Plots ######
#     # extract objectives and variables columns and plot them against each other
#     for x_i, x in enumerate(X.transpose()):
#         for f_i, f in enumerate(F.transpose()):
#             plot = Scatter(title=f'{var_labels[x_i]} vs. {obj_labels[f_i]}',
#                             labels=[var_labels[x_i], obj_labels[x_i]]
#                             )
#             xy = np.column_stack((x,f))
#             plot.add(xy)
#             plot.save(f'{preProcDir}/{var_labels[x_i].replace(" ", "_")}-vs-{obj_labels[f_i].replace(" ", "_")}.png')

#     # if there are more than 2 objectives create array of scatter plots comparing
#     # the trade-off between 2 objectives at a time
#     if len(F.transpose()) > 2:
#         ####### Pair Wise Objective Plots #######
#         # Pairwise Scatter Plots of Function Space
#         plot = Scatter(tight_layout=True)
#         plot.add(F, s=10)
#         plot.add(F[-1], s=30, color="red")
#         plot.save(f'{preProcDir}/pairwise-scatter.png')

# if __name__ == "__main__":
#     runGen0()
#     mapGen0()
