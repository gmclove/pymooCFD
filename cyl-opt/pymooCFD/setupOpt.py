import os
import numpy as np
from pymooCFD.util.sysTools import makeDir


#####################################
#### Genetic Algorithm Criteria #####
#####################################
n_gen = 50
pop_size = 100
n_offsprings = int(pop_size * (1 / 2)) # = number of evaluations each generation
#####################################
####### Define Design Space #########
#####################################
n_var = 2
var_labels = ['Amplitude', 'Frequency']
# use boolean to indicate if re-meshing is necessary because parameter is geomtric
# geoVarsI =   [False, False, False]  # , False]
varType =    ["real", "real"]  # options: 'int' or 'real'
xl =         [0.1, 0.1]  # lower limits of parameters/variables
xu =         [3.0, 1]  # upper limits of variables
if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
    raise Exception("Design Space Definition Incorrect")

#######################################
####### Define Objective Space ########
#######################################
obj_labels = ['Drag on Cylinder', 'Power Input']
n_obj = 2
n_constr = 0
### NORMALIZATION SCHEME
## Define values used to normalize objectives
## often these values are explored in optimization pre-processing
def normalize(obj):
    obj_max = [1.5, 2212.444544581198] # maximum possible value
    # utopia point (ideal value), aspiration point, target value, or goal
    obj_o = [0.5, 0.24582717162013315]
    # for loop through each individual
    obj_norm = []
    for obj_ind in obj:
        # individual objective(s) normalization
        obj_norm_ind = np.subtract(obj_ind, obj_o)/np.subtract(obj_max, obj_o)
        obj_norm.append(obj_norm_ind)
    return obj_norm

#####################################
##### Define Mesh Parameters ########
#####################################
# NOTE: only used in optimization studies with geometric parameters
# Generation 0 (random sampling of design space) mesh convergence study
# meshSizeMin = 0.05
# meshSizeMax = 0.5
# nMeshes = 5
# meshSizes = np.linspace(meshSizeMin, meshSizeMax, nMeshes)

####################################
####### Define Data Handling #######
####################################
archDir = 'archive'
os.makedirs(archDir, exist_ok=True)
nCP = 10  # number of generations between extra checkpoints
optDatDir = 'opt_run'
os.makedirs(optDatDir, exist_ok=True)
checkpointPath = os.path.join(optDatDir, 'checkpoint.npy')

############################################
###### Define CFD Pre/Post Processing ######
############################################
inputFile = '2D_cylinder.in'
baseCaseDir = 'base_case'
os.makedirs(baseCaseDir, exist_ok=True)
#hqSimDatPath = 'hq_sim_coor.txt'

#####################################################
###### Define Optimization Pre/Post Processing ######
#####################################################
procOptDir = 'procOpt'
os.makedirs(procOptDir, exist_ok=True)
### Plots Directory
plotDir = os.path.join(procOptDir, 'plots')
os.makedirs(plotDir, exist_ok=True)
### Mapping Objectives vs. Variables Directory
mapDir = os.path.join(plotDir, 'mapGen')
os.makedirs(mapDir, exist_ok=True)
#### Mesh Sensitivity Study ####
studyDir = os.path.join(procOptDir, 'meshStudy')
os.makedirs(studyDir, exist_ok=True)
# baseCaseMS = os.path.join(studyDir, 'cartMeshCase')
# copy_tree(baseCaseDir, baseCaseMS)
# mesh size factors 
meshSF = [0.25, 0.5, 1, 1.5, 2.0, 2.5, 3.0]   # np.linspace(0.5,4,8) #[0.5, 1, 1.5, 1.75, 2, 2.25, 2.5]

###################################################
#      High Quality Simulation Interpolation      #
###################################################
### Define grid interpolation parameters and perform interpolation on high 
## quality simulation. These results can be compared to lower quality 
## simulations on a universal grid. 
# hqSim_dir = 'hq_sim'
# hqSim_y2DatDir = os.path.join(hqSim_dir, 'dump')
# from pymooCFD.util.gridInterp import GridInterp3D, GridInterp2D, radialAvg
# y2DumpPrefix = 'pipe_expansion.sol'
# xmin, xmax = 1.0, 2.0
# ymin, ymax = -0.5, 0.5
# zmin, zmax = -0.5, 0.5
# t_begin, t_end = 80, 100
# t_resol = t_end - t_begin # highest quality
# gridInterp3D = GridInterp3D(y2DumpPrefix, xmin, xmax, ymin, ymax, zmin, zmax,
#                             t_begin, t_end, t_resol, 
#                             x_resol = 200j)
## SPECIAL CASE: Radial Averaging
## initialize 2D gridInterp object for interpolating onto after getting
## radial average
# gridInterp2D = GridInterp2D(gridInterp3D.y2DumpPrefix, 
#                             gridInterp3D.xmin, gridInterp3D.xmax, 
#                             0.01, gridInterp3D.ymax, 
#                             gridInterp3D.t_begin, gridInterp3D.t_end, 
#                             gridInterp3D.t_resol, 
#                             x_resol = gridInterp3D.x_resol)
## stop expensive spatiotemporal interoplation from being performed each time 
## by checking if path exists already
# hqGrid_uMag_path = os.path.join(hqSim_dir, 'hqGrid_uMag.npy')
# if not os.path.exists(hqGrid_uMag_path):
#     solnPaths = gridInterp3D.getY2SolnPaths(hqSim_y2DatDir)
#     coor, dat, t = gridInterp3D.getY2Data(solnPaths, 'U')
#     # manipulate YALES2 data
#     uMag = []
#     for U in dat:
#         uMag_t = np.linalg.norm(U, axis=1)
#         uMag.append(uMag_t)
#     # plt.scatter(coor, uMag)
#     ### time average
#     ## if mesh size does not change 
#     if all(coor_t.shape == coor[0].shape for coor_t in coor):
#         ## time average data then get single grid. Faster and more accurate
#         ## but only works when mesh is consistent aka no AMR.
#         uMag = np.array(uMag)
#         uMag_tAvg = np.mean(uMag, axis=0, dtype=np.float64)
#         hqGrid_uMag_tAvg = gridInterp3D.getInterpGrid(coor[0], uMag_tAvg)
#     else:
#         ## get grids then time average
#         y2Grids_uMag, t = gridInterp3D.getY2Grids(coor, uMag, t)
#         # time average
#         hqGrid_uMag_tAvg = np.mean(y2Grids_uMag, dtype=np.float64)
#     hqGrid_uMag = hqGrid_uMag_tAvg
#     gridInterp3D.plot3DGrid(hqGrid_uMag, 'hqGrid_uMag')
#     np.save('hq_sim/hqGrid_uMag_3D', hqGrid_uMag)
#     # radial average
#     hqGrid_uMag = radialAvg(hqGrid_uMag, gridInterp3D, gridInterp2D)
#     gridInterp2D.plot2DGrid(hqGrid_uMag, 'hqGrid_uMag_radAvg')
#     #save binary 
#     np.save(hqGrid_uMag_path, hqGrid_uMag)
# else:
#     hqGrid_uMag = np.load(hqGrid_uMag_path)

# hqGrid_phi_path = os.path.join(hqSim_dir, 'hqGrid_phi.npy')
# if not os.path.exists(hqGrid_phi_path):
#     solnPaths = gridInterp3D.getY2SolnPaths(hqSim_y2DatDir)
#     coor, phi, t = gridInterp3D.getY2Data(solnPaths, 'PHI')
#     ### time average
#     ## if mesh size does not change 
#     if all(coor_t.shape == coor[0].shape for coor_t in coor):
#         ## Time average data then get single grid. Faster and more accurate 
#         ## but only works when mesh is consistent through time aka no AMR.
#         phi = np.array(phi)
#         phi_tAvg = np.mean(phi, axis=0, dtype=np.float64)
#         hqGrid_phi_tAvg = gridInterp3D.getInterpGrid(coor[0], phi_tAvg)
#     else:
#         ## get grids then time average
#         y2Grids_phi, t = gridInterp3D.getY2Grids(coor, phi, t)
#         # time average
#         hqGrid_phi_tAvg = np.mean(y2Grids_phi, dtype=np.float64)
#     hqGrid_phi = hqGrid_phi_tAvg
#     np.save('hq_sim/hqGrid_phi_3D', hqGrid_phi)
#     gridInterp3D.plot3DGrid(hqGrid_phi, 'hqGrid_phi')
#     # radial average
#     hqGrid_phi = radialAvg(hqGrid_phi, gridInterp3D, gridInterp2D)
#     gridInterp2D.plot2DGrid(hqGrid_phi, 'hqGrid_uMag_radAvg')
#     # save binary 
#     np.save(hqGrid_phi_path, hqGrid_phi)
# else:
#     hqGrid_phi = np.load(hqGrid_phi_path)
##############################################################################
####################### PARALLEL PROCESSING ##################################

#####################################
#######  EXTERNAL CFD SOLVER  #######
#####################################
### make sure case executed propely 
def caseMonitor(caseDir):
    incomplete = False
    path = os.path.join(caseDir, 'dump')
    if not os.path.exists(path):
        incomplete = True
    return incomplete

#### SLURM JOB #####
useSlurm = True
jobFile = 'jobslurm.sh'
jobPath = os.path.join(baseCaseDir, jobFile)
jobLines = [
    '#!/bin/bash',
    '# partition using infiniband',
    '#SBATCH --partition=ib --constraint="ib&haswell_1|haswell_2|sandybridge"',
    '#SBATCH --nodes=4',
    '#SBATCH --ntasks-per-node=20',
    '#SBATCH --time=00:30:00',
    '#SBATCH --mem-per-cpu=2G',
    '#SBATCH --job-name=cyl',
    '#SBATCH --output=slurm.out'
    '',
    'module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)',
    #'module load ansys/fluent-21.2.0 ',
    'cd $SLURM_SUBMIT_DIR',
    'mpirun ./2D_cylinder'
    # 'time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out' #-pib -pinf
    ]
jobLines = "\n".join(jobLines)
with open(jobPath, 'w+') as f:
    f.writelines(jobLines)

#### Single Node Job #####
useSingleNode = False
procLim = 80  # Maximum processors to be used, defined in jobslurm.sh as well
nProc = 8 # Number of processors for each individual (EQUAL or SMALLER than procLim)
if useSingleNode:
    print('SINGLE NODE PARALLEL PROCESSING')
    print('     Number of parallel processes: ', procLim/nProc)
    print('     Number of processors for each individual: ', nProc)
    print('     Number of processors being utilized: ', procLim)

### Solver Execution Command
solverExecCmd = ['mpirun', '-np', str(nProc), '2D_cylinder']
# solverExecCmd = ['C:\"Program Files"\"Ansys Inc"\v211\fluent\ntbin\win64\fluent.exe', '2ddp', f'-t{nProc}', '-g', '-i', 'jet_rans-axi_sym.jou', '>', 'run.out']

assert useSlurm != useSingleNode
###################################
#######  PYTHON CFD SOLVER  #######
###################################
# if pop_size < 10:
#     n_workers = pop_size
# else:
#     n_workers = 1
# threads_per_worker = int(os.cpu_count() / n_workers)
# # limit threads per worker
# if threads_per_worker > 20:
#     threads_per_worker = 20

#### Option 1: Dask Computing Cluster
# HPC setup: https://docs.dask.org/en/latest/setup/hpc.html
# from dask.distributed import Client, LocalCluster, get_client
# class DaskClient(Client):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#         cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
#         # cluster.adapt(minimum = 0, maximum = 80)
#         self.client = Client(cluster)
#         # self.client = Client()
#         ### print info 
#         print('### Dask Distributed Computing Client Started Sucessfully')
#         print(self.client)
#         print('Number of Cores: ')
#         for key, value in self.client.ncores().items():
#             print(f'    {key} - {value} CPUs')
# # mpirun_nProc = threads_per_worker
# client = DaskClient

######  Option 2: MPI Pool Executor
# from mpi4py.futures import MPIPoolExecutor
# def mpiPoolExec():
#     pool = MPIPoolExecutor(max_workers = n_workers)
#     return pool
# client = mpiPoolExec


### overwrite client to None
client = None

    
'''
pyMOO SETUP
-----------
pymoo.org
'''
########################################################################################################################
######    PROBLEM    ######
from pymoo.core.problem import Problem
# from pymoo.core.problem import ElementwiseProblem

class GA_CFD(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=np.array(xl),
                         xu=np.array(xu),
                         *args,
                         **kwargs
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        gen = algorithm.callback.gen
        ###### Initialize Generation ######
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(optDatDir, f'gen{gen}')
        print('Starting ', genDir)
        #print(X)
        # create sub-directories for each individual
        indDirs = []
        for i in range(len(X)):
            indDirs.append(os.path.join(genDir, f'ind{i+1}'))
        
        ###### ARCHIVE/REMOVE PREVIOUS GENERATION DATA ######
        if gen > 2: 
            prev_genDir = os.path.join(optDatDir, f'gen{gen - 1}')
            # archive/remove generation folder to prevent build up of data
            from pymooCFD.util.handleData import removeDir  # archive
            removeDir(prev_genDir)
            # archive(genDir, archDir, background=True)

        ####################################
        ###### External CFD solver #########
        ####################################
        from pymooCFD.setupCFD import preProc, postProc
        ##### PRE-PROCCESS GENERATION #####
        if client is not None:
            ## Use client to parallelize 
            jobs = [client.submit(preProc, indDirs[i], x) for i, x in enumerate(X)]
            [job.result() for job in jobs]
        else:
            ## Use for loop 
            for i, x in enumerate(X):
                preProc(indDirs[i], x)

        ###### RUN GENERATION ######
        if useSlurm:
            from pymooCFD.execSimsBatch import slurmExec
            slurmExec(indDirs)
        if useSingleNode: 
            from pymooCFD.execSimsBatch import singleNodeExec 
            singleNodeExec(indDirs)
        ##### POST-PROCCESS GENERATION #####
        if client is not None: 
            ## Use client to parallelize 
            jobs = [client.submit(postProc, indDirs[i], x) for i, x in enumerate(X)]
            obj = np.row_stack([job.result() for job in jobs])
        else:
            ## Use for loop
            obj = np.zeros((len(X), n_obj))
            for i in range(len(X)):
                obj[i] = postProc(indDirs[i], X[i])
        
        ##################################
        ###### Python CFD solver #########
        ##################################
        # from pymooCFD.setupCFD import runCase 
        # jobs = [self.client.submit(runCase, indDirs[i], x) for i, x in enumerate(X)]
        # obj = np.row_stack([job.result() for job in jobs])
        
        # print(obj)
        # obj = normalize(obj)
        
        # make sure the setupCFD.runCFD does not return all zeros
        if not np.all(obj):
            print("ALL OBJECTIVES = 0")
            exit()
        

        out['F'] = obj
        
        # out['F'] = np.zeros((len(X), n_obj))

        print(f'GENERATION {gen} COMPLETE')
    
    # handle pickle sealization when saving algorithm object as checkpoint
    # self.client has an active network connect so it can not be serialized
    # def __getstate__(self):
    #     """Return state values to be pickled."""
    #     state = self.__dict__.copy()
    #     del state['client']
    #     return state
    
    # def __setstate__(self, state):
    #     """Restore state from the unpickled state values."""
    #     # print(state)
    #     self.__dict__.update(state)
    #     self.client = client #get_client()

problem = GA_CFD()
# TEST PROBLEM
# from pymoo.factory import get_problem
# problem = get_problem("bnh")
########################################################################################################################
######    DISPLAY    ######
from pymoo.util.display import Display

class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(n_obj):
            self.output.append(f"mean obj.{obj + 1}", np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append(f"best obj.{obj+1}", algorithm.pop.get('F')[:, obj].min())
        self.output.header()

display = MyDisplay()
########################################################################################################################
######    CALLBACK    ######
from pymoo.core.callback import Callback

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.gen = 1
        self.data['best'] = []

    def notify(self, alg):
        # save checkpoint 
        from pymooCFD.util.handleData import saveCP
        saveCP(alg)
        # increment generation
        self.gen += 1
        self.data["best"].append(alg.pop.get("F").min())
        ### For longer runs to save memory may want to use callback.data 
        ## instead of using algorithm.save_history=True which stores deep 
        ## copy of algorithm object every generation. 
        ## Example: self.data['var'].append(alg.pop.get('X'))
        ### Update global algorithm object for use within problem._evaluate()
        global algorithm 
        algorithm = alg

callback = MyCallback()
########################################################################################################################
######    OPERATORS   ######
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

sampling = MixedVariableSampling(varType, {
    "real": get_sampling("real_lhs"),  # "real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(varType, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(varType, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})
########################################################################################################################
######    TERMINATION CRITERION  ######
# https://pymoo.org/interface/termination.html
from pymoo.factory import get_termination
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
########################################################################################################################
######    ALGORITHM    ######
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

# initialize algorithm here
# will be overwritten in runOpt() if checkpoint already exists
algorithm = NSGA2(pop_size=pop_size,
                  n_offsprings=n_offsprings,
                  eliminate_duplicates=True,

                  termination = termination, 
                   
                  sampling = sampling,
                  crossover = crossover,
                  mutation = mutation,
                    
                  display = display,
                  callback = callback,                    
                  )
# setup run specific criteria 
algorithm.save_history = True
algorithm.seed = 1
algorithm.return_least_infeasible = True
algorithm.verbose = True

def setAlgorithm(alg):
    global algorithm
    algorithm = alg
    
class OptStudy: #(RunCFD):
    def __init__(self, alg = algorithm, prob = problem):
        # self.problem = problem
        self.algorithm = alg
        self.algorithm.setup(prob(client))
        self.algorithm.termination = termination
        # set algorithm global variable used by problem._evaluate()
        global algorithm
        algorithm = self.algorithm
