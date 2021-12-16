import numpy as np
import os
# import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree
import subprocess
# import glob
from scipy.integrate import quad

# from pymooCFD.genMesh import genMesh 
from pymooCFD.util.sysTools import makeDir
from pymooCFD.setupOpt import baseCaseDir, var_labels, inputFile #solverExecCmd, \
    # gridInterp2D, hqGrid_uMag, hqGrid_phi

cgnsFile = 'jet_rans-axi_sym.cgns' 

def preProc(caseDir, var):  # , jobName=jobName, jobFile=jobFile):
    """
    | preProc(caseDir, var, jobName=jobName, jobFile=jobFile)
    |
    |   CFD pre-process function. Edits CFD case input variables.
    |
    |   Parameters
    |   ----------
    |   caseDir : string
    |       CFD case directory containing input files used by CFD solver.
    |   var: list (or numpy array)
    |       List of variables typically generated my pyMOO and passed to CFD
    |       pre-process function which edits the case inputs.
    |
    |   Returns
    |   -------
    |   None
    |       Typically this function does not need to return anything because
    |       it's purpose is edit the CFD case input files.
    """
    # print(f'PRE-PROCESSING CFD CASE: {caseDir}')
    # cwd = os.getcwd()
    
    #### EXTERNAL CFD SOLVER ####
    ### Sometimes necessary to copy contents of base case into new folder 
    copy_tree(baseCaseDir, caseDir)
    ### other times simply creating an empty directory will do
    # os.makedirs(os.path.join(cwd, caseDir), exist_ok=True)
    # print(caseDir)
    ### establish paths within case directory
    dumpDir = os.path.join(caseDir, 'dump')
    inputPath = os.path.join(caseDir, inputFile)

    #### CHANGE RESTART ####
    dumpDir = os.path.join(caseDir, 'dump')
    if os.path.exists(dumpDir):
        try:
            from pymooCFD.util.yales2Tools import getLatestXMF
            latestXMF = getLatestXMF(dumpDir)
            with open(inputPath, 'r') as f:
                in_lines = f.readlines()
            kw = 'RESTART_TYPE = GMSH'
            kw_line, kw_line_i = findKeywordLine(kw, in_lines)
            in_lines[kw_line_i] = '#' + kw + '\n'
            kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
            kw_line, kw_line_i = findKeywordLine(kw, in_lines)
            in_lines[kw_line_i] = '#' + kw + '\n'
            kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
            kw_line, kw_line_i = findKeywordLine(kw, in_lines)
            in_lines[kw_line_i] = '#' + kw + '\n'
            in_lines.append('RESTART_TYPE = XMF' + '\n')
            in_lines.append('RESTART_XMF_SOLUTION = dump/' + latestXMF + '\n')
            with open(inputPath, 'w') as f:
                f.writelines(in_lines)
        except FileNotFoundError:
            print('     CHANGE RESTART FAILED: XMF file not found')
    
    ## Change YALES2 restart (only works if HDF dump is set up)
    # with open(inputFile, 'a') as f:
    #     f.write('RESTART_LOAD_ALL_DATA = TRUE # loads all the data in the HDF files')

    #### SLURM PRE-PROCESS #####
    ## when using slurm to exectute our simulations we might want to edit
    ## the file used to lauch our job
    # editJobslurm(gen, ind, caseDir)
    # editSlurmJob(caseDir, jobName=jobName, jobFile=jobFile)

    ####### EXTRACT VAR ########
    # Extract parameters for each individual
    amp = var[var_labels.index('Amplitude')]
    freq = var[var_labels.index('Frequency')]

    ####### SIMULATION INPUT PARAMETERS #########
    # open and read YALES2 input file to array of strings for each line
    with open(inputPath, 'r') as f_orig:
        in_lines = f_orig.readlines()
   
    # find line that must change using a keyword
    keyword = 'CYL_ROTATION_PROP'
    keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
    # create new string to replace line
    newLine = f'{keyword_line[:keyword_line.index("=")]}= {amp} {freq} \n'
    in_lines[keyword_line_i] = newLine
    # REPEAT FOR EACH LINE THAT MUST BE CHANGED

    with open(inputPath, 'w') as f_new:
        f_new.writelines(in_lines)
    
    ####### GENERATE MESH #########
    # meshFile = 'jet_rans-axi_sym.unv'
    # meshPath = os.path.join(cwd, caseDir, meshFile)
    # genMesh(meshPath, 1, outD)
    
def solve():
    pass

def postProc(caseDir, var):
    '''
    | postProc(caseDir, var)
    |
    |   CFD pre-process function. Edits CFD case input variables.
    |
    |   Parameters
    |   ----------
    |   caseDir : string
    |       CFD case directory containing output files used by CFD solver.
    |   var : list (or numpy array)
    |       List of variables typically generated my pyMOO and passed to CFD
    |       pre-process function which edits the case inputs. Now in
    |       post-process fuctiom this values are sometime needed to compute
    |       objectives but not always.
    |
    |   Returns
    |   -------
    |   obj : list of objectives
    |       Objectives extracted from the CFD case in post-processing
    '''
    # print(f'POST-PROCESSING CFD CASE: {caseDir}')
    # cwd = os.getcwd()

    ####### EXTRACT VAR ########
    # OPTIONAL: Extract parameters for each individual
    # sometimes variables are used in the computation of the objectives
    amp = var[var_labels.index('Amplitude')]
    freq = var[var_labels.index('Frequency')]
    
    ######## Compute Objectives ##########
    ######## Objective 1: Drag on Cylinder #########
    U = 1
    rho = 1
    D = 1
    # create string for directory of individual's data file
    dataDir = f'{caseDir}/ics_temporals.txt'
    data = np.genfromtxt(dataDir, skip_header=1)
    # try:
    #     data = np.genfromtxt(dataDir, skip_header=1)
    # except IOError as err:
    #     print(err)
    #     print('ics_temporals.txt does not exist')
    #     obj = [None] * n_obj
    #     return obj

    # collect data after 100 seconds of simulation time
    mask = np.where(data[:,1] > 100)
    # Surface integrals of Cp and Cf
    # DRAG: x-direction integrals
    # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
    p_over_rho_intgrl_1 = data[mask, 4]
    tau_intgrl_1 = data[mask, 6]
    F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
    C_drag = F_drag/((1/2)*rho*U**2*D**2)

    ######## Objective 2 #########
    # Objective 2: Power consumed by rotating cylinder
    D = 1  # [m] cylinder diameter
    t = 0.1  # [m] thickness of cylinder wall
    r_o = D/2  # [m] outer radius
    r_i = r_o-t  # [m] inner radius
    d = 2700  # [kg/m^3] density of aluminum
    L = 1  # [m] length of cylindrical tube
    V = L*np.pi*(r_o**2-r_i**2) # [m^3] volume of cylinder
    m = d*V # [kg] mass of cylinder
    I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
    P_cyc = 0.5*I*quad(lambda t : (amp*np.sin(t))**2, 0, 2*np.pi)[0]*freq  # [Watt]=[J/s] average power over 1 cycle

    obj = [C_drag, P_cyc]
    print(caseDir, ': ', obj)
    ###### SAVE VARIABLES AND OBJECTIVES TO TEXT FILES #######
    # save variables in case directory as text file after completing post-processing
    saveTxt(caseDir, 'var.txt', var)
    # save objectives in text file
    saveTxt(caseDir, 'obj.txt', obj)
    return obj


###############################################################################
###### FUNCTIONS ######
#######################
from pymooCFD.execSimsBatch import slurmExec, singleNodeExec
def runPop(popDir, subDir, X, execution):    
    dirs = preProcPop(popDir, subDir, X)
    
    if execution.lower() == 'slurm':
        slurmExec(dirs)
    else:
        singleNodeExec(dirs)

    obj = postProcPop(popDir, subDir, X)
    return obj


def preProcPop(popDir, subDir, X):
    dirs = []
    for x_i, x in enumerate(X):
        caseDir = os.path.join(popDir, f'{subDir}-{x_i}')
        dirs.append(caseDir)
        preProc(caseDir, x)
    return dirs


def postProcPop(popDir, subDir, X):
    obj = []
    for x_i, x in enumerate(X):
        caseDir = os.path.join(popDir, f'{subDir}-{x_i}')
        obj.append(postProc(caseDir, x))
    return obj


###############################################################################

def runCase(caseDir, x):
    print(f'Running Case {caseDir}: {x}')
    if completed(caseDir, x):
        obj = np.loadtxt(os.path.join(caseDir, 'obj.txt'))
        return obj
    else:
        preProc(caseDir, x)
        # avail_cpus = os.cpu_count()
        # print('Available CPUs: ', avail_cpus)
        # cmd = f'cd {caseDir} && mpirun -np {mpirun_nProc} {solverExec} > output.dat'
        # cmd = f'cd {caseDir} && mpirun {solverExec} > output.dat'
        # cmd = ['cd', caseDir, '&&', 'mpirun', '-np', str(mpirun_nProc), solverExec, '>', 'output.dat']
        # cmd = ['cd', caseDir, '&&', solverExec]
        compProc = subprocess.run(solveExecCmd, shell=True, text=True, check=True)
        print(compProc.returncode)
        print(compProc.stdout)
        print(compProc.stderr)
        obj = postProc(caseDir, x)
        return obj
    
def runGen(genDir, X):
    """
    Run an entire generation at a time. Used when working with singleNode or 
    slurmBatch modules found in the sub-package execSimsBatch. 
    """
    preProcGen(genDir, X)
    from pymooCFD.execSimsBatch.singleNode import execSims
    execSims(genDir, 'ind', len(X))
    obj = postProcGen(genDir, X)
    return obj

def preProcGen(genDir, X):
    indDirs = []
    for i, x in enumerate(X):
        indDir = os.path.join(genDir, f'ind{i + 1}')
        indDirs.append(indDir)
        # check if individual has already completed exection and postProc
        checkCompletion(x, indDir)
        # Create directory is it does not exist
        makeDir(indDir)
        preProc(indDir, x)
    return indDirs
 
def postProcGen(genDir, X):
    obj = [] #np.zeros((len(X), n_obj))
    for ind in range(len(X)):
        indDir = os.path.join(genDir, f'ind{ind + 1}')
        obj_ind = postProc(indDir, X)
        obj.append(obj_ind)
    return obj

def checkCompletion(var, caseDir):
    ###### Check Completion ######
    global varFile, objFile
    # load in previous variable file if it exist and
    # check if it is equal to current variables
    varFile = os.path.join(caseDir, 'var.txt')
    objFile = os.path.join(caseDir, 'obj.txt')
    if os.path.exists(varFile) and os.path.exists(objFile):
        try:
            prev_var = np.loadtxt(varFile)
            if np.array_equal(prev_var, var):
                print(f'{caseDir} already complete')
                return
        except OSError as err:
            print(err)

def completed(caseDir, var):
    ###### Check Completion ######
    # global varFile, objFile
    # load in previous variable file if it exist and
    # check if it is equal to current variables
    varFile = os.path.join(caseDir, 'var.txt')
    objFile = os.path.join(caseDir, 'obj.txt')
    if os.path.exists(varFile) and os.path.exists(objFile):
        try:
            prev_var = np.loadtxt(varFile)
            if np.array_equal(prev_var, var):
                print(f'{caseDir} already complete')
                return True
        except OSError as err:
            print(err)
            return False
    else:
        return False

# def dataMatches(fname, dat):
#     if 
    

def saveTxt(path, fname, data):
    datFile = os.path.join(path, fname)
    # save data as text file in directory  
    np.savetxt(datFile, data)


def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i


def editJobslurm(gen, ind, indDir):
    # change jobslurm.sh to correct directory and change job name
    with open(indDir + '/jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()
    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find('base_case')] + 'gen%i/ind%i' % (gen, ind) + '\n'
    job_lines[keyword_line_i] = newLine

    # find job-name line
    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'g%i.i%i' % (gen, ind) + '\n'
    job_lines[keyword_line_i] = newLine
    with open(indDir + '/jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)



# def runInd(x_i, x):
#     caseDir = os.path.join(genDir, f'ind{x_i + 1}') 
#     f = runCase(caseDir, x)
#     return f
