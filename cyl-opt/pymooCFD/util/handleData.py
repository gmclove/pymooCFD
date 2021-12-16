from pymooCFD.setupOpt import checkpointPath, optDatDir, nCP, archDir#, \
    # preProcDir, inputFile
from pymooCFD.util.sysTools import removeDir #, makeDir, emptyDir
# from pymooCFD.setupCFD import runCase


import numpy as np
import time
import os
import tarfile
# from dask.distributed import get_client #, Client

# from sys import exit


# def getGen(checkpointPath=checkpointPath):
#     try: 
#         loadCP(checkpointPath=checkpointPath)
#     except FileNotFoundError as err:
#         print(err)
#         return 0

def loadTxtCP(Xpath, Fpath):
    X = np.lodatxt(Xpath)
    F = np.loadtxt(Fpath)
    from pymooCFD.setupOpt import algorithm, problem
    
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    from pymoo.problems.static import StaticProblem    
    # now the population object with all its attributes is created (CV, feasible, ...)
    pop = Population.new("X", X)
    pop = Evaluator().eval(StaticProblem(problem, F=F), pop)
    # algorithm.sampling = pop
    algorithm.sampling = pop   

def archive(dirToComp, archDir=archDir, background=True):
    if background:
        from multiprocessing import Process
        p = Process(target=compressDir, args=(dirToComp, archDir))
        p.start()
    else:
        compressDir(dirToComp, archDir)


def compressDir(dirToComp, archDir):
    print(f'{dirToComp} compression started')
    # destination file naming
    timestr = time.strftime("%y%m%d-%H%M")
    try:
        fname = f'{dirToComp[dirToComp.rindex("/"):]}_{timestr}'
    except ValueError:
        fname = f'{dirToComp}_{timestr}'
    # concatenate compression file path and name
    compFile = os.path.join(archDir, f'{fname}.tar.gz')
    with tarfile.open(compFile, 'w:gz') as tar:
        tar.add(dirToComp)
    print(f'{dirToComp} compression finished')
    removeDir(dirToComp)


def saveCP(algorithm):
    gen = algorithm.callback.gen
    print(f'SAVING CHECKPOINT - GENERATION {gen}')
    # genDir = f'gen{gen}'
    # os.path.join(optDatDir, 'checkpoint')
    np.save(checkpointPath, algorithm)
    # gen0 and every nCP generations save additional static checkpoint
    if gen % nCP == 1:
        np.save(os.path.join(optDatDir, f'checkpoint-gen{gen}'), algorithm)
    # save text file of variables and objectives as well
    # this provides more options for post-processesing data
    genX = algorithm.pop.get('X')
    with open(f'{optDatDir}/gen{gen}X.txt', "w+") as file:  # write file
        np.savetxt(file, genX)
    try:
        genF = algorithm.pop.get('F')
        with open(f'{optDatDir}/gen{gen}F.txt', "w+") as file:  # write file
            np.savetxt(file, genF)
    except TypeError: #AttributeError
        print('     mid-generation')

def loadCP(checkpointPath=checkpointPath, hasTerminated=False):
    try:
        checkpoint, = np.load(checkpointPath, allow_pickle=True).flatten()
        # only necessary if for the checkpoint the termination criterion has been met
        checkpoint.has_terminated = hasTerminated
        alg = checkpoint
        # Update any changes made to the algorithms between runs
        from pymooCFD.setupOpt import pop_size, n_offsprings, xl, xu, termination, setAlgorithm
        alg.termination = termination
        alg.pop_size = pop_size
        alg.n_offsprings = n_offsprings
        alg.problem.xl = np.array(xl)
        alg.problem.xu = np.array(xu)
        setAlgorithm(alg)
        return alg
    except FileNotFoundError as err:
        print(err)
        raise Exception(f'{checkpointPath} load failed.')
        # return None

def printArray(array, labels, title):
    print(title, ' - ', end='')
    for i, label in enumerate(labels):
        print(f'{label}: {array[i]} / ', end='')
    print()
    
    


def runPop(X):
    # client = Client(cluster())
    # client = get_client()
    # def fun(x_i, x):
    #     caseDir = os.path.join(preProcDir, f'lim_perm_sim-{x_i}') 
    #     f = runCase(caseDir, x)
    #     return f
    # jobs = [client.submit(fun, x_i, x) for x_i, x in enumerate(X)]
    # obj = np.row_stack([job.result() for job in jobs])
    # client.close()
    
    
    # create sub-directories for each individual
    indDirs = []
    for i in range(len(X)):
        indDirs.append(os.path.join(genDir, f'ind{i+1}'))

    from pymooCFD.setupCFD import preProcGen, postProcGen
    ##### PRE-PROCCESS GENERATION #####
    preProcGen(perProcDir, X)
    # jobs = [self.client.submit(preProc, indDirs[i], x) for i, x in enumerate(X)]
    # [job.result() for job in jobs]
    for i, x in enumerate(X):
        preProc(indDirs[i], x)
    # for indDir in indDirs:
    #     copy_tree(baseCaseDir, indDir)
    
    ###### RUN GENERATION ######
    # from pymooCFD.setupCFD import runGen
    # obj = runGen(genDir, X)
    ###### RUN GENERATION ######
    from pymooCFD.execSimsBatch import slurmExec #singleNodeExec
    # singleNodeExec(indDirs)
    slurmExec(indDirs)
    
    ##### POST-PROCCESS GENERATION #####
    # jobs = [self.client.submit(postProc, indDirs[i], x) for i, x in enumerate(X)]
    # obj = np.row_stack([job.result() for job in jobs])
    obj = []
    obj = np.zeros(len(X))
    for i in range(len(X)):
        obj[i] = postProc(indDirs[i], X[i])

    
    return obj


def loadTxt(folder, fname):
    file = os.path.join(folder, fname)
    dat = np.loadtxt(file)
    return dat


def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i



# def yales2Restart(optDatDir=optDatDir):
#     ents = os.listdir(optDatDir)
#     for ent in ents:
#         if ent.isdir() and ent.startswith('gen'):
#             genDir = os.path.join(optDatDir,ent)
#             genEnts = os.listDir(genDir)
#             for indEnt in genEnts:
#                 if indEnt.isdir() and indEnt.startseith('ind'):
#                     indDir = os.path.join(genDir, indEnt)
#                     inputPath = os.path.join(indDir, inputFile)
#                     with open(inputPath, 'a') as f:
#                         f.write('RESTART_LOAD_ALL_DATA = TRUE # loads all the data in the HDF files')
#                     print('Appended ', inputPath)


# def popGen(gen, checkpointPath=checkpointPath):
#     '''

#     Parameters
#     ----------
#     gen : int
#         generation you wish to get population from
#     checkpointPath : str, optional
#         checkpoint file path where Algorithm object was saved using numpy.save().
#         The default is checkpointPath (defined in beginning of setupOpt.py).

#     Returns
#     -------
#     pop :
#         Contains StaticProblem object with population of individuals from
#         generation <gen>.

#     Notes
#     -----
#         - development needed to handle constraints
#     '''
#     alg = loadCP(checkpointPath=checkpointPath)
#     X = alg.callback.data['var'][gen]
#     F = alg.callback.data['obj'][gen]

#     from pymoo.model.evaluator import Evaluator
#     from pymoo.model.population import Population
#     from pymoo.model.problem import StaticProblem
#     # now the population object with all its attributes is created (CV, feasible, ...)
#     pop = Population.new("X", X)
#     pop = Evaluator().eval(StaticProblem(problem, F=F), pop)  # , G=G), pop)
#     return pop, alg


# def loadTxt(fileX, fileF, fileG=None):
#     print(f'Loading population from files {fileX} and {fileF}...')
#     X = np.loadtxt(fileX)
#     F = np.loadtxt(fileF)
#     # F = np.loadtxt(f'{optDatDir}/{fileF}')
#     if fileG is not None:
#         # G = np.loadtxt(f'{optDatDir}/{fileG}')
#         G = np.loadtxt(fileG)
#     else:
#         G = None

#     from pymoo.model.evaluator import Evaluator
#     from pymoo.model.population import Population
#     from pymoo.model.problem import StaticProblem
#     # now the population object with all its attributes is created (CV, feasible, ...)
#     pop = Population.new("X", X)
#     pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)

#     from pymooCFD.setupOpt import pop_size
#     # from pymoo.algorithms.so_genetic_algorithm import GA
#     # # the algorithm is now called with the population - biased initialization
#     # algorithm = GA(pop_size=pop_size, sampling=pop)
#     from pymoo.algorithms.nsga2 import NSGA2
#     algorithm = NSGA2(pop_size=pop_size, sampling=pop)

#     return algorithm




# def restartGen(gen, checkpointPath=checkpointPath):
#     pop, alg = popGen(gen, checkpointPath=checkpointPath)
#     alg.sampling()

#     # from pymoo.algorithms.so_genetic_algorithm import GA
#     # the algorithm is now called with the population - biased initialization
#     # algorithm = GA(pop_size=100, sampling=pop)

#     from pymoo.optimize import minimize
#     from pymooCFD.setupOpt import problem
#     res = minimize(problem,
#                    alg,
#                    ('n_gen', 10),
#                    seed=1,
#                    verbose=True)
#     return res


# def loadTxt():
#     try:
#         print('Loading from text files')
#         X = np.loadtxt('var.txt')
#         F = np.loadtxt('obj.txt')
#     except OSError as err:
#         print(err)
#         print('Failed to load text files')
#         print('Data loading failed returning "None, None"...')
#         return None, None

# def archive(dirName, archName = 'archive.tar.gz'):
#     with tarfile.open(archName, 'a') as tar:
#         tar.add(dirName)

# compressDir('../../dump')


# print('creating archive')
# out = tarfile.open('example.tar.gz', mode='a')
# try:
#     print('adding README.txt')
#     out.add('../dump')
# finally:
#     print('closing tar archive')
#     out.close()
#
# print('Contents of archived file:')
# t = tarfile.open('example.tar.gz', 'r')
# for member in t.getmembers():
#     print(member.name)
