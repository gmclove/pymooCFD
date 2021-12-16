# @Author: glove
# @Date:   2021-12-13T15:31:49-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-14T10:18:03-05:00



from pymooCFD.setupOpt import optDatDir, problem, client
from pymooCFD.util.handleData import loadCP, saveCP
from pymooCFD.util.sysTools import emptyDir, copy_and_overwrite #, archive
import numpy as np
import os

def runOpt(restart=True):
    if restart:
        algorithm = loadCP()
        print("Loaded Checkpoint:", algorithm)
        print(f'Last checkpoint at generation {algorithm.n_gen}')
        # restart client if being used
        if client is not None:
            client.restart()
            print("CLIENT RESTARTED")
    else:
        print('STARTING NEW OPTIMIZATION STUDY')
        # archive/empty previous runs data directory
        emptyDir(optDatDir)
        # load algorithm defined in setupOpt.py module
        from pymooCFD.setupOpt import algorithm
        algorithm.setup(problem,
                        seed=algorithm.seed,
                        verbose=algorithm.verbose,
                        save_history=algorithm.save_history,
                        return_least_infeasible=algorithm.return_least_infeasible
                        )
        # start client if being used
        if client is not None:
            client()
            print("CLIENT STARTED")



    ########################################################################################################################
    ######    OPTIMIZATION    ######
    # until the algorithm has not terminated
    while algorithm.has_next():
        ### First generation
        ## population is None so ask for new pop
        if algorithm.pop is None:
            print('     START-UP: first generation')
            evalPop = algorithm.ask()
            algorithm.pop = evalPop
            algorithm.off = evalPop
        ### Previous generation complete
        ## If current objective does not have None values then get new pop
        ## ie previous pop is complete
        ## evaluate new pop
        elif None not in algorithm.pop.get('F'):
            print('     START-UP: new generation')
            evalPop = algorithm.ask()
            algorithm.off = evalPop
        ### Mid-generation start-up
        ## evaluate offspring population
        else:
            print('     START-UP: mid-generation')
            evalPop = algorithm.off

        ## save checkpoint before evaluation
        saveCP(algorithm)

        ## evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(problem, evalPop)

        ## returned the evaluated individuals which have been evaluated or even modified
        ## checkpoint saved after in algorithm.callback.notify() method
        algorithm.tell(infills=evalPop)

        ## save top optimal evaluated cases in pf directory
        n_opt = 20
        for off_i, off in enumerate(algorithm.off):
            for opt_i, opt in enumerate(algorithm.opt[:n_opt]):
                if np.array_equal(off.X, opt.X):
                    pfDir = os.path.join(optDatDir, 'pf', f'opt{opt_i+1}')
                    offDir = os.path.join(optDatDir, f'gen{algorithm.n_gen}', f'ind{off_i+1}')
                    print(f'     Updating Pareto front folder: {offDir} -> {pfDir}')
                    copy_and_overwrite(offDir, pfDir)

        ## do some more things, printing, logging, storing or even modifying the algorithm object
        # print(algorithm.n_gen, algorithm.evaluator.n_eval)
        # print('Parameters:')
        # print(algorithm.pop.get('X'))
        # print('Objectives:')
        # print(algorithm.pop.get('F'))
        #algorithm.display.do(algorithm.problem,
        #                      algorithm.evaluator,
        #                      algorithm
        #                      )


    # obtain the result objective from the algorithm
    res = algorithm.result()

    # calculate a hash to show that all executions end with the same result
    print("hash", res.F.sum())










    # while optStudy.algorithm.has_next():
    #     optStudy.algorithm.next()
    #     optStudy.algorithm.display.do(optStudy.algorithm.problem,
    #                                   optStudy.algorithm.evaluator,
    #                                   optStudy.algorithm
    #                                   )

    # from pymoo.optimize import minimize
    # from pymooCFD.setupOpt import problem, callback, display, n_gen, DaskClient #, n_workers # , MyDisplay  # , termination

    # problem = problem(DaskClient())
    # # from dask_jobqueue import SLURMCluster
    # # cluster = SLURMCluster()
    # # cluster.scale(jobs=algorithm.pop_size)    # Deploy ten single-node jobs

    # # from dask.distributed import LocalCluster
    # # cluster = LocalCluster(n_workers = 10, processes = True)

    # # from dask.distributed import Client  # , LocalCluster
    # # client = Client(cluster(n_workers=n_workers))
    # # problem = problem(client)

    # res = minimize(problem=problem,
    #                algorithm=algorithm,
    #                # termination=termination,
    #                termination=('n_gen', n_gen),
    #                callback=callback,
    #                display=display,
    #                # display=MyDisplay(),
    #                seed=1,
    #                copy_algorithm=True,
    #                # pf=problem.pareto_front(use_cache=False),
    #                save_history=True,
    #                verbose=True
    #                )
    # # client.close()
    # # np.save("checkpoint", algorithm)
    # print("EXEC TIME: %.3f seconds" % res.exec_time)









        # from pymooCFD.util.handleData import loadTxt
            # algorithm = loadTxt('dump/gen13X.txt', 'dump/gen13F.txt')
        # except OSError as err:
        #     print(err)
        #     from pymooCFD.setupOpt import checkpointFile
        #     print(f'{checkpointFile} load failed.')
        #     print('RESTART FAILED')
        #     return







        # try:
        #     os.remove(f'{dataDir}/obj.npy')
        # except OSError as err:
        #     print(err)
        # try:
        #     checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
        #     print("Loaded Checkpoint:", checkpoint)
        #     # only necessary if for the checkpoint the termination criterion has been met
        #     checkpoint.has_terminated = False
        #     algorithm = checkpoint
        #     print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
        # except OSError as err:
        #     print(err)
        #     from pymooCFD.setupOpt import algorithm
            # try:
            #     X = np.loadtxt('var.txt')
            #     F = np.loadtxt('obj.txt')
            #     from pymoo.model.evaluator import Evaluator
            #     from pymoo.model.population import Population
            #     from pymoo.model.problem import StaticProblem
            #
            #     # now the population object with all its attributes is created (CV, feasible, ...)
            #     pop = Population.new("X", X)
            #     pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)
    ########################################################################################################################
    ######    PARALLELIZE    ######
    # from pymooCFD.setupOpt import GA_CFD, n_parallel_proc
    # if parallel.lower() == 'multiproc':
    #     ###### Multiple Processors ######
    #     # parallelize across multiple CPUs on a single machine (node)
    #     # NOTE: only one CFD simulation per CPU???
    #     import multiprocessing
    #     # the number of processes to be used
    #     n_proccess = n_parallel_proc
    #     pool = multiprocessing.Pool(n_proccess)
    #     problem = GA_CFD(parallelization=('starmap', pool.starmap))
    # elif any(selection.lower() == parallel.lower() for selection in multiProcSelect):
    #     from pymooCFD.setupCFD import runCFD
    #     ###### Multiple Nodes ######
    #     # Use Dask library to parallelize across
    #     ### SLURM Cluster ###
    #     from dask_jobqueue import SLURMCluster
    #     cluster = SLURMCluster()
    #     # ask to scale to a certain number of nodes
    #     cluster.scale(jobs=n_parallel_proc)  # Deploy n_parallel_proc single-node jobs
    #     ### Launch Client ###
    #     from dask.distributed import Client
    #     client = Client(cluster)
    #     problem = GA_CFD(parallelization=("dask", client, runCFD))
    #     pool = client
    # else:
    #     ###### Multiple Threads ######
    #     # Run on multiple threads within a single CPU
    #     from multiprocessing.pool import ThreadPool
    #     # the number of threads to be used
    #     n_threads = n_parallel_proc
    #     # initialize the pool
    #     pool = ThreadPool(n_threads)
    #     # define the problem by passing the starmap interface of the thread pool
    #     problem = GA_CFD(parallelization=('starmap', pool.starmap))
