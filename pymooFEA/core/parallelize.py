import multiprocessing as mp
import pymooFEA.config as config
import logging
# import subprocess
# import multiprocessing.pool



class Parallelize:
    def __init__(self, exec_funct, procLim, nProc,
                 nTasks=None,
                 externalSolver=False, solverCmd=None,
                 onlyParallelizeSolve=False):
        self.logger = logging.getLogger()
        self.onlyParallelizeSolve = onlyParallelizeSolve
        self.externalSolver = externalSolver
        self.solverCmd = solverCmd
        if self.externalSolver and self.solverCmd is None:
            raise Exception

        if nTasks is None:
            if nProc is None or procLim is None:
                self.nTasks = config.MP_POOL_NTASKS_MAX
            else:
                self.nTasks = int(procLim / nProc)
        else:
            self.nTasks = nTasks

        if self.externalSolver:
            self.exec_funct = self.solveExternal
            self.pool = mp.pool.ThreadPool(self.nTasks)
        else:
            self.exec_funct = self._solve
            self.pool = mp.Pool(self.nTasks)

    def exec(self, cases):
        #self.logger.info('PARALLELIZING . . .')
        if self.onlyParallelizeSolve:
            # print('\tParallelizing Only Solve')
            for case in cases:
                case.preProc()
            print('PARALLELIZING . . .')
            for case in cases:
                self.pool.apply_async(case.solve, ())
            self.pool.close()
            self.pool.join()
            for case in cases:
                case.postProc()
        else:
            print('PARALLELIZING . . .')
            for case in cases:
                self.pool.apply_async(case.run, ())
            self.pool.close()
            self.pool.join()

    # def solveExternal(self):
    #     self.logger.info('SOLVING AS SUBPROCESS...')
    #     self.logger.info(f'\tcommand: {self.solverCmd}')
    #     subprocess.run(self.solverCmd, cwd=self.abs_path,
    #                    stdout=subprocess.DEVNULL)


    ###  Parallel Processing  ###
    @classmethod
    def parallelizeInit(cls, externalSolver=None):
        if externalSolver is None:
            externalSolver = cls.externalSolver
        if cls.nTasks is None:
            if cls.nProc is not None and cls.nProc is not None:
                cls.nTasks = int(cls.procLim / cls.nProc)
            else:
                cls.nTasks = config.MP_POOL_NTASKS_MAX
        else:
            nTasks = cls.nTasks
        if externalSolver:
            assert cls.solverExecCmd is not None
            assert cls.nTasks is not None
            cls._solve = cls.solveExternal
            cls.pool = mp.pool.ThreadPool(cls.nTasks)
        else:
            cls._solve = cls._solve
            cls.pool = mp.Pool(cls.nTasks)

    @classmethod
    def parallelize(cls, cases):
        cls.parallelizeInit()
        #cls.logger.info('PARALLELIZING . . .')
        if cls.onlyParallelizeSolve:
            # print('\tParallelizing Only Solve')
            for case in cases:
                case.preProc()
            print('PARALLELIZING . . .')
            for case in cases:
                cls.pool.apply_async(case.solve, ())
            cls.pool.close()
            cls.pool.join()
            for case in cases:
                case.postProc()
        else:
            print('PARALLELIZING . . .')
            for case in cases:
                cls.pool.apply_async(case.run, ())
            cls.pool.close()
            cls.pool.join()
