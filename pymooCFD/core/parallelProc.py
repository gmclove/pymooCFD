# # import multiprocessing as mp
# # import subprocess
# #
# #
# # class ParallelProc:
# #     def __init__(self, externalSolver, BaseCase,
# #                  nProc=None,
# #                  procLim=None):
# #         self.externalSolver = externalSolver
# #         if externalSolver:
# #             assert nProc is not None
# #             assert procLim is not None
# #             assert BaseCase.solverExecCmd is not None
# #             BaseCase.solve = BaseCase.externalSolve
# #         else:
# #             BaseCase.solve = BaseCase._solve
# #         BaseCase.pp = self
# #
# #     def run(self, nTasks, cases=None):
# #         if cases is None:
# #             cases = self.cases
# #
# #         if self.externalSolver:
# #             pool = mp.pool.ThreadPool(nTasks)
# #             worker = self.externalSolverWorker
# #         else:
# #             pool = mp.Pool(nTasks)
# #             worker = self.pythonSolverWorker
# #
# #         for case in cases:
# #             pool.apply_async(worker, (case, ))
# #         pool.close()
# #         pool.join()
# 
#         # if self.externalSolver:
#         #     pool = mp.pool.ThreadPool(nTasks)
#         #     for case in cases:
#         #         pool.apply_async(self.externalSolverWorker, (case, ))
#         #     pool.close()
#         #     pool.join()
#         #
#         # else:
#         #     pool = mp.Pool(nTasks)
#         #     for case in cases:
#         #         pool.apply_async(self.pythonSolverWorker, (case, ))
#         #     pool.close()
#         #     pool.join()may result. If the watch comes in contact with mercury used in thermometers, the
#
#     def externalSolverWorker(case):
#         case.preProc()
#         if case.solverExecCmd is None:
#             case.logger.error('No external solver execution command give. \
#                                Please override solve() method with python CFD \
#                                solver or add solverExecCmd to CFDCase object.')
#             raise Exception('No external solver execution command give. Please \
#                             override solve() method with python CFD solver or \
#                             add solverExecCmd to CFDCase object.')
#         else:
#             subprocess.run(case.solverExecCmd, cwd=case.caseDir,
#                            stdout=subprocess.DEVNULL)
#         # if case._execDone():
#         #     case.logger.info('RUN COMPLETE')
#         # else:
#         #     case.logger.warning('RUN FAILED TO EXECUTE')
#         #     case.logger.info('RE-RUNNING')
#         #     case.run()
#         case.postProc()
#
#     def pythonSolverWorker(case):
#         case.preProc()
#         case.solve()
#         case.postProc()
