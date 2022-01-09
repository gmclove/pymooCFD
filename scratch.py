# @Author: glove
# @Date:   2021-12-10T11:22:40-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-14T15:45:19-05:00
import os
import subprocess
caseDir = os.path.join(os.getcwd(), 'test_case')
# caseDir = 'test_case'
solverExecCmd = ['cd', 'test_case', '&&', 'mpirun', '-n',
                 '10', '2D_cylinder' '>', 'pyTest.out']
subprocess.run(solverExecCmd,  # cwd=caseDir,
               stdout=subprocess.DEVNULL)

# import numpy as np
#
# A = np.around(np.arange(0.5, 1.5, 0.1), decimals=2)
# print(A)
# print([a for a in A])
# A = np.around(A, decimals=2)
# print(A)
# print([a for a in A])


# numpy.around(a, decimals=0)


# lines = ['asfasdf/n', 'assdasdf/n', 'sfassdf/n', 'sdfsf']
#
# for i, line in enumerate(lines):
#     if not line.endswith('\n'):
#         lines[i] += '\n'
#
# print(lines)


# import numpy as np
# print(int(23/5))

# class A:
#     @classmethod
#     def a(cls):
#         print('A')
#
# class B(A):
#     @classmethod
#     def a(cls):
#         print('B')
#
#     @staticmethod
#     def b(pts):
#         for pt in pts:
#
# b = B()
# b.a()
# B.a()
# A.a()
# A().a()

# import os
# import numpy as np
# print('scratch.py')
#
# from pymooCFD.studies.oscillCyl import BaseCase, MyOptStudy
#
# wds = ['test_case1', 'test_case2', 'test_case3']
# X = [[1.5, 0.1], [0.5, 0.2], [1, 0.3]]
# cases = []
# for i, wd in enumerate(wds):
#     cases.append(BaseCase('osc-cyl_base', wd, X[i]))
#
# from setupOpt import algorithm, problem
# MyOptStudy(algorithm, problem, BaseCase).runPop(cases)

# BaseCase('base_cases/osc-cyl_base', 'test_case', [0,0]).genMesh()

# import subprocess
# import multiprocessing as mp
# from tqdm import tqdm
#
# NUMBER_OF_TASKS = len(wds)
#
# procLim = 40
# nProc = 20
# nTask = int(procLim/nProc)
# progress_bar = tqdm(total=nTask)
#
#
# def solve(wd):
#     # command = ['python', 'worker.py', sec_sleep]
#     cmd = ['mpirun', '-n', str(nProc), '2D_cylinder']
#     print(cmd)
#     print(wd)
#     subprocess.run(cmd, cwd=wd, stdout=subprocess.DEVNULL)
#     print('COMPLETE', wd)
#
#
# def execCallback():
#     if not _execDone():
#         pool.apply_async(solve, (wd,), callback=execCallback)
#
#
# def _execDone():
#     return True
#     # progress_bar.update()
#
#
# if __name__ == '__main__':
#
#     pool = mp.Pool(nTask)
#
#     # for seconds in [str(x) for x in range(1, NUMBER_OF_TASKS + 1)]:
#     for wd in wds:
#         pool.apply_async(solve, (wd,), callback=execCallback)
#
#     # check if simulation completed correctly
#     # if not self._execDone():
#     #     pass
#
#     pool.close()
#     pool.join()


# import subprocess
# cmd = ['mpirun', '2D_cylinder']
# wds = ['test_case1', 'test_case2', 'test_case3']
#
# for wd in wds:
#     subprocess.Popen(cmd, cwd=wd)

# # def loadCases(directory):
# directory = os.path.join('opt_run', 'gen1')
# print(os.listdir(directory))
# cases = []
# ents = os.listdir(directory)
# for ent in ents:
#     ent_path = os.path.join(directory, ent)
#     if os.path.isdir(ent_path):
#         for e in os.listdir(ent_path):
#             caseCP = os.path.join(ent_path, 'case.npy')
#             if os.path.exists(caseCP):
#                 case, = np.load(caseCP, allow_pickle=True).flatten()
#                 cases.append(case)
# # return cases
# # loadCases('opt_run/gen1')


# import numpy as np
# class Test:
#     var = 1
#     def func(self):
#         print(self.var)
#     def __str__(self):
#         return 'TEST'
#
# test1 = Test()
# test2 = Test()
# l = np.array([test1, test2])
# np.save('cpTest.npy', l, allow_pickle=True)
# print(l)
# list = np.load('cpTest.npy', allow_pickle=True).flatten()
# print(list)


# import numpy as np
# class Test:
#     def __init__(self):
#         self.l = np.array([0.006599578695870828, 0.35595787774394816])
#     def __str__(self):
#         return f'{self.l}'
# # l = [0.006599578695870828, 0.35595787774394816]
# test = Test()
# print(test.l[0])
# # print(test.l)
# # print(f'{test.l}')
# print(test)
# print(f'{test}')
# print(type(test.l))
# # with np.printoptions(suppress=True):
# #     print(test.l)


# class Test:
#     var = 1
#     def func(self):
#         print(self.var)
#     def __str__(self):
#         return 'TEST'
#
# test1 = Test()
# test2 = Test()
# test1.__str__ = lambda self: 'Test 1'
# l = [[1, test1], [2, test2]]
# for e in l: print(f'\t\t {e[0]} {e[1]}')


# # from http://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
# import functools
# def run_once(f):
#     """Runs a function (successfully) only once.
#
#     The running can be reset by setting the `has_run` attribute to False
#     """
#     @functools.wraps(f)
#     def wrapper(*args, **kwargs):
#         if not wrapper.complete:
#             result = f(*args, **kwargs)
#             wrapper.complete = True
#             return result
#     wrapper.complete = False
#     return wrapper
#
# def calltracker(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         wrapper.complete = True
#         return result
#     wrapper.complete = False
#     return wrapper
#
# @run_once
# def test1():
#     print('test1')
#
# @calltracker
# def test2():
#     print('test2')
#     print(test2.complete)
#
# test1()
# test1()
#
# test2()
# test2()
#
# import numpy
# numpy.save('test2', test2)
# test2, = numpy.load('test2.npy', allow_pickle=True).flatten()
# test2()
#
# numpy.save('test1', test1)
# test1, = numpy.load('test1.npy', allow_pickle=True).flatten()
# test1()

# import shutil
# shutil.copytree('base_case', 'test_case')

# xl = [1, 2, 5]
# xu = [3, 4, 6]
# n_var= 3
# x_mid = [xl[x_i]+(xu[x_i]-xl[x_i])/2 for x_i in range(n_var)]
# print(x_mid)
