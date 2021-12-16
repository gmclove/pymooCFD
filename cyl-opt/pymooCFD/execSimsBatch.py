import os
import subprocess
from threading import Thread
import numpy as np
import time

from pymooCFD.setupOpt import solverExecCmd, procLim, nProc, jobFile, caseMonitor

def singleNodeExec(paths): #, procLim=procLim, nProc=nProc, solverFile=solverFile):
    print('EXECUTING BATCH OF SIMULATIONS')
    print('SINGLE NODE EXECUTION')
    # All processors will be queued until all are used
    n = 0
    currentP = 0
    pids = []
    proc_labels = {}
    n_sims = len(paths)
    
    while n < n_sims:
        caseDir = paths[n]
        #     n +=1
        #     continue

        if currentP + nProc <= procLim: # currentP != procLim:
            print(f'     ## Sending {caseDir} to simulation...')
            # cmd = solverExec
            # cmd = f'cd {caseDir} && mpirun -np {nProc} {solverExec} > output.dat'
            # cmd = ['cd', caseDir, '&&', 'mpirun', '-np', str(nProc), solverExec, '>', 'output.dat']
            # Send another simulation
            # pid = subprocess.Popen(cmd, shell=True)
            # out = open(os.path.join(caseDir, 'output.dat'), 'w')
            pid = subprocess.Popen(solverExecCmd, # '>', 'output.dat'], 
                                   cwd = caseDir, stdout=subprocess.DEVNULL) #stdout=out) 
            # Store the PID of the above process
            pids.append(pid)
            # store working directory of process 
            proc_labels[pid.pid] = caseDir
            # counters
            n += 1
            currentP = currentP + nProc
        # Then, wait until completion and fill processors again
        else:
            print('     WAITING')
            # wait for any pids to complete 
            waiting = True
            while waiting:
                # check all processes for completion every _ seconds 
                time.sleep(10)
                for pid in pids:
                    if pid.poll() is not None:
                        # remove completed job from pids list 
                        pids.remove(pid)
                        # reduce currentP by nProc
                        currentP -= nProc
                        # stop waiting 
                        waiting = False
                        print('     COMPLETED: ', proc_labels[pid.pid])
                
    # Wait until all PID in the list has been completed
    print('     WAITING')
    for pid in pids:
        pid.wait()

    print('BATCH OF SIMULATIONS COMPLETE')

def slurmExec(paths, batchSize=None):
    print('EXECUTING BATCH OF SIMULATIONS')
    print('SLURM EXECUTION')
    # def editJobslurm(caseDir, ind):
    #     def findKeywordLine(kw, file_lines):
    #         kw_line = -1
    #         kw_line_i = -1
    #         for line_i, line in enumerate(file_lines):
    #             line = file_lines[line_i]
    #             if line.find(kw) >= 0:
    #                 kw_line = line
    #                 kw_line_i = line_i
    #         return kw_line, kw_line_i
    #     # change jobslurm.sh to correct directory and change job name
    #     jobExecPath = os.path.join(caseDir, jobFile)
    #     with open(jobExecPath, 'r') as f_orig:
    #         job_lines = f_orig.readlines()
    #     # use keyword 'cd' to find correct line
    #     keyword = 'cd'
    #     keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    #     # create new string to replace line
    #     newLine = f'cd {os.path.join(os.getcwd(), caseDir)} \n'
    #     job_lines[keyword_line_i] = newLine

    #     # find job-name line
    #     keyword = 'job-name='
    #     keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    #     # create new string to replace line
    #     newLine = f'#SBATCH --job-name=i{ind+1} \n'
    #     job_lines[keyword_line_i] = newLine
    #     with open(jobExecPath, 'w') as f_new:
    #         f_new.writelines(job_lines)
    if batchSize is not None:
        print(f'     sending sims in batches of {batchSize}')
        paths_batches = [paths[i:i + batchSize] 
                           for i in range(0, len(paths), batchSize)]
        for paths_batch in paths_batches:
            print(f'     SUB-BATCH: {paths_batch}')
            slurmExec(paths_batch)
        return
    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for caseDir in paths:
        # print(f'     ## Sending {caseDir} to slurm queue...')
        ## edit jobslurm.sh path
        # editJobslurm(caseDir, ind)
        out = subprocess.check_output(['sbatch', jobFile], cwd = caseDir)
        # Extract number from following: 'Submitted batch job 1847433'
        # print(int(out[20:]))
        batchIDs.append([int(out[20:]), caseDir])
    batchIDs = np.array(batchIDs)
    print('     slurm job IDs:')
    print('\t\t' + str(batchIDs).replace('\n', '\n\t\t')) 
    waiting = True
    count = np.ones(len(batchIDs))
    prev_count = [0] #count
    threads = []
    while waiting:
        time.sleep(10)
        for bID_i, bID in enumerate(batchIDs):
            # grep for batch ID of each individual
            out = subprocess.check_output(f'squeue | grep --count {bID[0]} || :', shell=True)  # '|| :' ignores non-zero exit status error
            count[bID_i] = int(out)
            if int(out) == 0:
                ### check if simulation failed 
                incomplete = caseMonitor(caseDir)
                ## if failed launch slurmExec as subprocess
                if incomplete:
                    print(f'\n\t{bID} incomplete')
                    t = Thread(target=slurmExec, args=([bID[1]],))
                    t.start()
                    threads.append(t)
                else:
                    print(f'\n\t{bID} complete')
        ### update number of jobs waiting display
        if sum(count) != sum(prev_count):
            print(f'\n     sims still running or queued = {int(sum(count))}', end='')
        else:
            print('.', end='')
        prev_count = count
        ### check if all batch jobs are done
        if sum(count) == 0:
            waiting = False
            print('\n     DONE WAITING')

    ## wait for second run of failed cases to complete
    for thread in threads:
        thread.join()
    print()
    print('BATCH OF SLURM SIMULATIONS COMPLETE')
    
# def editJobslurm(path):
#     def findKeywordLine(kw, file_lines):
#         kw_line = -1
#         kw_line_i = -1
#         for line_i, line in enumerate(file_lines):
#             line = file_lines[line_i]
#             if line.find(kw) >= 0:
#                 kw_line = line
#                 kw_line_i = line_i
#         return kw_line, kw_line_i
    
#     i_gen = path.rindex('gen')
#     gen = path[i_gen+1:i_gen+2]
#     i_ind = path.rindex('ind')
#     ind = path[i_ind+1:i_ind+2]
    
#     # change jobslurm.sh to correct directory and change job name
#     with open(path, 'r') as f_orig:
#         job_lines = f_orig.readlines()
#     # use keyword 'cd' to find correct line
#     keyword = 'cd'
#     keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
#     # create new string to replace line
#     newLine = f'cd {os.path.join(os.getcwd(), path)} \n'
#     job_lines[keyword_line_i] = newLine

#     # find job-name line
#     keyword = 'job-name='
#     keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
#     # create new string to replace line
#     newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'g%ii%i' % (gen, ind) + '\n'
#     job_lines[keyword_line_i] = newLine
#     with open(path, 'w') as f_new:
#         f_new.writelines(job_lines)
    
# def slurmExec(dir, subdir, n_sims):
#     # # check if meshStudy is already complete
#     # completeCount = 0
#     # for sf in meshSF:
#     #     sfDir = f'{studyDir}/{sf}'
#     #     if os.path.exists(f'{sfDir}/solver01_rank00.log'):
#     #         completeCount += 1
#     # if completeCount == len(meshSF):
#     #     return

#     # Queue all the individuals in the generation using SLURM
#     batchIDs = []  # collect batch IDs
#     for n in range(n_sims):
#         caseDir = f'{dir}/{subdir}{n}'
#         if os.path.exists(f'{caseDir}/solver01_rank00.log'):
#             pass
#         else:
#             # create string for directory of individuals job slurm shell file
#             jobDir = f'{caseDir}/jobslurm.sh'
#             out = subprocess.check_output(['sbatch', jobDir])
#             # Extract number from following: 'Submitted batch job 1847433'
#             # print(int(out[20:]))
#             batchIDs.append(int(out[20:]))
#     # print(batchIDs)

#     waiting = True
#     count = np.ones(len(batchIDs))
#     # processes = []
#     while waiting:
#         for bID_i in range(len(batchIDs)):
#             # grep for batch ID of each individual
#             out = subprocess.check_output('squeue | grep --count %i || :' % batchIDs[bID_i], shell=True)  # '|| :' ignores non-zero exit status error
#             count[bID_i] = int(out)
#         # print(count)
#         # check if all batch jobs are done
#         if sum(count) == 0:
#             waiting = False
#         # print(count)
#         # print('SUM OF COUNT = %i' % sum(count))
#         time.sleep(1)

#     print('BATCH OF SLURM SIMULATIONS COMPLETE')


# def completed(caseDir, var):
#     ###### Check Completion ######
#     # global varFile, objFile
#     # load in previous variable file if it exist and
#     # check if it is equal to current variables
#     varFile = os.path.join(caseDir, 'var.txt')
#     objFile = os.path.join(caseDir, 'obj.txt')
#     if os.path.exists(varFile) and os.path.exists(objFile):
#         try:
#             prev_var = np.loadtxt(varFile)
#             if np.array_equal(prev_var, var):
#                 print(f'{caseDir} already complete')
#                 return True
#         except OSError as err:
#             print(err)
#             return False
#     else:
#         return False
