import os
import shutil
from pymooCFD.setupOpt import *

class PtsList(object):
    pass


def makeClean(recompBase=False, stashPrev=True):  # , rmGen=False):
    def findKeywordLine(kw, file_lines):
        kw_line = -1
        kw_line_i = -1

        for line_i in range(len(file_lines)):
            line = file_lines[line_i]
            if line.find(kw) >= 0:
                kw_line = line
                kw_line_i = line_i

        return kw_line, kw_line_i
    ########################################################################################################################
    if stashPrev:
        stashDir = 'stash'
        try:
            os.mkdir(stashDir)
        except OSError as err:
            print(err)
            print('"stash" directory already exists')
        try:
            shutil.copytree(dataDir, f'{stashDir}/prev_run')
            # shutil.move('obj.txt', f'{stashDir}/obj.txt')
            # shutil.move('var.txt', f'{stashDir}/var.txt')
        except OSError as err:
            print(err)
            print(f'{dataDir} do not exist')
        os.system(f'mv checkpoint* {stashDir}')

    # os.system('source activate pymoo-CFD')
    os.system('rm -rfv gen*/ checkpoint* output.dat __pycache__ plots/')
    os.system('rm -rfv base_case/output.dat base_case/dump/ base_case/solver01_*')  # base_case/'+dataFile)
    os.system('rm -fv obj.txt var.txt')
    ########################################################################################################################
    ###### MOO JOBSLURM ######
    ##########################
'''
    # change jobslurm.sh to correct procLim
    with open('./jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()

    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = 'cd ' + os.getcwd() + '\n'
    job_lines[keyword_line_i] = newLine

    # use keyword 'cd' to find correct line
    keyword = '#SBATCH --nodes='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword + str(procLim) + '\n'
    job_lines[keyword_line_i] = newLine

    # change slurm job-name
    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + jobName + '\n'
    job_lines[keyword_line_i] = newLine

    with open('./jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)

    ########################################################################################################################
    ###### BASE_CASE JOBSLURM ######
    ################################
    # change jobslurm.sh to correct procLim
    with open('./base_case/jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()

    # use keyword to find correct line
    keyword = '#SBATCH --nodes='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword + str(nProc) + '\n'
    job_lines[keyword_line_i] = newLine

    # change slurm job-name
    # keyword = 'job-name='
    # keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # # create new string to replace line
    # newLine = keyword_line[:keyword_line.find(keyword)] + keyword + jobName + '\n'
    # job_lines[keyword_line_i] = newLine

    with open('./base_case/jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)
'''
    ########################################################################################################################
    ###### RECOMPILE BASE_CASE #######
    ##################################
    # NOT POSSIBLE FROM INSIDE CONDA ENVIROMENT NEEDED TO RUN PYTHON SCRIPT
    # if recompBase:
    #     os.system('cd ./base_case && make veryclean')
    #     # os.system('source deactivate && source activate ~/.bashrc && yalesmodules')
    #     os.system('module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)'
    #                 ' && cd ./base_case && source deactivate && make')
