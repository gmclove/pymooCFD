from pymooFEA.util.handleData import findKeywordLine
import numpy as np
import os
import re
from glob import glob

def getLatestMesh(dumpDir):
    ents = os.listdir(dumpDir)
    ents.sort()
    for ent in ents:
        if ent.endswith('.mesh.h5'):
            latestMesh = ent
    return latestMesh


def getLatestSoln(dumpDir):
    ents = os.listdir(dumpDir)
    ents.sort()
    for ent in ents:
        if ent.endswith('.sol.h5'):
            latestSoln = ent
    return latestSoln

def getLatestXMF(dumpDir):
    ents = os.listdir(dumpDir)
    ents.sort()
    for ent in ents:
        if ent.endswith('.xmf') and not re.search('.sol.+_.+\\.xmf', ent):
            latestXMF = ent
    return latestXMF


def getLatestDataFiles(dumpDir):
    latestMesh = getLatestMesh(dumpDir)
    latestSoln = getLatestSoln(dumpDir)
    return latestMesh, latestSoln


def setRestart(inPath, dumpDir):
    # latestMesh, latestSoln = getLatestDataFiles(dumpDir)
    latestMesh = getLatestMesh(dumpDir)
    latestSoln = getLatestSoln(dumpDir)

    with open(inPath, 'r') as f:
        in_lines = f.readlines()
    kw = 'RESTART_TYPE = GMSH'
    kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    in_lines[kw_line_i] = '#' + kw
    kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
    kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    in_lines[kw_line_i] = '#' + kw
    kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
    kw_line, kw_line_i = findKeywordLine(kw, in_lines)
    in_lines[kw_line_i] = '#' + kw
    with open(inPath, 'w') as f:
        f.writelines(in_lines)

def getWallTime(caseDir):
    search_str = os.path.join(caseDir, 'solver01_rank*.log')
    fName, = glob(search_str)
    with open(fName, 'rb') as f:
        try:  # catch OSError in case of a one line file
            f.seek(-1020, os.SEEK_END)
        except OSError:
            f.seek(0)
        clock_line = f.readline().decode()
    if 'WALL CLOCK TIME' in clock_line:
        wall_time = int(float(clock_line[-13:]))
        print(f'YALES2 Wall Clock Time: {wall_time} seconds')
    else:
        print('no wall clock time found')
        wall_time = None
    return wall_time


def getY2LineData(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    line_dat = []
    t_step_dat = []
    for i, line in enumerate(lines):
        if line[0] == '&':
            line_dat.append(t_step_dat)
            t_step_dat = []
        else:
            t_step_dat.append(np.fromstring(line, sep=' ', dtype=np.float64))
    line_dat = np.array(line_dat)
    t = np.array([t_step[0][1] for t_step in line_dat])
    return line_dat, t
