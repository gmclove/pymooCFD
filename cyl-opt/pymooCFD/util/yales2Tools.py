from pymooCFD.util.handleData import findKeywordLine
import os
import re

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

    
    
    

    