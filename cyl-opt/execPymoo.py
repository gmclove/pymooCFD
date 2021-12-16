def main():
    # from pymooCFD.util.handleData import loadTxtCP, saveCP 
    # alg = loadTxtCP('opt_run/gen50X.txt', 'opt_run/gen50F.txt')
    # saveCP(alg)
    
    #from pymooCFD.util.handleData import archive 
    #archive('opt_run', background=False)
    
    #from pymooCFD.preProcOpt import meshStudy
    #meshStudy(restart=False)    

    #from pymooCFD.preProcOpt import runGen1
    #runGen1(restart=False)
    
    ##### PRE-PROC OPT: RUN CORNER CASES #####
    #from pymooCFD.setupOpt import xl, xu
    #from pymooCFD.preProcOpt import runCornerCases
    #runCornerCases(xl, xu)

    ##### RUN OPTIMIZATION STUDY #####
    from pymooCFD.runOpt import runOpt
#    runOpt(restart=True)
    runOpt(restart=False)

    # from pymooCFD.setupCFD import runCase
    # runCase('preProcOpt/maxTimeSim', [0.05, 10])    

    ###### GENERATE MESH ########
    #from pymooCFD.genMesh import genMesh 
    #genMesh('test_case/jet_cone-axi_sym.unv', 1, 0.02)

    ###### PRE-PROCESS SINGLE CASE ########
    # from pymooCFD.setupCFD import preProc
    # caseDir = 'test_case'
    # preProc(caseDir, [0,0])
    
    ###### POST-PROCESS SINGLE CASE ########
    # import numpy as np
    # from pymooCFD.setupCFD import postProc
    # caseDir = 'preProcOpt/cornerCases/amp-3.0_freq-1.0'
    # postProc(caseDir, [3, 1])
    
    # from pymooCFD.util.handleData import compressDir
    # compressDir('/dump')
    
    #from pymooCFD.util.handleData import loadCP
    #alg = loadCP()
    #print(alg.pop.get('X'))
    #print(alg.pop.get('F'))
    #alg.display.do(alg.problem, alg.evaluator, alg)
    
    #from pymooCFD.util.handleData import archive
    #archive('dump', background=False)
    
    #from pymooCFD.util.handleData import compressDir
    #compressDir('dump/chackpoint.npy')


if __name__ == '__main__':
    main()
