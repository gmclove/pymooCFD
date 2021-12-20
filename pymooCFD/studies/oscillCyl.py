# @Author: glove
# @Date:   2021-12-14T16:02:45-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T16:36:54-05:00

import os
import h5py
import numpy as np
from scipy.integrate import quad

from pymooCFD.util.yales2Tools import getLatestXMF
from pymooCFD.core.cfdCase import CFDCase
class OscillCylinder(CFDCase):
    ####### Define Design Space #########
    n_var = 2
    var_labels = ['Amplitude', 'Frequency']
    varType =    ["real", "real"]  # options: 'int' or 'real'
    xl =         [0.1, 0.1]  # lower limits of parameters/variables
    xu =         [3.0, 1]  # upper limits of variables
    if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
        raise Exception("Design Space Definition Incorrect")
    ####### Define Objective Space ########
    obj_labels = ['Drag on Cylinder', 'Power Input']
    n_obj = 2
    ####### Define Constraints ########
    n_constr = 0
    ##### Local Execution Command #####
    nProc = 8
    solverExecCmd = ['mpirun', '-np', str(nProc), '2D_cylinder']

    def __init__(self, baseCaseDir, caseDir, x):
        super().__init__(baseCaseDir, caseDir, x,
                        meshFile = '2D_cylinder.msh22',
                        datFile = 'ics_temporals.txt',
                        jobFile = 'jobslurm.sh',
                        inputFile = '2D_cylinder.in',
                        )

    def _preProc_restart(self):
        self._preProc()
        dumpDir = os.path.join(self.caseDir, 'dump')
        latestXMF = getLatestXMF(dumpDir)
        in_lines = self.inputLines
        kw = 'RESTART_TYPE = GMSH'
        kw_line, kw_line_i = self.findKeywordLine(kw, in_lines)
        in_lines[kw_line_i] = '#' + kw + '\n'
        kw = "RESTART_GMSH_FILE = '2D_cylinder.msh22'"
        kw_line, kw_line_i = self.findKeywordLine(kw, in_lines)
        in_lines[kw_line_i] = '#' + kw + '\n'
        kw = "RESTART_GMSH_NODE_SWAPPING = TRUE"
        kw_line, kw_line_i = self.findKeywordLine(kw, in_lines)
        in_lines[kw_line_i] = '#' + kw + '\n'
        in_lines.append('RESTART_TYPE = XMF' + '\n')
        in_lines.append('RESTART_XMF_SOLUTION = dump/' + latestXMF + '\n')
        self.inputLines = in_lines

    def _preProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        amp = self.x[0]
        freq = self.x[1]
        ####### SIMULATION INPUT PARAMETERS #########
        # open and read YALES2 input file to array of strings for each line
        in_lines = self.inputLines
        # find line that must change using a keyword
        keyword = 'CYL_ROTATION_PROP'
        keyword_line, keyword_line_i = self.findKeywordLine(keyword, in_lines)
        # create new string to replace line
        newLine = f'{keyword_line[:keyword_line.index("=")]}= {amp} {freq} \n'
        in_lines[keyword_line_i] = newLine
        # REPEAT FOR EACH LINE THAT MUST BE CHANGED
        self.inputLines = in_lines

    def _postProc(self):
        ####### EXTRACT VAR ########
        # Extract parameters for each individual
        amp = self.x[0]
        freq = self.x[1]
        ######## Compute Objectives ##########
        ######## Objective 1: Drag on Cylinder #########
        U = 1
        rho = 1
        D = 1
        # create string for directory of individual's data file
        data = np.genfromtxt(self.datPath, skip_header=1)
        # try:
        #     data = np.genfromtxt(dataDir, skip_header=1)
        # except IOError as err:
        #     print(err)
        #     print('ics_temporals.txt does not exist')
        #     obj = [None] * n_obj
        #     return obj

        # collect data after 100 seconds of simulation time
        mask = np.where(data[:,1] > 100)
        # Surface integrals of Cp and Cf
        # DRAG: x-direction integrals
        # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
        p_over_rho_intgrl_1 = data[mask, 4]
        tau_intgrl_1 = data[mask, 6]
        F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
        C_drag = F_drag/((1/2)*rho*U**2*D**2)

        ######## Objective 2 #########
        # Objective 2: Power consumed by rotating cylinder
        D = 1  # [m] cylinder diameter
        t = 0.1  # [m] thickness of cylinder wall
        r_o = D/2  # [m] outer radius
        r_i = r_o-t  # [m] inner radius
        d = 2700  # [kg/m^3] density of aluminum
        L = 1  # [m] length of cylindrical tube
        V = L*np.pi*(r_o**2-r_i**2) # [m^3] volume of cylinder
        m = d*V # [kg] mass of cylinder
        I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
        P_cyc = 0.5*I*quad(lambda t : (amp*np.sin(t))**2, 0, 2*np.pi)[0]*freq  # [Watt]=[J/s] average power over 1 cycle

        obj = [C_drag, P_cyc]
        self.logger.info(f'{self.caseDir}: {obj}')
        return obj

from pymooCFD.core.optStudy import OptStudy
class OscillCylinderOpt(OptStudy):
    def __init__(self, algorithm, problem, baseCase,
                *args, **kwargs):
        super().__init__(algorithm, problem, baseCase,
                        baseCaseDir = 'osc-cyl_base',
                        optDatDir = 'cyl-opt_run',
                        *args, **kwargs)

    def execute(self, cases):
        self.singleNodeExec(cases)

MyOptStudy = OscillCylinderOpt
BaseCase = OscillCylinder
