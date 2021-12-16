# @Author: glove
# @Date:   2021-12-15T14:59:23-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-15T17:20:34-05:00

from pymooCFD.core.optStudy import OptStudy

class LocalCompDistOpt(OptStudy):

    def __init__(self):
        super().__init__()

    def execute(self):
        self.singleNodeExec()


from pymooCFD.core.cfdCase import CFDCase

class YALES2LocalCompDist(CFDCase):
    n_var = 2
    var_labels = ['Number of Processors', 'Time Step']

    n_obj = 2
    obj_labels = ['Wall Time', 'Fidelity']

    n_constr = 0

    nProc = 8

    def __init__(self, baseCaseDir, caseDir, x,
                 *args, **kwargs):
        super().__init__(baseCaseDir, caseDir, x,
                         *args, **kwargs)

    def _preProc(self):
        self.nProc = self.x[0]

    def _postProc(self):
        pass
