# @Author: glove
# @Date:   2021-12-15T14:59:21-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:22-05:00

# from pymooCFD.core.cfdCase import CFDCase
import config
import logging
import os
import copy
import numpy as np

from pymoo.visualization.scatter import Scatter

from pymooCFD.util.loggingTools import MultiLineFormatter
from pymooCFD.util.handleData import saveTxt


class MeshStudy: #(CFDCase):
    def __init__(self, cfdCase,
                 size_factors = np.around(np.arange(0.5, 1.5, 0.1), decimals=2)
                 ):
        super().__init__()
        self.logger = getLogger()
        self.baseCase = cfdCase
        self.size_factors = size_factors
        self.cases = None #[]
        self.folder = os.path.join(cfdCase.caseDir, 'meshStudy')

    def gen(self):
        if self.size_factors is None:
            self.logger.warning(
                'self.size_factors is None but self.genMeshStudy() called')
            return
        self.logger.info('\tGENERATING MESH STUDY . . .')
        self.logger.info(f'\t\tFor Mesh Size Factors: {self.size_factors}')
        # Pre-Process
        study = []
        var = []
        self.cases = []
        for sf in self.size_factors:
            msCase = copy.deepcopy(self.baseCase)
            self.cases.append(msCase)
            fName = f'meshSF-{sf}'
            path = os.path.join(self.folder, fName)
            self.logger.info(f'\t\tInitializing {path} . . .')
            msCase.__init__(path, self.baseCase.x, meshSF=sf)
            msCase.meshStudy = None
            if msCase.meshSF != sf or msCase.numElem is None:
                # only pre-processing needed is generating mesh
                msCase.meshSF = sf
                msCase.genMesh()  # NOT NESSECESARY BECAUSE FULL PRE-PROCESS DONE AT RUN
            else:
                self.logger.info(
                    f'\t\t\t{msCase} already has number of elements: {msCase.numElem}')
            # sfToElem.append([msCase.meshSF, msCase.numElem])
            saveTxt(msCase.caseDir, 'numElem.txt', [msCase.numElem])
            study.append([msCase.caseDir, str(
                msCase.numElem), str(msCase.meshSF)])
            # var.append(msCase.x)
        study = np.array(study)
        saveTxt(self.folder, 'study.txt', study, fmt="%s")
        # Data
        dat = np.array([[case.meshSF, case.numElem]
                        for case in self.msCases])
        # Print
        with np.printoptions(suppress=True):
            self.logger.info(
                '\tMesh Size Factor | Number of Elements\n\t\t' + str(dat).replace('\n', '\n\t\t'))
            saveTxt(self.folder, 'meshSFs-vs-numElem.txt', dat)

        self.saveCP()

    def run(self):
        if self.size_factors is None:
            self.logger.error(
                'EXITING MESH STUDY: Mesh Size Factors set to None. May be trying to do mesh study on a mesh study case.')
            return
        # if meshSFs is None:
        #     meshSFs = self.meshSFs
        # if self.msCases is None:
        #     self.genMeshStudy()
        self.logger.info(f'MESH STUDY')
        if self.cases is None:
            self.logger.info('\tNo Mesh Cases Found: self.cases is None')
            self.logger.info(f'\t {self.size_factors}')
        else:
            prev_meshSFs = [case.meshSF for case in self.cases]
            self.logger.info(
                f'\tCurrent Mesh Size Factors:\n\t\t{self.size_factors}')
            self.logger.info(
                f'\tPrevious Mesh Study Size Factors:\n\t\t{prev_meshSFs}')
            if all(sf in prev_meshSFs for sf in self.size_factors):
                self.logger.info(
                    '\tALL CURRENT MESH SIZE FACTORS IN PREVIOUS MESH SIZE FACTORS')
                for msCase in self.cases:
                    incomp_cases = []
                    if msCase.f is None or np.isnan(np.sum(msCase.f)):
                        self.logger.info(f'INCOMPLETE: Mesh Case - {msCase}')
                        self.logger.debug('\t msCase.f has None or NaN value')
                        incomp_cases.append(msCase)
                    self.logger.info('RUNNING: Incomplete mesh study cases')
                    self.baseCase.parallelize(incomp_cases)
                self.plot()
                self.logger.info('SKIPPED: MESH STUDY')
                return
            else:
                self.logger.info(
                    '\t\tOLD MESH SIZE FACTORS != NEW MESH SIZE FACTORS')
        # if not restart or self.msCases is None:
        #     self.genMeshStudy()
        # else:
        #     print('\tRESTARTING MESH STUDY')
        # else:genMeshStudy
        #     self.msCases =
        self.gen()
        # Data
        dat = np.array([[case.meshSF, case.numElem]
                        for case in self.cases])
        # Print
        with np.printoptions(suppress=True):
            self.logger.info(
                '\tMesh Size Factor | Number of Elements\n\t\t' + str(dat).replace('\n', '\n\t\t'))
            saveTxt(self.folder, 'numElem-vs-meshSFs.txt', dat)

        self.exec()
        self.plot()

    def plot(self):
        self.logger.info('\tPLOTTING MESH STUDY')
        _, tail = os.path.split(self.baseCase.caseDir)
        a_numElem = np.array([case.numElem for case in self.cases])
        a_sf = np.array([case.meshSF for case in self.cases])
        msObj = np.array([case.f for case in self.cases])
        solnTimes = np.array([case.solnTime for case in self.cases])
        # Plot
        # number of elements vs time
        plot = Scatter(title='Mesh Study: ' + tail, legend=True, grid=True,
                       labels=['Number of Elements', 'Solution Time [s]'],
                       tight_layout=True
                       )
        for i in range(len(a_numElem)):
            pt = np.array([a_numElem[i], solnTimes[i]])
            plot.add(pt, label=a_sf[i], marker='o', linestyle="-")
        plot.do()
        plot.ax.legend(title='Mesh Size Factor',
                       bbox_to_anchor=(1.01, 1.0))
        # plot.ax.get_legend().set_title('Mesh Size Factors')
        fName = f'ms_plot-{tail}-numElem_v_time.png'
        fPath = os.path.join(self.folder, fName)
        plot.save(fPath, dpi=100)
        for obj_i, obj_label in enumerate(self.baseCase.obj_labels):
            # Number of elements vs Objective
            plot = Scatter(title='Mesh Study: ' + tail, legend=True, grid=True,
                           labels=['Number of Elements', obj_label],
                           tight_layout=True
                           )
            for i in range(len(a_numElem)):
                pt = np.array([a_numElem[i], msObj[i, obj_i]])
                plot.add(pt, label=a_sf[i], marker='o', linestyle="-")
            plot.do()
            # plot.ax.get_legend().set_title('Mesh Size Factors')
            plot.ax.legend(title='Mesh Size Factor',
                           bbox_to_anchor=(1.01, 1.0))
            fName = f'ms_plot-{tail}-obj{obj_i}.png'
            fPath = os.path.join(self.folder, fName)
            plot.save(fPath, dpi=100)

            # Time vs Objective
            plot = Scatter(title='Mesh Study: ' + tail, legend=True, grid=True,
                           labels=['Solution Time [s]', obj_label],
                           tight_layout=True
                           )
            for i in range(len(a_numElem)):
                pt = np.array([solnTimes[i], msObj[i, obj_i]])
                plot.add(pt, label=a_sf[i], marker='o', linestyle="-")
            plot.do()
            plot.ax.legend(title='Mesh Size Factor',
                           bbox_to_anchor=(1.01, 1.0))
            # plot.ax.get_legend().set_title('Mesh Size Factors')
            fName = f'ms_plot-{tail}-solnTime_v_obj{obj_i}.png'
            fPath = os.path.join(self.folder, fName)
            plot.save(fPath, dpi=100)

            # Number of Elements vs Objective vs time
            plot = Scatter(title='Mesh Study: ' + tail, legend=True, grid=True,
                           labels=['Number of Elements', obj_label, 'Solution Time [s]'],
                           tight_layout=True, bbox_to_anchor=(1.05, 1.0)
                           )
            for i in range(len(a_numElem)):
                pt = np.array([a_numElem[i], msObj[i, obj_i], solnTimes[i]])
                plot.add(pt, label=a_sf[i], marker='o', linestyle="-")
            plot.do()
            plot.ax.legend(title='Mesh Size Factor',
                           bbox_to_anchor=(1.01, 1.0))
            fName = f'ms_plot-{tail}-numElem_v_obj{obj_i}_v_time.png'
            fPath = os.path.join(self.meshStudyDir, fName)
            plot.save(fPath, dpi=100)
        self.saveCP()
