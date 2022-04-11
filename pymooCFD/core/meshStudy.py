# @Author: glove
# @Date:   2021-12-15T14:59:21-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T09:37:22-05:00

# from pymooCFD.core.cfd_case import CFDCase
from pymooCFD import config
import logging
import os
import copy
import numpy as np

from pymoo.visualization.scatter import Scatter

from pymooCFD.util.loggingTools import MultiLineFormatter
from pymooCFD.util.handleData import saveTxt


class MeshStudy: #(CFDCase):
    def __init__(self, cfd_case,
                 size_factors = np.around(np.arange(0.5, 1.5, 0.1), decimals=2)
                 ):
        super().__init__()
        self.folder = os.path.join(cfd_case.abs_path, 'meshStudy')
        os.mkdir(self.folder)
        # os.makedirs(self.folder,)
        self.logger = self.getLogger()
        self.base_case = cfd_case
        self.cases = None #[]
        self.size_factors = size_factors
        self.logger.info('INITIALIZED: Mesh Study')


    def run(self):
        if self.size_factors is None:
            self.logger.error(
                'EXITING MESH STUDY: Mesh Size Factors set to None. May be trying to do mesh study on a mesh study case.')
            return
        # if size_factors is None:
        #     size_factors = self.size_factors
        # if self.msCases is None:
        #     self.genMeshStudy()
        self.logger.info(f'MESH STUDY')
        if self.cases is None:
            self.logger.info('\tNo Mesh Cases Found: self.cases is None')
            self.logger.info(f'\t {self.size_factors}')
        else:
            prev_size_factors = [case.meshSF for case in self.cases]
            self.logger.info(
                f'\tCurrent Mesh Size Factors:\n\t\t{self.size_factors}')
            self.logger.info(
                f'\tPrevious Mesh Study Size Factors:\n\t\t{prev_size_factors}')
            if all(sf in prev_size_factors for sf in self.size_factors):
                self.logger.info(
                    '\tALL CURRENT MESH SIZE FACTORS IN PREVIOUS MESH SIZE FACTORS')
                for msCase in self.cases:
                    incomp_cases = []
                    if msCase.f is None or np.isnan(np.sum(msCase.f)):
                        self.logger.info(f'INCOMPLETE: Mesh Case - {msCase}')
                        self.logger.debug('\t msCase.f has None or NaN value')
                        incomp_cases.append(msCase)
                    self.logger.info('RUNNING: Incomplete mesh study cases')
                    self.base_case.parallelize(incomp_cases)
                self.plot()
                self.logger.info('SKIPPED: MESH STUDY')
                return
            else:
                self.logger.info(
                    '\t\tOLD MESH SIZE FACTORS != NEW MESH SIZE FACTORS')
        self.gen()
        # Data
        dat = np.array([[case.meshSF, case.numElem]
                        for case in self.cases])
        # Print
        with np.printoptions(suppress=True):
            self.logger.info(
                '\tMesh Size Factor | Number of Elements\n\t\t' + str(dat).replace('\n', '\n\t\t'))
            saveTxt(self.folder, 'numElem-vs-size_factors.txt', dat)

        self.exec()
        self.plot()

    def gen(self):
        if self.size_factors is None:
            self.logger.warning(
                'self.size_factors is None but self.gen() called')
            return
        self.logger.info('\tGENERATING MESH STUDY . . .')
        self.logger.info(f'\t\tFor Mesh Size Factors: {self.size_factors}')
        # Pre-Process
        study = []
        var = []
        self.cases = []
        for sf in self.size_factors:
            msCase = copy.deepcopy(self.base_case)
            self.cases.append(msCase)
            fName = f'meshSF-{sf}'
            path = os.path.join(self.folder, fName)
            self.logger.info(f'\t\tInitializing {path} . . .')
            msCase.__init__(path, self.base_case.x, meshSF=sf)
            msCase.meshStudy = None
            if msCase.meshSF != sf or msCase.numElem is None:
                # only pre-processing needed is generating mesh
                msCase.meshSF = sf
                msCase.genMesh()  # NOT NESSECESARY BECAUSE FULL PRE-PROCESS DONE AT RUN
            else:
                self.logger.info(
                    f'\t\t\t{msCase} already has number of elements: {msCase.numElem}')
            # sfToElem.append([msCase.meshSF, msCase.numElem])
            saveTxt(msCase.abs_path, 'numElem.txt', [msCase.numElem])
            study.append([msCase.abs_path, str(
                msCase.numElem), str(msCase.meshSF)])
            # var.append(msCase.x)
        study = np.array(study)
        saveTxt(self.folder, 'study.txt', study, fmt="%s")
        # Data
        dat = np.array([[case.meshSF, case.numElem]
                        for case in self.cases])
        # Print
        with np.printoptions(suppress=True):
            self.logger.info(
                '\tMesh Size Factor | Number of Elements\n\t\t' + str(dat).replace('\n', '\n\t\t'))
            saveTxt(self.folder, 'size_factors-vs-numElem.txt', dat)

        self.base_case.saveCP()

    def exec(self):
        self.logger.info('\tEXECUTING MESH STUDY')
        # self.logger.info(f'\t\tPARALLELIZING:\n\t\t {self.msCases}')
        self.base_case.parallelize(self.cases)
        obj = np.array([case.f for case in self.cases])
        self.logger.info('\tObjectives:\n\t\t' +
                         str(obj).replace('\n', '\n\t\t'))
        self.base_case.saveCP()

    def plot(self):
        self.logger.info('\tPLOTTING MESH STUDY')
        _, tail = os.path.split(self.base_case.abs_path)
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
        for obj_i, obj_label in enumerate(self.base_case.obj_labels):
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
            fPath = os.path.join(self.folder, fName)
            plot.save(fPath, dpi=100)
        self.base_case.saveCP()

    ################
    #    LOGGER    #
    ################
    def getLogger(self):
        _, tail = os.path.split(self.folder)
        # Get Child Logger using hierarchical "dot" convention
        logger = logging.getLogger(__name__ + '.' + self.folder)
        logger.setLevel(config.CFD_CASE_LOGGER_LEVEL)
        # Filters
        # Filters added to logger do not propogate up logger hierarchy
        # Filters added to handlers do propogate
        # filt = DispNameFilter(self.abs_path)
        # logger.addFilter(filt)
        # File Handle
        logFile = os.path.join(self.folder, f'{tail}.log')
        fileHandler = logging.FileHandler(logFile)
        logger.addHandler(fileHandler)
        # Stream Handler
        # parent root logger takes care of stream display
        # Formatter
        formatter = MultiLineFormatter(
            '%(asctime)s :: %(levelname)-8s :: %(name)s :: %(message)s')
        fileHandler.setFormatter(formatter)
        # Initial Message
        logger.info('-' * 30)
        logger.info('LOGGER INITIALIZED')
        # Plot logger
        plot_logger = logging.getLogger(Scatter.__name__)
        plot_logger.setLevel(config.PLOT_LOGGER_LEVEL)
        return logger


    @property
    def size_factors(self): return self._size_factors

    @size_factors.setter
    def size_factors(self, size_factors):
        if size_factors is None:
            self._size_factors = size_factors
            return
        size_factors, counts = np.unique(size_factors, return_counts=True)
        for sf_i, n_sf in enumerate(counts):
            if n_sf > 1:
                self.logger.warning(
                    f'REPEATED MESH SIZE FACTOR - {size_factors[sf_i]} repeated {n_sf} times')

        if self.cases is None:
            self._size_factors = size_factors
        else:
            prev_size_factors = [case.meshSF for case in self.cases]
            self.logger.debug(f'Current Mesh Size Factors:\n\t{self.size_factors}')
            self.logger.debug(
                f'Previous Mesh Study Size Factors:\n\t{prev_size_factors}')
            if all(sf in prev_size_factors for sf in self.size_factors):
                self.logger.debug(
                    'ALL CURRENT MESH SIZE FACTORS IN PREVIOUS MESH SIZE FACTORS')
                self._size_factors = size_factors
            else:
                self.logger.debug(
                    'OLD MESH SIZE FACTORS != NEW MESH SIZE FACTORS')
                self._size_factors = size_factors
                self.gen()
