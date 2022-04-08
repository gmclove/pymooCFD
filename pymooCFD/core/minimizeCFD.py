

class PicklePath:
    def __init__(self, dir_path=None, sub_dirs=[]):
        if dir_path is None:
            dir_path = self.__class__.__name__
        cwd = os.getcwd()
        if cwd in dir_path:
            _, self.relative_path = dir_path.split(cwd)

        self.abs_path = np.abspath(path)
        self.cp_path = os.path.join(self.abs_path, '')


        for i, sub_dir in enumerate(sub_dirs):
            sub_dirs[i] = os.path.join(self.abs_path, sub_dir)
            # os.makedirs(sub_dirs[i]) #, exist_ok=True)
        self.sub_dirs = sub_dirs

    def makePaths(self):
        pass

    def save_self(self):
        self.saveNumpyFile(self.cp_path, self)

    def get_self(self):
        return self.loadNumpyFile(self.cp_path)

    def update_self(self, loaded_self=None):
        if loaded_self is None:
            loaded_self = self.get_self
        self.__dict__.update(loaded_self.__dict__)


    @staticmethod
    def loadNumpyFile(path):
        if not path.endswith('.npy'):
            path = path + '.npy'
        old_path = path.replace('.npy', '.old.npy')
        if os.path.exists(old_path):
            os.rename(old_path, path)
        files = np.load(path, allow_pickle=True).flatten()
        latest_file = files[0]
        return latest_file

    @staticmethod
    def saveNumpyFile(path, data):
        if not path.endswith('.npy'):
            path = path + '.npy'
        temp_path = path.replace('.npy', '.temp.npy')
        old_path = path.replace('.npy', '.old.npy')
        np.save(temp_path, data)
        if os.path.exists(path):
            os.rename(path, old_path)
        os.rename(temp_path, path)
        if os.path.exists(old_path):
            os.remove(old_path)
        return path

from pymooCFD.core.pymooBase import CFDProblem_GA, CFDAlgorithm
from pymooCFD.core.optStudy import OptStudy

algorithm = CFDAlgorithm()

class MinimizeCFD(PicklePath):
    def __init__(self, CFDCase,
                 algorithm=algorithm, #Problem,
                 dir_path=None):
        if dir_path is None:
            dir_path = 'optStudy-'+self.__class__.__name__
        super().__init__(dir_path)
        self.CFDCase = CFDCase
        ###################
        #    OPERATORS    #
        ###################
        from pymoo.factory import get_sampling, get_crossover, get_mutation
        from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

        sampling = MixedVariableSampling(CFDCase.var_type, {
            "real": get_sampling("real_lhs"),  # "real_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(CFDCase.var_type, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })

        mutation = MixedVariableMutation(CFDCase.var_type, {
            "real": get_mutation("real_pm", eta=3.0),
            "int": get_mutation("int_pm", eta=3.0)
        })
        algorithm.sampling = sampling
        algorithm.crossover = crossover
        algorithm.mutation = mutation
        self.algorithm = algorithm

        self.runs = [OptStudy(algorithm, BaseCase, runDir='run00')]



        # self.algorithm = CFDAlgorithm(sampling, crossover, mutation)
