import numpy as np
import os
import logging

class PyClassDB:
    def __init__(self, PyClass, location=None,
                 char_exceptions='._- ',
                 default_char='_'
                 ):
        if location is None:
            location = 'PyDB-' + self.__class__.__name__
        self.logger = logging.getLogger()
        self.location = os.path.abspath(location)
        os.makedirs(self.location, exist_ok=True)
        self.PyClass = PyClass
        self.char_exceptions = char_exceptions
        self.default_char = default_char
        self.logger.info('PyClassDB initialized')

    def objToFileName(self, obj):
        s = self._objToFileName(obj)  # get database specific file name
        fName = ''.join([c if (c.isalnum() or c in self.char_exceptions)
                        else self.default_char for c in s])
        if not fName.endswith('.npy'):
            fName = fName + '.npy'
        return fName

    def objToFilePath(self, obj):
        fName = self.objToFileName(obj)
        return os.path.join(self.location, fName)

    def exists(self, obj):
        fPath = self.objToFilePath(obj)
        if os.path.exists(fPath):
            return True

    def load(self, obj):
        if self.exists(obj):
            fPath = self.objToFilePath(obj)
            loaded_obj = self.loadNumpyFile(fPath)
            if self.objToFileName(obj) == self.objToFileName(loaded_obj):
                return loaded_obj
            else:
                self.logger.error('object -> file name != loaded object -> file name')

    def remove(self, obj):
        fPath = self.objToFilePath(obj)
        try:
            os.remove(fPath)
        except FileNotFoundError as err:
            self.logger.error(err)

    def save(self, obj):
        if type(obj) == self.PyClass:
            fPath = self.objToFilePath(obj)
            self.saveNumpyFile(fPath, obj)
        else:
            self.logger.error(
                f'PyDB: {obj} not of type {self.PyClass}')

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

    def _objToFileName(self, obj):
        fName = ''
        dictonary = obj.__dict__
        for i, (key, val) in enumerate(dictonary.items()):
            if i != len(dictonary)-1:
                fName += key + '-' + str(val) + '_'
            else:
                fName += key + '-' + str(val)
        return fName


class CFDCaseDB(PyClassDB):
    def __init__(self, CFDCase, location, precision=30):
        super().__init__(CFDCase, location)
        self.precision = precision
        cp_path = os.path.join(self.location, 'cfdCaseDB.npy')
        self.saveNumpyFile(cp_path, self)

    def stochasticSave(self, obj):
        pass

    def _objToFileName(self, obj):
        vars = obj.x
        with np.printoptions(precision=self.precision, suppress=True, floatmode='fixed'):
            fName = str(vars).replace(
                '[', '').replace(']', '').replace(' ', '_')
        return fName

#
# illegal_symbols = ['\', '//', '\#', '<', '&', '%', '*', '$', '+', '>', '!', '?', ',
#                     '|', '`', """, '=', ':', '@', '{', '}', ]
# from dataclasses import dataclass
# from typing import Any
#
#
# @dataclass
# class PyObjFile:
#     name: str
#     obj: Any
#     key: Any
#     val: Any
#     char_exceptions: str = '._- '
#     default_char: str = '_'
#
#     @property
#     def name(self, name):
#         return ''.join(c for c in name
#                         if (c.islaum() or c in self.char_exceptions)
#                         else self.default_char
#                         )
#
#     def __str__(self):
#         return f'{self.name}-{self.key}:{self.val}'
#
#     def __eq__(self, other):
#         if other.__class__ is not self.__class__:
#             return NotImplemented
#         return (self.key, self.val) == (other.key, other.val)
