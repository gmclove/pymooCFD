import os
import shutil
import numpy as np


def saveTxt(path, fname, data, **kwargs):
    data = np.array(data)
    if not data.shape:
        data = [data]
    datFile = os.path.join(path, fname)
    # save data as text file in directory
    np.savetxt(datFile, data, **kwargs)


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def makeDir(path):
    try:
        os.mkdir(path)
        print(f'empty directory created: {path}')
    except OSError as err:
        print(err)
        pass


def removeDir(path):
    print(f'removing {path}..')
    try:
        shutil.rmtree(path)
        print(f"{path} removed successfully")
    except OSError as err:
        print(err)


def emptyDir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)
