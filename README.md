# pymooCFD

## Installing Anaconda
To best utilize this package install either miniconda or anaconda.
Anaconda is a larger more inclusive distribution of program packages
while miniconda is a more light weight distribution.

See here for further details: https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda

Install here: https://docs.anaconda.com/anaconda/install/

If you choose miniconda there may be python packages not mentioned here that will need to be installed.

## Build Conda Environment
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

If you already have anaconda installed make sure to update it before continuing.
```bash
conda update conda
```
Create a new conda environment with Python 3.9 installed and numPy.
Next activate the environment, so we can continue the build.  
```bash
conda create -n pymooCFD -y python==3.9 numpy
```
```bash
conda activate pymooCFD
```


#### PIP Installation
This package should come with the base anaconda python environment but to make sure execute command:
```bash
conda install pip
```

#### pymoo Installation
Next install pymoo using pip.

```bash
pip install -U pymoo
````
Please see pymoo.org for the latest installation instructions if modules do not compile and for the latest updates.
Pymoo will run without modules compiled just slower.

https://pymoo.org/installation.html

#### GMSH Installation
GMSH is the recommended meshing software for use with pymoo-CFD.
GMSH offers a well-developed python application programming interface (API).
This makes communication between pymoo and the meshing software straightforward
when setting up optimization problems with geometric parameters that require re-meshing the domain.

The latest version of GMSH can be install using pip or directly from gmsh.info homepage.  
```bash
pip install --upgrade gmsh
```
Please refer to gmsh.info for future installation instructions and the latest updates on the software.
As with pymoo this software is actively being developed.

http://gmsh.info/

#### Jupyter Notebook Installation
Jupyter Notebook offers an excellent way to pre/post-process the optimization problems
and CFD simulations.

First add conda-forge as a channel on anaconda.
Conda-forge is a GitHub organization containing repositories of conda recipes.
Next install jupyter notebook using the conda-forge channel.
```bash
conda config --add channels conda-forge
```
```bash
conda install -c conda-forge notebook
```
Finally, type the following command to launch jupyter notebook:
```bash
jupyter notebook
```
To add some helpful extensions to jupyter notebook run the following command:
```bash
conda install nb_conda
```

* IMPORTANT NOTE: When launching jupyter notebook make sure you are not in a directory
below the *anaconda3* directory where your conda virtual environment is stored.
This will mean the notebook can not access the virtual environment and will throw an error when importing the pymoo package.
Best practice is typically be to launch notebook in the home or user main directory which is where the *anaconda3* directory is usually located.

#### Paraview Installation
Paraview is a great tool for CFD post-processing. Paraview also allows for easy automation of post-processing through a python script. It is equipped  with a Python API and a tracing tool that writes a python script for you as you post-processes in the graphic user interface.

Please note that the paraview conda package is the most prone to conflicts with
other program packages so please install this package towards the end of your
virtual environment build.
```bash
conda install --channel conda-forge paraview
```
To launch paraview simply use the command:
```bash
paraview
```
To start tracing you actions go to Tools > Start Trace. For 'Properties To Trace On Create' selecting 'any *modified* properties' and checking the box next to 'Skip Rendering Components' should give you the script you need to automate your post-processing through python.

In order to transfer data from paraview to your python script use the File > Save Data... feature in paraview to write a data file. A simple modification to the file path in your paraview python script will allow you to save paraview data to a desired location and then load it back into python with a command such as numpy.loadtxt().

https://www.paraview.org/
https://anaconda.org/conda-forge/paraview

<!--
#### Option 2:
Not robust, had problems. May require re-running python ipykernel install after changes to kernel

Finally, install ipykernel and set up the pymoo-CFD conda environment as a kernel in Jupyter Notebook.
This will allow you to use the pymoo-CFD environment for your post-processing python notebooks.
```bash
conda install -c anaconda ipykernel
```
```bash
python -m ipykernel install --user --name=pymoo-CFD
```
To launch Jupyter Notebook with conda environemnt configured use the command:
```bash
jupyter notebook
``` -->

### Dask installation
Dask is a great tool for distributive computing of python code.
This means if you are using a python CFD solver this is likely your best tool for
distributing the CFD solver computing across a single machine or high performance computing clusters.

On high performance computing clusters with a job scheduler already in use such as SLURM or PBS use the following installation for dask-jobqueue and it's dependents.
```bash
conda install dask-jobqueue -c conda-forge
```
For use on a single machine Dask.Distributed can be used instead.
```bash
conda install dask distributed -c conda-forge
```
For interactive monitoring of your Dask jobs install the Jupyter Qt Console.
```bash
conda install qtconsole
```
<!--
### Diagrams installation
```bash
sudo apt-get install graphviz
pip install diagrams
```
-->

### Other Packages
Any other packages needed should be installed in the conda environment using the command:
```bash
conda install -c <channel> <package>
```
where the channels we have set up so far are 'conda-forge' or the default 'anaconda'.
The package could be something like 'h5py' which is used for post-processing YALES2 simulations.
```bash
conda install h5py
```

Remember that the pymooCFD environment should be active during this build.

## Workflow
### Setup Optimization Study
- Think through your optimization study. What scripting of the CFD meshing and
solving schemes will be necessary? What programs will need to interact to execute
my CFD model from start to finish?
GMSH is the recommended mesher if one is not already scripted. If there are no
geometric parameters then it could be easier to forgo the scripting of the mesh
and conduct mesh studies manually. There are many factors to consider at the
beginning of an optimization study that if properly thought through can same a
lot of trouble. Start with a simple case to practice.

- Create a base model if you don't already have one. This is the model will
later be parameterized.
Once a base model is running the user should manually adjust the desired optimization
parameters. A brief manual search of the parameter space will give the user an
idea of the challenges they will face is creating a modeling scheme that is robust enough
to give viable results throughout the search space.

- Create a subclass of CFDCase (further subclasses of FluentCase(CFDCase) or YALES2Case(CFDCase) work too).
This class will override the _preProc(self) and _postProc(self) methods where
the pre-processing and post-processing scripts will be executed. If a python CFD
solver is being used then the _solve(self) will also have to be overridden.


- Replace contents of base_case folder with all necessary execution files for base simulation.
  - It is important to know where the parameter values you wish to optimize are located in these files. Ideally the CFD solver uses an input file where all the parameters are located.

- Open setupCFD.py and setup pre/post-processing that will be performed every time an individual simulation is run.
  - Pre-processing involves taking in variables pass from setupOpt.py's _evaluate() method in the custom GA_CFD() class.

```python3
def preProc(caseDir, var, jobName=jobName, jobFile=jobFile):
  # OPEN INPUT FILE, EDIT PARAMETERS USING var
  # OPTIONAL jobName AND jobFile INPUTS FOR SLURM EXECUTION SYSTEM
    # USE OF SLURM SYSTEM REQUIRES EDITING DIRECTORY IN FILE THAT LAUNCHES SLURM JOB
```

- Open setupOpt.py and edit the variables at the top of the file.
  - n_gen: number of generations before termination
  - n_var: number of variables   
  - xu: upper limits of variables   
  - n_obj: number of objectives

    Define your design space (the variables/parameters being studied).
  Define your number of objectives (n_obj) and the number of constraints on the objective space.
  If there are constraints on the objective space make sure to define them as functions that are less than 0 when constraint is violated.
  See https://pymoo.org/getting_started.html for more details on constraints including normalization procedure.

- Before queuing your slurm job (or however you are running pymooExec.py) run 'python makeClean.py' from inside the case directory.
  - makeClean.py: removes files generated by previous pyMOO runs, cleans up base_case folder and changes jobslurm.sh working directory, job-name and #SBATCH --nodes
    using the input from pymooIN.py


### Restart Case
Most likely there will be many errors to work when implementing a new pyMOO-CFD study.
Make sure to run 'python makeClean.py' from inside your case directory to remove all files generated by previous run.
makeClean.py will also adjust

WARNING: population size will not update when restarting from checkpoint.npy


# References

Thesis Advisor: Dr. Yves Dubief, University of Vermont, https://scholar.google.com/citations?user=nE7d3OkAAAAJ&hl=en


PyMOO - Multi-objective Optimization in Python
https://pymoo.org

@ARTICLE{pymoo,
    author={J. {Blank} and K. {Deb}},
    journal={IEEE Access},
    title={Pymoo: Multi-Objective Optimization in Python},
    year={2020},
    volume={8},
    number={},
    pages={89497-89509},
}


https://www.coria-cfd.fr/index.php/YALES2
