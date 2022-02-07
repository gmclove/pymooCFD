# pymooCFD

## Build Anaconda Environment
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

Create a new conda environment with Python 3.7 installed and numPy.
Next activate the environment, so we can continue the build.  
```bash
conda create -n pymoo-CFD  # python==3.7 numpy
```
```bash
conda activate pymoo-CFD
```
If you already have anaconda installed make sure to update it before continuing.
```bash
conda update conda
```

#### PIP Installation
```bash
conda install pip
```

#### pyMOO Installation
Next install pymoo using pip.
Please see pymoo.org for future installation instructions if modules do not compile and for the latest updates.
Pymoo will run without modules compiled just slower.

```bash
pip install -U pymoo
````

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
This is a GitHub organization containing repositories of conda recipes.
Next install jupyter notebook using the conda-forge channel.
```bash
conda config --add channels conda-forge
```
```bash
conda install -c conda-forge notebook
```
Finally, after making sure you are in the desired directory type the following command to launch jupyter notebook:
```bash
jupyter notebook
```
* IMPORTANT NOTE: When launching jupyter notebook make sure you are not in a directory
below the anaconda3 directory where your conda virtual environment is stored.
This will mean the notebook can not access the virtual environment and will throw an error when importing pymoo package.
Best practice is typically be to launch notebook in the home/user main directory which is where the anaconda3 directory is usually located.

#### Paraview Installation
Paraview is a great tool for CFD post-processing. Paraview also allows for easy automation of post-processing through a python script. It is equipped  with a Python API and a tracing tool that writes a python script for you as you post-processes in the graphic user interface.
```bash
conda install --channel conda-forge paraview
```
To launch paraview simply use the command:
```bash
paraview
```
To start tracing you actions go to Tools > Start Trace. For 'Properties To Trace On Create' selecting 'any *modified* properties' and checking the box next to 'Skip Rendering Components' should give you the script you need to automate your post-processing through python.

In order to transfer data from paraview to your python script use the File > Save Data... feature in paraview to write a data file. A simple modification to the file path in your paraview python script will allow you to save paraview data to a desired location and then load it back into python with a command such as np.loadtxt().

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

### Diagrams installation
```bash
sudo apt-get install graphviz
pip install diagrams
```


#### Other Packages
Any other packages needed should be installed in the conda environment using the command:
```bash
conda install -c <channel> <package>
```
where the channels we have set up so far are 'conda-forge' or the default 'anaconda'.
The package could be something like 'h5py' which is used for post-processing YALES2 simulations.
```bash
conda install h5py
```

Remember that the pymoo-CFD environment should be active during this build.

## Workflow
### Setup Optimization Study
- Copy template case

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
