# Solar-Forcing

This repository and package can be used to generate solar forcing datasets to be used for the Community Earth System Model (CESM)

## Installation

For now, the two main steps to install this environment/package are the following

### Step 0 - clone this repository

Run
```
git clone https://github.com/NCAR/solar-forcing.git
```

### Install your conda environment

Type the following into your terminal from the main directory for this repository

```
conda env create -f envs/environment.yml
```

then 

```
conda activate solar-forcing-dev
```

### Install the package
Once you are in that environment, go ahead install the package!

```
pip install -e .
```

There ya go! You can now call
```
jupyterlab
```

which will fire up a JupyterLab window on your computer, where you can into the `notebooks` directory to interact with the examples.
