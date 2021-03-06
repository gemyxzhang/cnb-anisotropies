# cnb-anisotropies

This repository contains code that accompanies the paper [Multi-Messenger Astrophysics with the Cosmic Neutrino Background](https://arxiv.org/abs/2103.01274). The `perturbations.c` and `perturbations.h` files contain modifications made to the corresponding [CLASS](http://class-code.net/) files to extract neutrino perturbation information. 

## Prerequisites

Install [CO*N*CEPT](https://jmd-dk.github.io/concept/) on to your machine with the command 
```
bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/master/installer)
```
Then in the `/path/to/concept_installation/python/bin/` directory, run 
```
./pip install notebook 
./pip install healpy
```

## Usage 

To download the code in this repository to your local machine, do 
```
git clone https://github.com/gemyxzhang/cnb-anisotropies
``` 
Then replace the files at `/path/to/concept_installation/class/source/perturbations.c` and `/path/to/concept_installation/class/include/perturbations.h` with the `perturbations.c` and `perturbations.h` files downloaded from this repository. 

To recompile CO*N*CEPT with the new perturbations files, do 
```
cd /path/to/concept_installation/class 
make clean 
PYTHON=../python/bin/python3.8 make
```
Note that the `python3.8` in the above command should be replaced with the python version available in your local CO*N*CEPT install. 

To run the `cnb_utils.py` and `figures.ipynb` files from this repository, place them into `/path/to/concept_installation/class/notebooks/`. Open Jupyter Notebooks by 

```
/path/to/concept_installation/python/bin/jupyter notebook
```

### Remote access 

For running the Jupyter Notebooks on remote clusters, run 
```
/path/to/concept_installation/python/bin/jupyter notebook --no-browser --ip=0.0.0.0 --port=8080
```
and then in a new terminal window on your local machine, run 
```
ssh -N -f -L localhost:1234:localhost:8080 <usernanme>@<hostname>
```
Pasting the link from the Jupyter server output on the cluster into a local broswer allows you to access the `figures.ipynb` code in your browser. 
