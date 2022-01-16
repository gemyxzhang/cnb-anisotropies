# cnb-anisotropies

This repository contains code that accompanies the papers [Multi-Messenger Astrophysics with the Cosmic Neutrino Background](https://arxiv.org/abs/2103.01274) and [Impact of Warm Dark Matter on the Cosmic Neutrino Background Anisotropies](https://arxiv.org/abs/2201.01888). The `perturbations.c` and `perturbations.h` files contain modifications made to the corresponding [CLASS](http://class-code.net/) files to extract neutrino perturbation information. The notebook file computes and shows, for example, the CNB anisotropies and the corresponding expected neutrino capture rate variations for the PTOLEMY experiment, as shown in the maps below: 
<p float="left">
  <img src="/skymap.png" width="47%" />
  <img src="/ratemap.png" width="47%" /> 
</p>

The `cnb_utils.py` file contains utility functions used to compute CNB anisotropies for both papers. The `wdm_cl.py` and `wdm_pk.py` files contain code that makes the figures for the warm dark matter (WDM) paper. 

**Note**: In the WDM paper, we use synchronous gauge when specifying the CLASS parameters because we noticed that the power spectra in the WDM case in the limit that `m_ncdm` is large deviates significantly from the CDM power spectra in Newtonian gauge when we expect them to agree. In synchronous gauge, however, the results agree well and thus gives us sensible results for the CNB anisotropies. 

## Prerequisites

Install [CO*N*CEPT](https://jmd-dk.github.io/concept/) on to your machine with the command 
```
bash <(wget -O- --no-check-certificate https://raw.githubusercontent.com/jmd-dk/concept/v1.0.1/install)
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
PYTHON=/path/to/concept_installation/python/bin/python3.8 make
```
Note that the `python3.8` in the above command should be replaced with the python version available in your local CO*N*CEPT install. 

To run the `cnb_utils.py`, `figures.ipynb`, `wdm_cl.py` and `wdm_pk.py` files from this repository, place them into `/path/to/concept_installation/class/notebooks/`. To run the notebook file, you can open Jupyter Notebooks by 

```
/path/to/concept_installation/python/bin/jupyter notebook
```

### Remote access 

For running the Jupyter Notebooks on remote clusters, run 
```
/path/to/concept_installation/python/bin/jupyter notebook --no-browser --ip=127.0.0.1 --port=7095
```
and then in a new terminal window on your local machine, run 
```
ssh -N -f -L 7095:localhost:7095 <usernanme>@<hostname>
```
Copying and pasting the URL from the Jupyter server output on the cluster (`http://127.0.0.1:7095/?token=...`) into a local broswer allows you to access the `figures.ipynb` code in your browser. 
