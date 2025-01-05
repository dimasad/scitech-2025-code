# scitech-2025-code
Code for paper "Variational System Identification of Aircraft",
presented in AIAA SciTech 2025, https://arc.aiaa.org/doi/10.2514/6.2025-1253

Instructions for Running Code
=============================

The code has been tested on Ubuntu 24.04 but should work on any system with
python 3.10+ and JAX/FLAX. The instructions below assume a Ubuntu 24.04 distro,
possibly in WSL.


Clone repository
----------------

Clone and enter the repository.

```bash
git clone https://github.com/dimasad/scitech-2025-code.git
cd scitech-2025-code
```


Install dependencies
--------------------

Install a suitable Python version, create a Python virtual environment, and 
enter it.

```bash
sudo apt install python3.12-venv
python3.12 -m venv .venv
. .venv/bin/activate
```

Finally, install the visid python package, version 0.1.1.

```bash
pip install git+https://github.com/dimasad/visid.git@v0.1.1
```

Download flight data
--------------------

Download the data of Jategaonkar (2015) and Morelli and Klein (2016) from

- https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/flt_data.zip
- https://software.nasa.gov/software/LAR-16100-1

Place all `.mat` and `.asc` data files of those packages in the `data/` folder.

The following data files are used in the tests:

- `fAttasAil1.mat`
- `fAttasAil1_pqrDot.asc`
- `fAttasAilRud1.asc`
- `fAttasAilRud1.mat`
- `fAttasAilRud2.asc*`
- `fAttasAilRud2.mat`
- `fAttasElv1.asc*`
- `fAttasElv1.mat`
- `fAttasElv1_pqrDot.asc`
- `fAttasElv2.asc*`
- `fAttasElv2.mat`
- `fAttasRud1.mat`
- `fAttasRud1_pqrDot.asc`
- `fAttas_qst01.asc`
- `fAttas_qst01_withHeader.asc`
- `fAttas_qst02.asc`
- `hfb320_1_10.asc`
- `totter_9752ed20_data.mat`
- `totter_f1_014_data.mat`
- `totter_f1_017_data.mat`
- `tu144_f20_4d_data.mat`
- `tu144_f20_4f_pred_data.mat`

Run Examples
------------

Each example is an executable python script. Just run it with no arguments for
the options used in the article. The results will be saved in the `output/` 
folder.

```bash
./attas_lat.py
./attas_lat_pr.py
./attas_sp.py
./hfb320.py
./totter_lat.py
./totter_sp.py
./tu144_sp.py
```

To change any option, see the script's help.

```bash
./attas_lat_pr.py --help
```