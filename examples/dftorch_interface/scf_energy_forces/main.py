"""
main.py
====================================
Main sedacs prototype driver to perform a
graph-based addaptive construction of the density
matrix together with a full self-consistent charge
optimization, followed by single-point calculation 
of the energy and forces.

"""

import sys

import numpy as np
from sedacs.driver.graph_adaptive_kernel_scf import get_adaptiveSCFDM
from sedacs.driver.graph_adaptive_sp_energy_forces import get_adaptive_sp_energy_forces
from sedacs.driver.init import get_args, init
from sedacs.file_io import read_latte_tbparams
from sedacs.charges import get_charges

from dftorch.Constants import Constants
from dftorch.Structure import Structure
import torch
import os
import warnings
import logging
import sedacs.globals as gl

### Configure torch and torch.compile ###
# Silence warnings and module logs
warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = ""               # disable PT2 logging
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
logging.getLogger("torch.fx").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.recording").setLevel(logging.CRITICAL)
# Enable dynamic shape capture for dynamo
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# default data type
torch.set_default_dtype(torch.float64)

torch.cuda.empty_cache()
device = "cuda"


#device = "cpu"

# Pass arguments from command line
args = get_args()

# Initialize sedacs
np.set_printoptions(threshold=sys.maxsize)


#print(args)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, graphweights  = init(
    args
)


print(vars(sy))

sy.latticeVectors = np.array([[40.230,0,0],[0,40.23,0],[0,0,40.230]])

dftorch_params = {
    "UNRESTRICTED": False,
    "SHARED_MU": False,  # if True, use shared chemical potential for both spin channels in unrestricted calculations. Otherwise, use separate chemical potentials for each spin channel.
    "BROKEN_SYM": False,  # if True, mix 2 % of lumo in homo at initialization


    "DELTA_SCF": False, # if True, perform delta SCF for targeted, non-aufbau excited state. Performs GS SCF, then ES SCF.
    "DELTA_SCF_TARGET": "SINGLET", # options: '"SINGLET" or "TRIPLET"'. desired lowest excited state
    "DELTA_SCF_SMEARING": False,  # if True, occupations for GS orbital and target ES orbital will be set to 0.5


    "coul_method": "!PME",  # 'FULL' for full coulomb matrix, 'PME' for PME method
    "Coulomb_acc": 1e-6,  # Coulomb accuracy for full coulomb calcs or t_err for PME
    "cutoff": 12.0,  # Coulomb cutoff
    "PME_order": 4,  # Ignored for FULL coulomb method

    "SCF_MAX_ITER": 90,  # Maximum number of SCF iterations
    "SCF_TOL": 1e-6,  # SCF convergence tolerance on density matrix
    "SCF_ALPHA": 0.1,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.

    "KRYLOV_MAXRANK": 15,  # Maximum Krylov subspace rank
    "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in SCF
    "KRYLOV_TOL_MD": 1e-6,  # Krylov subspace convergence tolerance in MD SCF
    "KRYLOV_START": 3,  # Number of initial SCF iterations before starting Krylov acceleration
    #"d3_params": {"s6": 1.0, "s8": 0.5883, "a1": 0.5719, "a2": 3.6107},
    #"solvent_param_file": "/Users/anthonybaldo/Documents/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/solvation/param_gbsa1_h2o.txt",
    #"solvent": "water",
    #"solvation_model": "gbsa",
}

LBOX = torch.tensor([40.23,40.23,40.23], device=device)

sy.lbox = LBOX
print(LBOX)

print(vars(sdc))
sdc.dftorch_params="/global/homes/a/abaldo2/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/"
print(vars(sdc))



const = Constants(
    sdc.coordsFileName,
    sdc.dftorch_params,
).to(device)

structure1 = Structure(
    sdc.coordsFileName,
    sy.lbox,
    const,
    charge=0,
    Te=sdc.etemp,
    device=device,
)



print(sy)

sdc.verb = True


# Perform a graph-adaptive calculation of the density matrix through SCF cycles
mu = 0.0
graphDH, sy.charges, mu, parts, partsCoreHalo, subSysOnRank = get_adaptiveSCFDM(
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu, graphweights=graphweights
)
# Perform a single-point graph-adaptive calculation of the energy and forces
graphDH, sy.charges, energy, forces, mu, parts, partsCoreHalo, subSysOnRank = get_adaptive_sp_energy_forces(
    sdc, eng, comm, rank, numranks, sy, parts, partsCoreHalo, hindex, graphNL, mu
)
print("total energy:", energy)
print("forces:", forces[0])
