"""
main.py
====================================
This script runs a Born-Oppenheimer molecular dynamics (BOMD) simulation using the LATTE interface through SEDACS package.
User can enalbe shadow molecular dynamics simulation by setting the shadow_md flag to 1.

"""

import sys
import argparse
import math
import torch
import numpy as np
import gc
import time
torch.set_default_dtype(torch.float64)

from sedacs.driver.init import init
from sedacs.graph import get_initial_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.driver.graph_kernel_byparts import get_kernel_byParts, rankN_update_byParts
from sedacs.driver.graph_adaptive_kernel_scf import get_adaptive_KernelSCFDM
from sedacs.driver.graph_adaptive_sp_energy_forces import get_adaptive_sp_energy_forces
from sedacs.file_io import read_latte_tbparams
from sedacs.periodic_table import PeriodicTable
from sedacs.neighbor_list import calculate_dist_dips 
from mpi4py import MPI


from dftorch.Constants import Constants
from dftorch.Structure import Structure
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


os.environ["DFTORCH_PARAMS_PATH"] = "/global/homes/a/abaldo2/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/"


####
# Global Constants
# Coversion factor from mass*velocity^2 to kinetic energy
MVV2KE = 166.0538782 / 1.602176487
# Conversion factor from kinetic energy to temperature
KE2T = 1.0 / 0.000086173435
# Conversion factor from force to velocity
F2V = 0.01602176487 / 1.660548782


def main(args):
    """! main program"""
    # Set random seed
    torch.manual_seed(137)
    np.random.seed(137)
    # Set numpy printing threshold
    np.set_printoptions(threshold=sys.maxsize)
    # Initialize sedacs parameters
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, graphweights = init(
        args
    )
    if rank == 0:
        # Open files to write down information during MD simulation
        MD_xyz = open("MD.xyz", "w")
        Energy_dat = open("Energy.dat", "w")
    # Set verbosity
    sdc.verb = False
    # Get device
    device = args.device
    if device == "cuda":
        local_rank = rank % torch.cuda.device_count()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    # Chemical potential
    mu = args.mu
    # Degree of localization for adaptive halo expansion
    localization = args.localization
    # Number of timesteps
    MD_Iter = args.md_iter
    # Size of the timestep
    dt = args.dt
    # System temperature
    Temperature = args.temp
    # If we want to run shadow md
    shadow_md = args.shadow_md
    # If we want to use kernel
    use_kernel = args.use_kernel
    # Initialize periodic table
    pt = PeriodicTable()
    # Get the atomic symbols for each atom in the system
    element_type = np.array(sy.symbols)[sy.types]
    # DFtorch params
    sy.latticeVectors = np.array([[8.0,0,0],[0,8.0,0],[0,0,8.0]])
    #sy.latticeVectors = np.array([[15.459,0,0],[0,15.459,0],[0,0,15.459]])
    #sy.latticeVectors = np.array([[40.23,0,0],[0,40.23,0],[0,0,40.23]])
    #sy.latticeVectors = np.array([[21.83,0,0],[0,21.83,0],[0,0,21.83]])

    dftorch_params = {
        "UNRESTRICTED": False,
        "SHARED_MU": False,  # if True, use shared chemical potential for both spin channels in unrestricted calculations. Otherwise, use separate chemical potentials for each spin channel.
        "BROKEN_SYM": False,  # if True, mix 2 % of lumo in homo at initialization


        "DELTA_SCF": False, # if True, perform delta SCF for targeted, non-aufbau excited state. Performs GS SCF, then ES SCF.
        "DELTA_SCF_TARGET": "SINGLET", # options: '"SINGLET" or "TRIPLET"'. desired lowest excited state
        "DELTA_SCF_SMEARING": False,  # if True, occupations for GS orbital and target ES orbital will be set to 0.5


        "coul_method": "!PME",  # 'FULL' for full coulomb matrix, 'PME' for PME method
        "Coulomb_acc": 1e-6,  # Coulomb accuracy for full coulomb calcs or t_err for PME
        "cutoff": 5.0,  # Coulomb cutoff
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

    LBOX = torch.tensor([8.0,8.0,8.0], device=device)
    #LBOX = torch.tensor([15.459,15.459,15.459], device=device)
    #LBOX = torch.tensor([40.23,40.23,40.23], device=device)
    #LBOX = torch.tensor([21.83,21.83,21.83], device=device)

    sy.lbox = LBOX
    #print(LBOX)

    #print(vars(sdc))
    sdc.dftorch_params="/global/homes/a/abaldo2/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/"
    #print(vars(sdc))



    const = Constants(
        'COORD_8WATER.xyz',
        sdc.dftorch_params,
    ).to(device)

    structure1 = Structure(
        'COORD_8WATER.xyz',
         sy.lbox,
         const,
         charge=0,
         Te=sdc.etemp,
        device=device,
    )

    sy.hubbard_u = structure1.Hubbard_U.numpy(force=True)
    print('DFTorch Hubbard Us: ', sy.hubbard_u)

    # Get the atomic masses for each atom in the system
    Mnuc = [pt.mass[pt.get_atomic_number(symbol)] for symbol in sy.symbols]
    Mnuc = np.array(Mnuc)[sy.types]
    # Convert the hubbard u and atomic masses to a tensor
    Hubbard_U = torch.tensor(structure1.Hubbard_U)
    Mnuc = torch.tensor(Mnuc)
    # Read the box size as a tensor
    LBox = torch.tensor(
        [sy.latticeVectors[0][0], sy.latticeVectors[1][1], sy.latticeVectors[2][2]]
    )
    # Read the coordinates as tensors
    coords = torch.tensor(sy.coords)
    # Perform a graph-adaptive calculation of the charges with SCF cycles
    tic = time.perf_counter()
    graphDH, sy.charges, mu, parts, partsCoreHalo, subSysOnRank = get_adaptive_KernelSCFDM(
        sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu, alpha=localization, graphweights=graphweights, device=device,
    )
    toc = time.perf_counter()
    print("Time for SCF", toc - tic, "(s)")

    njumps = 1
    partsCoreHalo = []
    for i in range(sdc.nparts):
        coreHalo, nc, nh = get_coreHaloIndices(parts[i], graphDH, njumps)
        partsCoreHalo.append(coreHalo)
        print("After SCF, core,halo size:", i, "=", nc, nh)
    # Perform a single-point graph-adaptive calculation of the energy and forces
    graphDH, sy.charges, EPOT, entropy, FTOT, mu, parts, partsCoreHalo, subSysOnRank = (
        get_adaptive_sp_energy_forces(
            sdc, eng, comm, rank, numranks, sy, parts, partsCoreHalo, hindex, graphDH, mu, alpha=localization, device=device, shadow_md=shadow_md,
        )
    )
    # Convert the charges to a tensor
    q = torch.tensor(sy.charges)
    # Read the number of atoms
    Nr_atoms = sy.nats
    # Convert the potential energy and forces to tensors
    EPOT = torch.tensor(EPOT)
    FTOT = torch.tensor(FTOT)

    # Initial BC for n, using the net Mulliken occupation per atom as extended electronic degrees of freedom
    n = q
    n_0 = q
    n_1 = q
    n_2 = q
    n_3 = q
    n_4 = q
    n_5 = q
    n_6 = q  # Set all "old" n-vectors to the same at t = t0
    # Coefficients for modified Verlet integration
    C0 = -14
    C1 = 36
    C2 = -27
    C3 = -2
    C4 = 12
    C5 = -6
    C6 = 1
    kappa = 1.84
    alpha = 0.0055

    # Initialize velocities
    V = torch.sqrt(Temperature / KE2T / MVV2KE / Mnuc).unsqueeze(1) * torch.randn_like(
        coords
    )
    # Compute and remove center of mass velocity
    COM_V = torch.sum(V.T * Mnuc, axis=1) / Mnuc.sum()
    V = V - COM_V
    # Remove net angular momentum
    R_center = torch.sum(coords.T * Mnuc, axis=1) / Mnuc.sum()
    R_shift = coords - R_center.unsqueeze(0)
    L = torch.sum(
        torch.linalg.cross(R_shift, Mnuc.unsqueeze(1) * V), dim=0
    )
    # vectorized calculation of inertia tensor
    I = torch.zeros((3, 3), dtype=torch.double)
    I[0, 0] = torch.sum(Mnuc * (R_shift[:, 1] ** 2 + R_shift[:, 2] ** 2))
    I[1, 1] = torch.sum(Mnuc * (R_shift[:, 0] ** 2 + R_shift[:, 2] ** 2))
    I[2, 2] = torch.sum(Mnuc * (R_shift[:, 0] ** 2 + R_shift[:, 1] ** 2))
    I[0, 1] = I[1, 0] = -torch.sum(Mnuc * R_shift[:, 0] * R_shift[:, 1])
    I[0, 2] = I[2, 0] = -torch.sum(Mnuc * R_shift[:, 0] * R_shift[:, 2])
    I[1, 2] = I[2, 1] = -torch.sum(Mnuc * R_shift[:, 1] * R_shift[:, 2])
    I_inv = torch.linalg.inv(I)
    # vectorized calculation of angular velocity
    omega = I_inv @ L
    V = V - torch.linalg.cross(omega.unsqueeze(0), coords - R_center.unsqueeze(0))

    # Record unwrapped coordsinates
    unwrap_coords = coords.clone().detach().double()

    renew = 0
    # MAIN MD LOOP {dR2(0)/dt2: V(0)->V(1/2); dn2(0)/dt2: n(0)->n(1); V(1/2): R(0)->R(1); dR2(1)/dt2: V(1/2)->V(1)}
    for MD_step in range(MD_Iter):
        # Calculate kinetic energy from particle velocities
        EKIN = torch.sum(0.5 * MVV2KE * torch.dot(Mnuc, torch.sum(V**2, dim=1)))
        # Calculate temperature from kinetic energy
        Temperature = (2.0 / 3.0) * KE2T * EKIN / Nr_atoms
        # Calculate the total energy from kinetic and potential energy
        ETOT = EKIN.item() + EPOT
        # Current time
        Time = (MD_step) * dt
        print(
            f"Time = {Time:<16.8f} Etotal = {ETOT:<16.8f} Temperature = {Temperature:<16.8f} SUM(q[n]) = {torch.sum(q).item():<16.16f}"
        )

        # dR2(0)/dt2: V(0)->V(1/2)
        V = V + 0.5 * dt * F2V * FTOT / Mnuc.unsqueeze(1)  # - 0.2 * V
        if rank == 0:
            # Here we record the time, temperature, and charges. Note that the last term, q, would be constant if not solving exact charges during MD
            with torch.no_grad():
                Energy_dat.write(
                    f"{Time/1000:<16.8f} {ETOT.item():<16.16f} {entropy:<16.16f} {Temperature.item():<16.8f} {EKIN.item():<16.16f} {EPOT.item():<16.16f} {torch.sum(q).item():<16.16f} {torch.sum(n_0).item():<16.16f} {mu:<16.16f} {torch.linalg.norm(q - n_0) / torch.sqrt(torch.tensor(sy.nats))}\n"
                )

            # Here we dump the MD trajectory
            if (MD_step % 1) == 0:
                # MD_xyz.write(f'## MD_step= {MD_step}\n')
                MD_xyz.write(f"{Nr_atoms}\n\n")
                for I in range(Nr_atoms):
                    MD_xyz.write(
                        f"{element_type[I].item()} {sy.coords[I, 0].item()} {sy.coords[I, 1].item()} {sy.coords[I, 2].item()} {sy.charges[I]}\n"
                    )
                MD_xyz.flush()
                Energy_dat.flush()

        if numranks > 1 and (Time % 10) == 9:
            coords_sum = torch.zeros_like(coords)
            comm.Allreduce(coords, coords_sum, op=MPI.SUM)
            coords = coords_sum / numranks

        # Caculate the residual between q[n] and n 
        Res = q - n_0 # Note that n_0 is n from the previous step
        # if use_kernel:
        if use_kernel:
            if MD_step == 0 or renew == 1:
                get_kernel_byParts(sdc, rank, numranks, parts, partsCoreHalo, sy, mu, device=device) 
                syk_ker_list = []
                for i, subSy in enumerate(sy.subSy_list):
                    syk_ker_list.append(subSy.ker.clone())
                dn2dt2 = 0
                renew = 0
            if MD_step > 0:
                for i, subSy in enumerate(sy.subSy_list):
                    subSy.ker = syk_ker_list[i] 
                dn2dt2 = -rankN_update_byParts(
                        q.to(device), n_0.to(device), 6, sdc, rank, numranks, comm, parts, partsCoreHalo, sy, mu=mu, device=device
                        )
                dn2dt2 = dn2dt2.to("cpu")
        else:
            dn2dt2 = 0.8 * Res
        # Propagating charge vector n for a better initial guess
        # Or Propagating charge vector for shadow MD
        n = (
            2 * n_0
            - n_1
            + kappa * dn2dt2 
            + alpha
            * (
                C0 * n_0
                + C1 * n_1
                + C2 * n_2
                + C3 * n_3
                + C4 * n_4
                + C5 * n_5
                + C6 * n_6
            )
        )
        n_6 = n_5
        n_5 = n_4
        n_4 = n_3
        n_3 = n_2
        n_2 = n_1
        n_1 = n_0
        n_0 = n
        sy.charges = n.numpy()

        # Update positions with full Verlet step for R
        disp = dt * V
        coords = coords + disp
        with torch.no_grad():
            unwrap_coords = unwrap_coords + disp
        # Reset coordinates within the periodic box
        coords = coords - LBox * torch.floor(coords / LBox)
        # Update sy.coords in the system object
        sy.coords = coords.numpy()
        # Update neighbor list
        coords_T = torch.from_numpy(sy.coords).to(args.device).T.contiguous()
        sy.nbr_state.update(coords_T)
        sy.nl_disps, sy.nl_dists, sy.nl = calculate_dist_dips(coords_T, sy.nbr_state)
        sy.nl = sy.nl.cpu()
        sy.nl_disps = sy.nl_disps.cpu()
        sy.nl_dists = sy.nl_dists.cpu()

        if not shadow_md:
            # Perform a graph-adaptive calculation of the charges with SCF cycles
            graphDH, sy.charges, mu, parts, subSysOnRank = get_adaptive_KernelSCFDM(
                sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu, alpha=localization, graphweights=graphweights, device=device,
            )

        if Time % 100 == 99:
            if rank == 0:
                nl = torch.where(
                        (sy.nl_dists > sdc.rcut) | (sy.nl_dists == 0.0), -1, sy.nl
                        )
                nl = nl.sort(dim=1, descending=True)[0]
                nl = nl[:, : torch.max(torch.sum(nl != -1, dim=1))]
                num_neighbors = torch.sum(nl != -1, dim=1)
                nl = torch.cat((num_neighbors.unsqueeze(1), nl.sort(dim=1, descending=True)[0]), dim=1)
                nl = nl.cpu().numpy()
                graph, graphweights = get_initial_graph(sy.coords, nl, sdc.rcut, sdc.maxDeg, np.diag(sy.latticeVectors), graphweights=True, verb=False)
                parts = graph_partition(
                    sdc, eng, graph, sdc.partitionType, sdc.nparts, sy.coords, graphweights=graphweights, verb=True
                )

            parts = comm.bcast(parts, root=0)

            renew = 1
            
        njumps = 1
        partsCoreHalo = []
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], graphDH, njumps)
            partsCoreHalo.append(coreHalo)
            print("MD_step, core,halo size:", MD_step, i, "=", nc, nh)
        # Perform a single-point graph-adaptive calculation of the energy and forces
        graphDH, sy.charges, EPOT, entropy, FTOT, mu, parts, partsCoreHalo, subSysOnRank = (
            get_adaptive_sp_energy_forces(
                sdc,
                eng,
                comm,
                rank,
                numranks,
                sy,
                parts,
                partsCoreHalo,
                hindex,
                graphDH,
                mu,
                alpha=localization,
                shadow_md=shadow_md,
                device=device,
            )
        )
        q = torch.tensor(sy.charges)
        # Constant shift in charges to maintain exact charge neutrality
        #q = q - (torch.sum(q)/len(q))
        # Convert the energy and forces to tensors
        EPOT = torch.tensor(EPOT)
        FTOT = torch.tensor(FTOT)

        # dR2(1)/dt2: V(1/2)->V(1)
        V = V + 0.5 * dt * F2V * FTOT / Mnuc.unsqueeze(1)
    if rank == 0:
        MD_xyz.close()
        Energy_dat.close()

    print(ETOT)


if __name__ == "__main__":
    # Pass arguments from command line
    parser = argparse.ArgumentParser(
        description="Extended-Lagrangian Born-Oppenheimer molecular dynamics with SEDACS-LATTE interface"
    )
    parser.add_argument(
        "--device", help="CPU/GPU device", type=str, default="cuda",
    )
    parser.add_argument("--use-torch", help="Use pytorch", required=False, action="store_true")
    parser.add_argument(
        "--input-file",
        help="Specify input file",
        required=False,
        type=str,
        default="input.in",
    )
    parser.add_argument(
        "--md_iter", type=int, default=10000, help="Number of timesteps"
    )
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep size (fs)")
    parser.add_argument(
        "--temp", type=float, default=0.0, help="Initial system temperature (K)"
    )
    parser.add_argument(
        "--mu", type=float, default=0.0, help="Initial Chemical potential (eV)"
    )
    parser.add_argument(
        "--localization", type=float, default=0.7, help="Degree of localization for adaptive halo expansion"
    )
    parser.add_argument(
        "--shadow_md",
        type=int,
        default=1,
        help="Set to 1/0 to enable/disable shadow MD",
    )
    parser.add_argument(
        "--use_kernel",
        type=int,
        default=1,
        help="Set to 1/0 to enable/disable kernel calculation",
    )
    args = parser.parse_args()

    print("Start running MD......")
    main(args)


