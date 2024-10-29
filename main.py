import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '0'

import argparse
import matplotlib.pyplot as plt
import netket as nk
from utils import (Config,
                   load_state, save_state, load_log)
from model import *
from system import *
from netket.optimizer import *

def main(exp_config:Config):
    global_env = globals()
    system=exp_config._build_system(global_env)
    print(system)
    workdir=exp_config.workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    exp_config.move_config(save_path=os.path.join(workdir, 'config.py'))
    
    model=exp_config._build_model(global_env)
    print(model)
    sampler = nk.sampler.MetropolisLocal(system.Hilbert_space)
    vstate = nk.vqs.MCState(sampler, model, n_samples=exp_config.n_samples)
    optimizer = exp_config._build_optimizer(global_env)
    if exp_config.SR_enabled:
        vmc_dirver = nk.driver.VMC(system.Hamiltonian, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=exp_config.SR_diag_shift))
    else:
        vmc_dirver = nk.driver.VMC(system.Hamiltonian, optimizer, variational_state=vstate)
    
    checkpoint_folder=exp_config.load_from
    if checkpoint_folder!='' and os.path.exists(checkpoint_folder):
        vstate.parameters=load_state(os.path.join(checkpoint_folder, "log.mpack"))['params']
    
    vmc_dirver.run(n_iter=exp_config.epochs, save_params_every=exp_config.save_inter, out=os.path.join(workdir, "log"))
    save_state(vstate, os.path.join(workdir, "log.mpack"))

    plt.figure(figsize=(10, 6))
    log_data = load_log(os.path.join(workdir, "log.log"))
    iters=log_data['Energy']['iters']
    E_mean=log_data['Energy']['Mean']
    print('Calculating exact GS energy...')
    E_eigen=system.eigen_energies().min()
    print(f'Exact GS energy: {E_eigen}')
    plt.plot(iters,E_mean)
    plt.hlines([E_eigen], xmin=min(iters), xmax=max(iters), color='black', label=f"Exact GS energy: {E_eigen}")
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(workdir, 'Energy.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        default='E:\OneDrive\StonyBrook\QML\OTriS\config\sample_ising.py',
                        type=str,
                        help='the path of config file')
    args = parser.parse_args()
    exp_config=Config(args.config)
    print(f"Experiment config:\n{exp_config}")
    main(exp_config)
