import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '0'

import matplotlib.pyplot as plt
import netket as nk
from utils import Ising_System, load_state, save_state, load_log
from model import TransformerModel, JasShort, FFNModel

system=Ising_System(
    Lattice_length=4,
    Lattice_dim=2,
    PBC=True,
    Spin=1/2,
    Coupling=-1,
    Field_tranverse=-0.5
)

workdir="E:\OneDrive\StonyBrook\QML\OTriS\dev"
if not os.path.exists(workdir):
    os.makedirs(workdir)

model=TransformerModel(masked=True,
                       num_heads=2,
                       num_layers=2,
                       embed_size=32,
                       ffn_dim=32,
                       vocab_size=system.Hilbert_space.size,
                       max_length=system.Hilbert_space.size)

sampler = nk.sampler.MetropolisLocal(system.Hilbert_space)
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
if os.path.exists(os.path.join(workdir, "log.mpack")):
    vstate.parameters=load_state(os.path.join(workdir, "log.mpack"))['variables']['params']
optimizer = nk.optimizer.Sgd(learning_rate=0.1)
vmc_dirver = nk.driver.VMC(system.Hamiltonian, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
log = nk.logging.RuntimeLog()
vmc_dirver.run(n_iter=10, save_params_every=50, out=os.path.join(workdir, "log"))
save_state(vstate, os.path.join(workdir, "log.mpack"))

plt.figure()
log_data = load_log(os.path.join(workdir, "log.log"))
iters=log_data['Energy']['iters']
E_mean=log_data['Energy']['Mean']
E_eigen=system.eigen_energies().min()
plt.plot(iters,E_mean)
plt.hlines([E_eigen], xmin=min(iters), xmax=max(iters), color='black', label=f"Exact GS energy: {E_eigen}")
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()
