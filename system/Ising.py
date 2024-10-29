import pickle as pkl
import netket as nk
from netket.exact import lanczos_ed

class Ising_System:
    def __init__(self,
                Lattice_length:int=4,
                Lattice_dim:int=2,
                PBC:bool=True,
                Spin:float=1/2,
                Coupling:float=-1,
                Field_tranverse:float=-0.5,
                ) -> None:
        self.Lattice_length=Lattice_length
        self.Lattice_dim=Lattice_dim
        self.PBC=PBC
        self.Spin=Spin
        self.Coupling=Coupling
        self.Field_tranverse=Field_tranverse
        self.init_system()

    def init_system(self):
        self.Lattice_graph=nk.graph.Hypercube(length=self.Lattice_length, n_dim=self.Lattice_dim, pbc=self.PBC)
        self.Hilbert_space=nk.hilbert.Spin(s=self.Spin, N=self.Lattice_graph.n_nodes)
        self.Hamiltonian=nk.operator.Ising(hilbert=self.Hilbert_space, graph=self.Lattice_graph, h=self.Field_tranverse, J=self.Coupling)

        self.str_repr=f'{self.__class__.__name__}\n\tLattice_length={self.Lattice_length}\n\tLattice_dim={self.Lattice_dim}\n\tPBC={self.PBC}\n\tSpin={self.Spin}\n\tCoupling={self.Coupling}\n\tField_tranverse={self.Field_tranverse}\n\tHilbert_space={self.Hilbert_space}\n\tHamiltonian={self.Hamiltonian}\nLattice_graph={self.Lattice_graph}'
    
    def eigen_energies(self, n_eigen:int=4):
        return lanczos_ed(self.Hamiltonian, k=n_eigen)

    def __str__(self):
        return self.str_repr
    def __repr__(self):
        return self.str_repr