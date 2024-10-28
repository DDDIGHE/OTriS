'''
Author: airscker
Date: 2024-10-28 13:54:24
LastEditors: airscker
LastEditTime: 2024-10-28 18:32:03
Description: Basic utils for the project.

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import flax
import importlib.util
import pickle as pkl
import netket as nk
from netket.exact import lanczos_ed

#--------------------------------------------#
# Load saved logs
#--------------------------------------------#
def load_log(path:str):
    '''
    ## Load the log from a file
    ### Args:
        - path: str, path to load the log
    ### Returns:
        - log: dict, the log data
    '''
    assert path.endswith('.log'), f'Logging file must be a log file but {path} is given'
    with open(path, 'r') as f:
        log = json.load(f)
    return log

#--------------------------------------------#
# Save and Load State Params
#--------------------------------------------#
def save_state(state, path:str, **kwargs):
    '''
    ## Save the model's states to a file
    ### Args:
        - state: netket state object, which contains the parameters of model
        - path: str, path to save the parameters
    ### Example:
    ```python
        model = FFN(alpha=1)
        vstate = nk.vqs.MCState(sampler, model)
        save_state_params(vstate, './model_prams.mpack', hilbert_space=hilbert_space)
    ```
    '''
    # saved_info={
    #             # 'model':state.model,
    #             # 'sampler':state.sampler,
    #             'params':state.parameters,
    #             # 'hilbert_space':state.hilbert,
    #             **kwargs}
    # with open(path, 'wb') as f:
    #     pkl.dump(saved_info,f)
    assert path.endswith('.mpack'), f'Config file must be a mpack file but {path} is given'
    with open(path, 'wb') as f:
        f.write(flax.serialization.to_bytes(state))

def load_state(path):
    '''
    ## Load the state parameters from a file
    ### Args:
        - path: str, path to load the parameters
    ### Returns:
        - info: dict, the parameters of the model, with data structure:
            {
            'variables':{'params':vstate.parameters},
            'sampler_state':vstate.sampler_state,
            'n_steps_proc':vstate.n_steps_proc,
            'rng:':vstate.rng,
            'rule_state':vstate.rule_state,
            'σ':vstate.σ,
            'n_samples':vstate.n_samples,
            'n_discard_per_chain':vstate.n_discard_per_chain,
            'chunk_size':vstate.chunk_size,
            }
    '''
    # with open(path, 'rb') as f:
    #     info = pkl.load(f)
    # return info
    with open(path, 'rb') as f:
        info = flax.serialization.msgpack_restore(f.read())
    f.close()
    print(f'Parameters have been loaded from {path}')
    return info

#--------------------------------------------#
# Import Module
#--------------------------------------------#
def import_module(module_path: str):
    '''
    ## Import python module according to the file path

    ### Args:
        - module_path: the path of the module to be imported

    ### Return:
        - the imported module
    '''
    assert module_path.endswith(
        '.py'), f'Config file must be a python file but {module_path} is given'
    total_path = os.path.abspath(module_path)
    assert os.path.exists(
        total_path), f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('', total_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

#--------------------------------------------#
# Generate basic configs for different systems
#--------------------------------------------#

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

        self.str_repr=f'Lattice_length={self.Lattice_length}\nLattice_dim={self.Lattice_dim}\nPBC={self.PBC}\nSpin={self.Spin}\nCoupling={self.Coupling}\nField_tranverse={self.Field_tranverse}'

    def save_config(self,path:str):
        assert path.endswith('.py'), f'Config file must be a python file but {path} is given'
        with open(path,'w') as f:
            f.write(self.str_repr)
        f.close()
        print(f'Config file has been saved to {path}')
    
    def load_config(self,path:str):
        assert path.endswith('.py'), f'Config file must be a python file but {path} is given'
        imported_config=import_module(path)
        self.Lattice_length=imported_config.Lattice_length
        self.Lattice_dim=imported_config.Lattice_dim
        self.PBC=imported_config.PBC
        self.Spin=imported_config.Spin
        self.Coupling=imported_config.Coupling
        self.Field_tranverse=imported_config.Field_tranverse
        self.init_system()
    
    def eigen_energies(self, n_eigen:int=4):
        return lanczos_ed(self.Hamiltonian, k=n_eigen)

    def __str__(self):
        return self.str_repr
    
    def __repr__(self):
        return self.str_repr