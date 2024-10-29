'''
Author: airscker
Date: 2024-10-28 13:54:24
LastEditors: airscker
LastEditTime: 2024-10-29 02:52:42
Description: Basic utils for the project.

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import flax
import importlib.util

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
# Convert dictionary to readable format
#--------------------------------------------#

def readable_dict(data: dict, i=0, show=False, indent='\t', sep='\n'):
    """
    ## Convert a dictionary to a more readable format.

    ### Args:
        - data: Pass the data to be printed
        - i: Control the indentation of the output
        - show: Whether to print out the mesage in console
        - indent: the indent letter used to convert dictionary
        - spe: the seperation letter used to seperate dict elements

    ### Return:
        - A string represent the dictionary
    """
    info = ''
    for key in data:
        info += indent*i
        info += f'{key}: '
        if isinstance(data[key], dict):
            info += f"{sep}{readable_dict(data[key], i+1,indent=indent,sep=sep)}"
        else:
            info += f"{data[key]}{sep}"
    if show:
        print(info)
    return info

#--------------------------------------------#
# Read the source code of a module
#--------------------------------------------#

def module_source(module_path: str):
    '''
    ## Get python module's source code according to the file path

    ### Args:
        - module_path: the path of the module to be imported

    ### Return:
        - the source code of the module
    '''
    assert module_path.endswith(
        '.py'), f'Config file must be a python file but {module_path} is given'
    total_path = os.path.abspath(module_path)
    assert os.path.exists(
        total_path), f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('', total_path)
    module_loader = module_spec.loader
    return module_loader.get_source('')

#--------------------------------------------#
# Genarate a formmated config file to be filled
#--------------------------------------------#

def generate_config(path: str):
    assert path.endswith('.py'), f'Config file must be a python file but {path} is given'
    _str1="system = dict(backbone='',params=dict())\nmodel = dict(backbone='',params=dict())\n"
    _str2="vstate=dict(n_samples=1008)\nwork_config = dict(work_dir='./dev')\n"
    _str3="checkpoint_config = dict(load_from='', resume_from='', save_inter=50)\n"
    _str4="optimizer = dict(backbone='AdamW', params=dict(lr=0.0001))\nSR_conditioner = dict(enabled=True, diag_shift=0.1)\n"
    _str5="hyperpara = dict(epochs=100)"
    _str=_str1+_str2+_str3+_str4+_str5
    with open(path,'w') as f:
        f.write(_str)
    f.close()
    print(f'Config file generated to {path}')

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
    saved_info={
                # 'model':state.model,
                # 'sampler':state.sampler,
                'params':state.parameters,
                # 'hilbert_space':state.hilbert,
                **kwargs}
    # with open(path, 'wb') as f:
    #     pkl.dump(saved_info,f)
    assert path.endswith('.mpack'), f'Config file must be a mpack file but {path} is given'
    with open(path, 'wb') as f:
        f.write(flax.serialization.msgpack_serialize(saved_info))

def load_state(path):
    '''
    ## Load the state parameters from a file
    ### Args:
        - path: str, path to load the parameters
    ### Returns:
        - info: dict, the parameters of the model, with data structure:
            {
            'params':vstate.parameters,
            ...
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