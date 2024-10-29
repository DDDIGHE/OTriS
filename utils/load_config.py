import os
import time
import shutil
import warnings

from yapf.yapflib.yapf_api import FormatCode
from ._base import import_module, readable_dict, module_source


class Config:
    """
    ## Load Training Configuration from Python File

    ### Args:
        - configpath: The path of the config file
        - config_module: If we don't want to import configuration from file, we can directly specify the module to be used

    ### Attributions:
        - paras: The parameters in config file, dtype: dict
            - mandatory:
                system=None,
                model=None,
                vstate=None,
                work_config=None,
                checkpoint_config=None,
                optimizer=None,
                SR_conditioner=None,
                hyperpara=None

    ### Example:
    
    ```python
        lattice_length = 12
        lattice_dim = 1
        system = dict(backbone='Ising_System',
                    params=dict(Lattice_length=lattice_length,
                                Lattice_dim=lattice_dim,
                                PBC=True,
                                Spin=0.5,
                                Coupling=1,
                                Field_tranverse=1))
        model = dict(backbone='TransformerModel',
                    params=dict(masked=True,
                                num_heads=2,
                                num_layers=2,
                                embed_size=32,
                                ffn_dim=32,
                                vocab_size=lattice_length**lattice_dim,
                                max_length=lattice_length**lattice_dim))
        vstate = dict(n_samples=1008)
        work_config = dict(work_dir='./dev')
        checkpoint_config = dict(load_from='E:\OneDrive\StonyBrook\QML\dev',
                                save_inter=50)
        optimizer = dict(backbone='Adam',
                        params=dict(learning_rate=0.0001, b1=0.9, b2=0.999, eps=1e-8))
        SR_conditioner = dict(enabled=True, diag_shift=0.1)
        hyperpara = dict(epochs=2)
    ```
    """

    def __init__(self, configpath: str=None, config_module=None):
        self.paras = dict(system=None,
                          model=None,
                          vstate=None,
                          work_config=None,
                          checkpoint_config=None,
                          optimizer=None,
                          SR_conditioner=None,
                          hyperpara=None)
        if config_module is not None:
            self.config = config_module
        else:
            self.config = import_module(configpath)
        self.config_keys = dir(self.config)
        self.configpath = configpath
        self._check_config()
        # self.__para_config()
        # print(self.paras)

    def move_config(self,formatted=True,source_code:str='',save_path:str=None):
        if save_path is None:
            save_path=os.path.join(self.paras['work_config']['work_dir'], 'config.py')
        if self.configpath is not None:
            if formatted:
                source_code=FormatCode(module_source(self.configpath))[0]
                with open(save_path,'w+')as f:
                    f.write(source_code)
                f.close()
            else:
                shutil.copyfile(self.configpath,save_path)
        else:
            if formatted:
                source_code=FormatCode(source_code)[0]
            with open(save_path,'w+')as f:
                f.write(source_code)
            f.close()
    def _check_config(self):
        paras_config = self.config_keys
        error = []
        for key in self.paras.keys():
            if key not in paras_config:
                error.append(key)
            else:
                self.paras[key] = getattr(self.config, key)
        assert len(error) == 0, f'These basic configurations are not specified in {self.configpath}:\n{error}'

    def _build_model(self,imported_env):
        module=imported_env[getattr(self.config,'model')['backbone']]
        model_info = getattr(self.config, 'model')
        if 'params' not in model_info.keys():
            model_params = {}
        else:
            model_params = model_info['params']
        return module(**model_params)
    @property
    def workdir(self):
        return self.paras['work_config']['work_dir']
    @property
    def n_samples(self):
        return self.paras['vstate']['n_samples']

    @property
    def SR_diag_shift(self):
        return self.paras['SR_conditioner']['diag_shift']

    @property
    def SR_enabled(self):
        return self.paras['SR_conditioner']['enabled']

    @property
    def load_from(self):
        return self.paras['checkpoint_config']['load_from']
    
    @property
    def save_inter(self):
        return self.paras['checkpoint_config']['save_inter']
    
    @property
    def epochs(self):
        return self.paras['hyperpara']['epochs']
    def _build_optimizer(self,imported_env):
        module=imported_env[getattr(self.config,'optimizer')['backbone']]
        optimizer_info = getattr(self.config, 'optimizer')
        if 'params' not in optimizer_info.keys():
            optimizer_params = {}
        else:
            optimizer_params = optimizer_info['params']
        return module(**optimizer_params)

    def _build_system(self,imported_env):
        module=imported_env[getattr(self.config,'system')['backbone']]
        system_info = getattr(self.config, 'system')
        if 'params' not in system_info.keys():
            system_params = {}
        else:
            system_params = system_info['params']
        return module(**system_params)

    def __repr__(self) -> str:
        return readable_dict(self.paras)
    def __str__(self) -> str:
        return readable_dict(self.paras)
