lattice_length=12
lattice_dim=1
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
vstate=dict(n_samples=1008)
work_config = dict(work_dir='./dev')
checkpoint_config = dict(load_from='E:\OneDrive\StonyBrook\QML\dev', save_inter=50)
optimizer = dict(backbone='Adam', params=dict(learning_rate=0.0001, b1= 0.9, b2 = 0.999, eps = 1e-8))
SR_conditioner = dict(enabled=True, diag_shift=0.1)
hyperpara = dict(epochs=2)