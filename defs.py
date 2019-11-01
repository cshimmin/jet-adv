N_CONST      = 64 # number of constituents
MIN_PT       = 0.5e-3 # minimum constituent pT (TeV)
JET_MASS_MIN = 60e-3 # minimum jet mass (TeV)
JET_PT_MIN   = 300e-3 # minimum jet pT (TeV)


BENCHMARK_LL_ARGS = dict(
    n_units           = (1024,1024,1024,1024),
    dropout           = 0.5,
    res               = False,
    n_res_units       = 384,
    batch_norm        = False,
    shuffle_particles = False,
    randomize_phi     = True,
)

BENCHMARK_LL_SHUF_ARGS = dict(
    n_units           = (2048,1024,512,512,),
    dropout           = 0.0,
    res               = False,
    n_res_units       = 384,
    
    batch_norm        = False,
    shuffle_particles = True,
    randomize_phi     = False,
)

BENCHMARK_PFN_ARGS = dict(
    #Phi_sizes = (100,128),
    #F_sizes = (100, 100),
    Phi_sizes     = (256,256,256,256,),
    F_sizes       = (256,256,256,256,),
    Phi_dropouts  = 0.,
    F_dropouts    = 0.,
    randomize_phi = True,
)

BENCHMARK_HL_ARGS = dict(
    features   = ('pt','eta','mass','D2',),
    n_layers   = 3,
    n_units    = 384,
    dropout    = 0.0,
)

VALIDATION_FRACTION = 0.15