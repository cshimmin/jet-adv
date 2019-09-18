N_CONST      = 96 # number of constituents
MIN_PT       = 0.5e-3 # minimum constituent pT (TeV)
JET_MASS_MIN = 60e-3 # minimum jet mass (TeV)
JET_PT_MIN   = 300e-3 # minimum jet pT (TeV)

CLUSTER_ARGS = dict(
    ntrk       = 64,     # max number of constituents
    min_ntrk   = 4,       # require 4 so that D2 is well-defined
    min_jet_pt = 1.5,     # minimum jet pT (TeV)
    min_trk_pt = 1e-3,    # minimum constituent pT (TeV)
    R          = 1.0,     # anti kT distance parameter
    unit       = 1e-3,    # work in units of TeV for dynamic range
)

# best res: l=3, u=512/256
BENCHMARK_LL_ARGS = dict(
    n_layers    = 5,
    n_units     = 1024,
    dropout     = 0.0,
    res         = False,
    n_res_units = 384,
    batch_norm  = False,
)

BENCHMARK_HL_ARGS = dict(
    n_feature  = 5,
    n_layers   = 5,
    n_units    = 384,
    dropout    = 0.25,
)

VALIDATION_FRACTION = 0.15