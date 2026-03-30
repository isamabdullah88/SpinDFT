import os

PHASE = 'FM'
STRAIN_TYPE = 'Biaxial'

RATTLE = False
RELAX = True
VCRELAX = False
SOC = False

NSCF_NBNDS = 55

INPUT_SCF ={
    "control": {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
        "disk_io": "low",
        'etot_conv_thr': 1.0e-5,
        'forc_conv_thr': 1.0e-4
    },
    "system": {
        "ecutwfc": 60,
        "ecutrho": 600,
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.002,
        "nspin": 2,
        "nosym": False,
        "starting_magnetization(1)": 3.0,
        "starting_magnetization(2)": [0.0 if PHASE == 'FM' else -3.0],
        "vdw_corr": "grimme-d3"
    },
    "electrons": {
        # "mixing_beta": 0.1, 
        "conv_thr": 1.0e-9,
        "diagonalization": "david",
        # "electron_maxstep": 10,
        "mixing_mode": "local-TF"
    }
}


if PHASE == 'AFM':
    INPUT_SCF["system"]["tot_magnetization"] = 0.0
    INPUT_SCF["system"]["nosym"] = True # AFM ordering breaks symmetries

KPTS = (10, 10, 1)

TB2J_KPTS = (36, 36, 1)

if RELAX:
    INPUT_SCF["control"]["calculation"] = "relax"
    
    INPUT_SCF["ions"] = {
        "ion_dynamics": "bfgs"
    }
    
    KPTS = (6, 6, 1)

if VCRELAX:
    INPUT_SCF["control"]["calculation"] = "vc-relax"

    INPUT_SCF["system"]["nosym"] = False # Allow symmetries for vc-relax

    INPUT_SCF["ions"] = {
        "ion_dynamics": "bfgs"
    }

    INPUT_SCF["cell"] = {
        "cell_dynamics": "bfgs",
        "press_conv_thr": 0.2,
        "cell_dofree": "2Dxy"
    }

    KPTS = (10, 10, 1)

PSEUDOS = {
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "I":  "I.pbe-n-kjpaw_psl.0.2.UPF"
}

pseudo_dir = "./SSSP_1.3.0_PBE_efficiency/"

PSEUDO_DIR = os.path.abspath(pseudo_dir)

INPUT_SCF["control"]["pseudo_dir"] = PSEUDO_DIR

NUM_RATTLE = 10 if RATTLE else 1
STDEV_RATTLE = 0.04 if RATTLE else 0.0
STRAIN_RANGE = 0.06