import os

PHASE = 'AFM'
RELAX = True
VCRELAX = False

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
        "ecutwfc": 40,
        "ecutrho": 320,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.01,
        "nspin": 2,
        # "nosym": True,
        "starting_magnetization(1)": 3.0,
        "starting_magnetization(2)": [0.0 if PHASE == 'FM' else -3.0],
        # "tot_magnetization": 0.0,
        "vdw_corr": "grimme-d3"
    },
    "electrons": {
        # "mixing_beta": 0.1, 
        "conv_thr": 1.0e-5,
        "diagonalization": "cg",
        # "electron_maxstep": 10,
        "mixing_mode": "local-TF"
    }
}

if RELAX:
    INPUT_SCF["control"]["calculation"] = "relax"

if VCRELAX:
    INPUT_SCF["control"]["calculation"] = "vc-relax"

    INPUT_SCF["ions"] = {
        "ion_dynamics": "bfgs"
    }

    INPUT_SCF["cell"] = {
        "cell_dynamics": "bfgs",
        "press_conv_thr": 0.2,
        "cell_dofree": "2Dxy"
    }

KPTS = (6, 6, 1)

if PHASE == 'AFM':
    INPUT_SCF["system"]["tot_magnetization"] = 0.0

PSEUDOS = {
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "I":  "I.pbe-n-kjpaw_psl.0.2.UPF"
}

pseudo_dir = "./SSSP_1.3.0_PBE_efficiency/"

PSEUDO_DIR = os.path.abspath(pseudo_dir)

INPUT_SCF["control"]["pseudo_dir"] = PSEUDO_DIR
