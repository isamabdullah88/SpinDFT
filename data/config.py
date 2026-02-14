import os

INPUT_SCF ={
    "control": {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
        "disk_io": "low"
    },
    "system": {
        "ecutwfc": 30,
        "ecutrho": 300,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.01,
        "nspin": 2,
        # "nosym": True,
        "starting_magnetization(1)": 3.0,
        "starting_magnetization(2)": -3.0,
        "tot_magnetization": 0.0,
        "vdw_corr": "grimme-d3"
    },
    "electrons": {
        # "mixing_beta": 0.1, 
        "conv_thr": 1.0e-4,
        "diagonalization": "cg",
        # "electron_maxstep": 10,
        "mixing_mode": "local-TF"
    }
}

PSEUDOS = {
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "I":  "I.pbe-n-kjpaw_psl.0.2.UPF"
}

pseudo_dir = "./SSSP_1.3.0_PBE_efficiency/"

PSEUDO_DIR = os.path.abspath(pseudo_dir)

INPUT_SCF["control"]["pseudo_dir"] = PSEUDO_DIR
