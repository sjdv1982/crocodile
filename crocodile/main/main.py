import json
import os, sys
workdir = None

from crocodile.nuc.reference import Reference
from crocodile.main.grow import grow
from crocodile.main import pdb

def import_config(workdir) -> None:
    # Import crocodile_library_config, adding workdir to the search path
    try:
        old_sys_path = sys.path.copy()
        sys.path[:] = old_sys_path + [workdir]
        import crocodile_library_config
    except ImportError:
        print("Cannot locate crocodile_library_config.py. This file must be in your working directory or in the Python path", file=sys.stderr)
        exit(1)
    finally:
        sys.path[:] = old_sys_path

def locate_files(workdir) -> None:
    

    # Check that basic files and directories exist
    files = ["constraints.json", "reference.pdb", "commands.txt"]
    directories = ["grids", "proteins", "state"]
    
    # TODO: make this nicer
    for f in files + directories:
        assert os.path.exists(f),f 

def load_commands(command_file) -> list[dict]:
    commands = []
    with open(command_file) as f:
        for l in f:
            l = l.strip()
            if not len(l):
                continue
            command = {}
            ll = l.split()
            # provisional... validation TODO
            assert ll[2] == "as", l
            assert ll[4] == "from", l
            command["type"] = ll[0]
            command["name"] = ll[1]
            command["fragment"] = int(ll[3])
            command["origin"] = ll[5]

            commands.append(command)
    return commands

def main() -> None:
    workdir = os.getcwd()
    import_config(workdir)
    locate_files(workdir)

    from crocodile_library_config import mononucleotide_templates

    with open("constraints.json") as f:
        constraints = json.load(f)

    commands = load_commands("commands.txt")
    
    refe_ppdb = pdb.parse_pdb(open("reference.pdb").read()) # TODO: add missing atoms
    refe = Reference(
        ppdb=refe_ppdb,
        mononucleotide_templates=mononucleotide_templates,
        rna=True,
        ignore_unknown=False,
        ignore_missing=False,
        ignore_reordered=False,
    )

    state = {"reference": refe}



    print("OK")
    print(commands)
    grow(commands[0], constraints, state)