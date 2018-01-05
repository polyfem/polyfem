#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System libs
import os
import sys
import json
import argparse
import tempfile
import subprocess

# Local libs
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../python'))
import paths


def main():
    file_exe = paths.get_executable("solver_test", os.path.join(paths.CPP_DIR, 'build'))
    file_mat = os.path.join(paths.DATA_DIR, 'mats/stiffness.txt')
    file_rhs = os.path.join(paths.DATA_DIR, 'mats/rhs.txt')
    base_args = [file_exe, file_mat, file_rhs]
    res = subprocess.run(base_args + ['-S', '1'], stdout=subprocess.PIPE, check=True)
    solvers = res.stdout.decode("utf-8").split()
    db = []
    for solver in solvers:
        # print(solver)
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            args = base_args + ['--solver', solver, '--json', tmp.name]
            # print(' '.join(args))
            subprocess.check_call(args)
            with open(tmp.name, 'r') as f:
                try:
                    output = json.load(f)
                    db.append(output)
                except ValueError:
                    print("[Solver] Warning: output json file is empty!")

    file_db = 'timings.json'
    with open(file_db, 'w') as f:
        f.write(json.dumps(db, indent=4))


if __name__ == "__main__":
    main()
