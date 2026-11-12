"""Collect comparable solver inputs and outputs for a spline regression scene."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", choices=("cube_twist", "penguin"), default="cube_twist")
    options = parser.parse_args()
    binary = options.binary.resolve()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).resolve().parents[1] / "polyspline-data"

    scene = json.loads((data / (options.scene + ".json")).read_text())
    scene["common"] = str(data / "common.json")
    scene["geometry"]["mesh"] = str(data / scene["geometry"]["mesh"])
    scene["output"] = {
        "json": str(output / "stats.json"),
        "data": {
            "full_mat": str(output / "full_stiffness.mtx"),
            "stiffness_mat": str(output / "constrained_stiffness.mtx"),
            "solution": str(output / "solution.txt"),
        },
    }
    config = output / "input.json"
    config.write_text(json.dumps(scene, indent=4) + "\n")
    (output / "platform.txt").write_text(platform.platform() + "\n")
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            [str(binary), "--json", str(config), "--output_dir", str(output),
             "--max_threads", "1"],
            cwd=output, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, POLYFEM_RBF_DIAGNOSTICS="1"),
        )
    if result.returncode:
        print((output / "run.log").read_text())
        raise SystemExit(result.returncode)
    stats = json.loads((output / "stats.json").read_text())
    print(json.dumps({key: value for key, value in stats.items()
                      if key.startswith("err_")}, indent=2))


if __name__ == "__main__":
    main()
