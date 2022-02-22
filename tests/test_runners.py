import subprocess
import os
import sys
import json
import pathlib


def run_one(args, errs, margin):
    path = os.getcwd()
    polyfem_exe = os.path.join(path, "PolyFEM_bin")

    if sys.platform == "win32":
        polyfem_exe += ".exe"

    cmd = [polyfem_exe] + args + ["--output", "tmp.json", "--f_delta", "0"]



    print(" ".join(cmd))
    subprocess.check_output(cmd)
    # out = out.decode("utf-8")

    err_str = ["err_l2","err_lp","err_h1","err_h1_semi","err_linf","err_linf_grad"]

    with open("tmp.json") as f:
        data = json.load(f)

        for i,k in enumerate(err_str):
            expected = errs[i]
            value = data[k]
            if abs(value - expected) >= margin:
                print(data)
                print("{} value: {}, expected: {}, difference {} >= {}".format(k, value, expected, abs(value - expected), margin))
            assert(abs(value - expected) < margin)


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1] / "data" / "data"
    cube_mesh = str(data_dir / "contact/meshes/3D/simple/cube.msh")
    run_one(
        ["--mesh", cube_mesh, "--cmd", "--n_refs", "1", "--sform", "Laplacian"],
        [0.11665508022536396,  0.1900498909169514,  1.0710196019811378,  1.0646476318883393,  0.2805319397256366,  0.6292106310115898],
        1e-8)
    run_one(
        ["--mesh", cube_mesh, "--cmd", "--n_refs", "1", "--sform", "Helmholtz"],
        [0.11665515369492689, 0.1900436246695708, 1.0710458804858578, 1.0646740596173738, 0.2805185423746003, 0.6293487025846484],
        1e-8)

    run_one(
        ["--mesh" , cube_mesh , "--cmd" , "--n_refs" , "1" , "--problem" , "ElasticExact" , "--tform" , "HookeLinearElasticity"],
        [0.0051194614099276635, 0.007268490977599741, 0.0415182455700811, 0.04120140568099371, 0.010053424098237947, 0.05496122617564021],
        1e-8)

    run_one(
        ["--mesh", cube_mesh, "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "LinearElasticity"],
        [0.0051194614099276635, 0.007268490977599741, 0.0415182455700811, 0.04120140568099371, 0.010053424098237947, 0.05496122617564021],
        1e-8
    )

    run_one(
        ["--mesh", cube_mesh, "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "NeoHookean"],
        [0.005117700018049065, 0.007263985026249126, 0.04150655801411695, 0.04118984710707866, 0.010044897918133419, 0.054999520673069326],
        1e-8
    )

    run_one(
        ["--mesh", cube_mesh, "--cmd", "--problem", "ElasticExact", "--tform", "NeoHookean", "-p", "3", "--solver", "Eigen::SimplicialLDLT"],
        [0.00026827281925665265, 0.0003087929463643, 0.0023639542322829218, 0.00234868246188718, 0.0003737955727988801, 0.003894218476978001],
        1e-8
    )

    if sys.platform != "darwin": # too much memory on CI
        run_one(
            ["--mesh", cube_mesh, "--cmd", "--problem", "ElasticExact", "--tform", "NeoHookean", "-p", "4", "--solver", "Eigen::SimplicialLDLT"],
            [1.5726893318749926e-07, 3.1110016107991764e-07, 2.0552150207653863e-06, 2.049188928389409e-06, 5.178874747956526e-07, 7.138068100334043e-06],
            1e-8
        )

    run_one(
        ["--mesh", cube_mesh, "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "SaintVenant"],
        [0.00511732845448437, 0.00727686270970279, 0.041523655306609904, 0.04120712195132144, 0.010067087360058515, 0.05493035171552578],
        1e-8
    )


    if sys.platform != "win32":
        run_one(
            ["--json", str(data_dir / "contact/examples/3D/unit-tests/5-cubes-fast.json"),  "--cmd"],
            [0.011771999999999986,0.039362054350912154,0.011771999999999986,7.982054466618666e-17,0.058859999999999996,0.058859999999999996],
            1e-8
        )

    run_one(
        ["--febio", str(data_dir / "lin-neo.feb"), "--cmd", "--compute_errors"],
        [0.003317541547798692,0.0013471375792506904,0.0036206540278807355,0.001450190907514749,0.001075813728306999,0.001075813728306999],
        1e-8
    )

