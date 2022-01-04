import subprocess
import os
import sys


def run_one(args, errs, margin):
    path = os.getcwd()
    poolyfem_exe = os.path.join(path, "PolyFEM_bin")

    if sys.platform == "win32":
        poolyfem_exe += ".exe"

    cmd = [poolyfem_exe] + args



    print(" ".join(cmd))
    out = subprocess.check_output(cmd)
    out = out.decode("utf-8")

    err_str = ["L2 error","Lp error","H1 error","H1 semi error","Linf error","grad max error"]

    for i,k in enumerate(err_str):
        index = out.find(k)
        index += len(k)+2
        end = out.find("\n", index)
        num = float(out[index:end])
        if abs(num - errs[i]) >= margin:
            print(out)
            print("{} err: {}, expected: {}, difference {} >= {}".format(k, num, errs[i], abs(num - errs[i]), margin))
        assert(abs(num - errs[i]) < margin)

if __name__ == "__main__":
    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--n_refs", "1", "--sform", "Laplacian"],
        [0.11665508022536396,  0.1900498909169514,  1.0710196019811378,  1.0646476318883393,  0.2805319397256366,  0.6292106310115898],
        1e-8)
    run_one(
        ["--mesh","../data/data/contact/meshes/3D/simple/cube.msh","--cmd","--n_refs","1","--sform","Helmholtz"],
        [0.11665515369492689, 0.1900436246695708, 1.0710458804858578, 1.0646740596173738, 0.2805185423746003, 0.6293487025846484],
        1e-8)

    run_one(
        ["--mesh" , "../data/data/contact/meshes/3D/simple/cube.msh" , "--cmd" , "--n_refs" , "1" , "--problem" , "ElasticExact" , "--tform" , "HookeLinearElasticity"],
        [0.00513178898301962, 0.00727770219768311, 0.041479461223626474, 0.041160787713989275, 0.010068875317772641, 0.054789862715217436],
        1e-8)

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "LinearElasticity"],
        [0.00513178898301962, 0.00727770219768311, 0.041479461223626474, 0.041160787713989275, 0.010068875317772641, 0.054789862715217436],
        1e-8
    )

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "NeoHookean"],
        [0.0051299463758777445, 0.007273504953205199, 0.041472207643984106, 0.04115370769501033, 0.010061026263237665, 0.05483214512781843],
        1e-8
    )

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--problem", "ElasticExact", "--tform", "NeoHookean", "-p", "3", "--solver", "Eigen::SimplicialLDLT"],
        [0.000269138830916869,0.0003076729617746929,0.0023574478071778024,0.0023420342553558283,0.0003709604445908649,0.0038506926501288507],
        1e-8
    )

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--problem", "ElasticExact", "--tform", "NeoHookean", "-p", "4", "--solver", "Eigen::SimplicialLDLT"],
        [1.3287076839116818e-07, 2.628334984159712e-07, 1.7222873642153572e-06, 1.7171543680878033e-06, 4.423485882777132e-07, 5.921716770113337e-06],
        1e-8
    )

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "SaintVenant"],
        [0.005130143730924825, 0.007286507292818326, 0.041484890207615074, 0.04116646378835483, 0.010083460622798765, 0.05474744761783359],
        1e-8
    )


    if sys.platform != "win32":
        run_one(
            ["--json",  "../data/data/contact/examples/3D/unit-tests/5-cubes-fast.json",  "--cmd"],
            [0.011771999999999986,0.039362054350912154,0.011771999999999986,7.982054466618666e-17,0.058859999999999996,0.058859999999999996],
            1e-8
        )

    run_one(
        ["--febio", "../data/data/lin-neo.feb", "--cmd", "--compute_errors"],
        [0.003317541547798692,0.0013471375792506904,0.0036206540278807355,0.001450190907514749,0.001075813728306999,0.001075813728306999],
        1e-8
    )

