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
        [0.005129946375877744, 0.007273504953205196, 0.041472207643984106, 0.04115370769501033, 0.01006102626323766, 0.054832145127818456],
        1e-7
    )

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--problem", "ElasticExact", "--tform", "NeoHookean", "-p", "3", "--solver", "Eigen::SimplicialLDLT"],
        [0.00026914175293163043, 0.00030767125405190725, 0.0023574622840964975, 0.002342048491762364, 0.0003709364513102658, 0.0038508053568526235],
        1e-7
    )

    if sys.platform != "darwin": # too much memory on CI
        run_one(
            ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--problem", "ElasticExact", "--tform", "NeoHookean", "-p", "4", "--solver", "Eigen::SimplicialLDLT"],
            [1.3287076839149297e-07, 2.628334984177216e-07, 1.7222873642188449e-06, 1.7171543680912764e-06, 4.423485882799039e-07, 5.921716770190064e-06],
            1e-7
        )

    run_one(
        ["--mesh", "../data/data/contact/meshes/3D/simple/cube.msh", "--cmd", "--n_refs", "1", "--problem", "ElasticExact", "--tform", "SaintVenant", "--solver", "Eigen::SimplicialLDLT"],
        [0.005130143730924622, 0.00728650729281807, 0.041484890207615144, 0.04116646378835493, 0.010083460622798305, 0.054747447617836904],
        1e-7
    )


    if sys.platform != "win32":
        run_one(
            ["--json",  "../data/data/contact/examples/3D/unit-tests/5-cubes-fast.json",  "--cmd"],
            [0.011771999999999979, 0.03936205435091213, 0.011771999999999979, 6.534697153261478e-17, 0.05885999999999997, 0.05885999999999997],
            1e-7
        )

    run_one(
        ["--febio", "../data/data/lin-neo.feb", "--cmd", "--compute_errors"],
        [0.00331754154779869, 0.0013471375792506895, 0.003620654027880733, 0.0014501909075147475, 0.0010758137283069988, 0.0010758137283069988],
        1e-7
    )

