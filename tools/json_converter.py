import json
import argparse
import jsbeautifier


def copy_entry(key, f, t):
    if (key in f):
        t[key] = f[key]


def rename_entry(key, f, key2, t):
    if (key in f):
        t[key2] = f[key]


def remove_empty_dicts_from_list(li):
    assert (type(li) == list)
    for val in li:
        if type(val) is dict:
            remove_empty_dicts_from_dict(val)
        elif type(val) == list:
            remove_empty_dicts_from_list(val)


def remove_empty_dicts_from_dict(di):
    for key, val in list(di.items()):
        if type(val) is dict:
            if len(val) == 0:
                del di[key]
            else:
                remove_empty_dicts_from_dict(val)
                if len(val) == 0:
                    del di[key]
        elif type(val) == list:
            remove_empty_dicts_from_list(val)
            if len(val) == 0:
                del di[key]


def PolyFEM_convert(old):
    j = {}
    rename_entry("default_params", old, "common", j)
    copy_entry("root_path", old, j)
    copy_entry("authen_t1", old, j)

    # Meshes to Geometry
    j["geometry"] = []
    if "meshes" in old:
        for o in old["meshes"]:
            n = {}
            copy_entry("type", o, n)
            copy_entry("mesh", o, n)
            # n["is_obstacle"] = False
            copy_entry("enabled", o, n)

            # Transformation
            n["transformation"] = {}
            rename_entry("position", o, "translation", n["transformation"])
            copy_entry("rotation", o, n["transformation"])
            copy_entry("rotation_mode", o, n["transformation"])
            copy_entry("scale", o, n["transformation"])
            copy_entry("dimensions", o, n["transformation"])

            rename_entry("body_id", o, "volume_selection", n)
            rename_entry("boundary_id", o, "surface_selection", n)

            copy_entry("n_refs", old, n)
            n["advanced"] = {}
            copy_entry("force_linear_geometry", old, n["advanced"])
            copy_entry("refinement_location", old, n["advanced"])
            copy_entry("normalize_mesh", old, n["advanced"])
            copy_entry("min_component", old, n["advanced"])

            j["geometry"].append(n)

    if "mesh" in old:
        n = {}
        n["mesh"] = old["mesh"]
        j["geometry"].append(n)

    # Obstacles to Geometry
    if "obstacles" in old:
        for i, o in enumerate(old["obstacles"]):
            n = {}
            copy_entry("type", o, n)
            copy_entry("mesh", o, n)
            n["is_obstacle"] = True
            copy_entry("enabled", o, n)

            # Transformation
            n["transformation"] = {}
            rename_entry("position", o, "translation", n["transformation"])
            copy_entry("rotation", o, n["transformation"])
            copy_entry("rotation_mode", o, n["transformation"])
            copy_entry("scale", o, n["transformation"])
            copy_entry("dimensions", o, n["transformation"])

            copy_entry("n_refs", old, n)
            n["advanced"] = {}
            copy_entry("refinement_location", old, n["advanced"])

            if "displacement" in o:
                if "boundary_conditions" not in j:
                    j["boundary_conditions"] = {"obstacle_displacements": []}
                n["surface_selection"] = 1000 + i
                j["boundary_conditions"]["obstacle_displacements"].append({
                    "id": 1000 + i,
                    "value": o["displacement"]
                })

            j["geometry"].append(n)

    # Space
    j["space"] = {}
    copy_entry("discr_order", old, j["space"])
    if "bodies_discr_order" in old:
        j["space"]["discr_order"] = []
        for order in old["bodies_discr_order"]:
            j["space"]["discr_order"].append({
                "id": order["body_id"],
                "order": order["discr"]
            })
    copy_entry("pressure_discr_order", old, j["space"])
    copy_entry("use_p_ref", old, j["space"])

    j["space"]["advanced"] = {}
    copy_entry("particle", old, j["space"]["advanced"])
    copy_entry("discr_order_max", old, j["space"]["advanced"])
    # copy_entry("serendipity", old, j["space"]["advanced"])
    copy_entry("isoparametric", old, j["space"]["advanced"])
    # copy_entry("use_spline", old, j["space"]["advanced"])
    copy_entry("bc_method", old, j["space"]["advanced"])
    copy_entry("n_boundary_samples", old, j["space"]["advanced"])
    if "poly_bases" in old and "poly_basis_type" not in j["space"]:
        j["space"]["poly_basis_type"] = {}
        copy_entry("poly_bases", old, j["space"]["poly_basis_type"])
    copy_entry("quadrature_order", old, j["space"]["advanced"])
    copy_entry("integral_constraints", old, j["space"]["advanced"])
    copy_entry("n_harmonic_samples", old, j["space"]["advanced"])
    copy_entry("force_no_ref_for_harmonic", old, j["space"]["advanced"])
    copy_entry("B", old, j["space"]["advanced"])
    copy_entry("h1_formula", old, j["space"]["advanced"])
    copy_entry("count_flipped_els", old, j["space"]["advanced"])

    # Time
    j["time"] = {}
    copy_entry("t0", old, j["time"])
    copy_entry("tend", old, j["time"])
    copy_entry("dt", old, j["time"])
    copy_entry("time_steps", old, j["time"])
    rename_entry("time_integrator", old, "integrator", j["time"])

    if "time_integrator_params" in old:
        j["time"]["newmark"] = {}
        j["time"]["BDF"] = {}
        copy_entry("gamma", old["time_integrator_params"],
                   j["time"]["newmark"])
        copy_entry("beta", old["time_integrator_params"], j["time"]["newmark"])
        rename_entry(
            "num_steps", old["time_integrator_params"], "steps", j["time"]["BDF"])

    # Contact

    j["contact"] = {}
    rename_entry("has_collision", old, "enabled", j["contact"])
    copy_entry("dhat_percentage", old, j["contact"])
    copy_entry("dhat", old, j["contact"])
    copy_entry("epsv", old, j["contact"])
    rename_entry("mu", old, "friction_coefficient", j["contact"])
    rename_entry("coeff_friction", old, "friction_coefficient", j["contact"])
    copy_entry("collision_mesh", old, j["contact"])

    # Solver

    j["solver"] = {}
    j["solver"]["linear"] = {}

    rename_entry("solver_type", old, "solver", j["solver"]["linear"])
    rename_entry("precond_type", old, "precond", j["solver"]["linear"])

    j["solver"]["nonlinear"] = {}
    j["solver"]["nonlinear"]["line_search"] = {}

    rename_entry("nl_solver", old, "solver", j["solver"]["nonlinear"])
    if "solver_params" in old:
        rename_entry("fDelta", old["solver_params"],
                     "f_delta", j["solver"]["nonlinear"])
        rename_entry("gradNorm", old["solver_params"],
                     "grad_norm" if old["solver_params"].get(
                         "useGradNorm", False) else "x_delta",
                     j["solver"]["nonlinear"])
        rename_entry("nl_iterations", old["solver_params"],
                     "max_iterations", j["solver"]["nonlinear"])
        rename_entry("relativeGradient", old["solver_params"],
                     "relative_gradient", j["solver"]["nonlinear"])
        rename_entry("use_grad_norm_tol", old["solver_params"],
                     "use_grad_norm_tol", j["solver"]["nonlinear"]["line_search"])

    if "line_search" in old and old["line_search"] == "bisection":
        old["line_search"] = "backtracking"
    rename_entry("line_search", old, "method",
                 j["solver"]["nonlinear"]['line_search'])

    j["solver"]["augmented_lagrangian"] = {}
    rename_entry("force_al", old, "force", j["solver"]["augmented_lagrangian"])
    rename_entry("al_weight", old, "initial_weight",
                 j["solver"]["augmented_lagrangian"])
    rename_entry("max_al_weight", old, "max_weight",
                 j["solver"]["augmented_lagrangian"])

    j["solver"]["contact"] = {}

    copy_entry("friction_iterations", old, j["solver"]["contact"])
    copy_entry("friction_convergence_tol", old, j["solver"]["contact"])
    copy_entry("barrier_stiffness", old, j["solver"]["contact"])
    copy_entry("lagged_damping_weight", old, j["solver"]["contact"])

    if "solver_params" in old:
        j["solver"]["contact"]["CCD"] = {}
        rename_entry("broad_phase_method",
                     old["solver_params"], "broad_phase", j["solver"]["contact"]["CCD"])
        rename_entry("ccd_tolerance", old["solver_params"],
                     "tolerance", j["solver"]["contact"]["CCD"])
        rename_entry("ccd_max_iterations",
                     old["solver_params"], "max_iterations", j["solver"]["contact"]["CCD"])

    copy_entry("ignore_inertia", old, j["solver"])

    j["solver"]["advanced"] = {}

    copy_entry("cache_size", old, j["solver"]["advanced"])
    copy_entry("lump_mass_matrix", old, j["solver"]["advanced"])

    if "problem" in old:
        if old["problem"] == "GenericScalar" or old["problem"] == "GenericTensor":
            if "problem_params" in old:
                if "boundary_conditions" not in j:
                    j["boundary_conditions"] = {}
                copy_entry("rhs", old["problem_params"],
                           j["boundary_conditions"])
                copy_entry("dirichlet_boundary",
                           old["problem_params"], j["boundary_conditions"])
                copy_entry("neumann_boundary",
                           old["problem_params"], j["boundary_conditions"])
                copy_entry("pressure_boundary",
                           old["problem_params"], j["boundary_conditions"])

                j["initial_conditions"] = {}
                rename_entry(
                    "initial_solution", old["problem_params"], "solution", j["initial_conditions"])
                rename_entry(
                    "initial_velocity", old["problem_params"], "velocity", j["initial_conditions"])
                rename_entry(
                    "initial_acceleration", old["problem_params"], "acceleration", j["initial_conditions"])

        else:
            rename_entry("problem_params", old, "preset_problem", j)
            j["preset_problem"]["type"] = old["problem"]
    else:
        print("Missing problem assuming generic")
        if "problem_params" in old:
            if "boundary_conditions" not in j:
                j["boundary_conditions"] = {}
            copy_entry("rhs", old["problem_params"],
                       j["boundary_conditions"])
            copy_entry("dirichlet_boundary",
                       old["problem_params"], j["boundary_conditions"])
            copy_entry("neumann_boundary",
                       old["problem_params"], j["boundary_conditions"])
            copy_entry("pressure_boundary",
                       old["problem_params"], j["boundary_conditions"])

            j["initial_conditions"] = {}
            rename_entry(
                "initial_solution", old["problem_params"], "solution", j["initial_conditions"])
            rename_entry(
                "initial_velocity", old["problem_params"], "velocity", j["initial_conditions"])
            rename_entry(
                "initial_acceleration", old["problem_params"], "acceleration", j["initial_conditions"])

    # Materials

    material_name = "NeoHookean"
    if "scalar_formulation" in old:
        material_name = old["scalar_formulation"]
    elif "tensor_formulation" in old:
        material_name = old["tensor_formulation"]
    else:
        print("Warning using default material name:", material_name)

    if "params" in old:
        j["materials"] = {}
        j["materials"]["type"] = material_name
        copy_entry("lambda", old["params"], j["materials"])
        copy_entry("mu", old["params"], j["materials"])
        copy_entry("k", old["params"], j["materials"])
        copy_entry("elasticity_tensor", old["params"], j["materials"])
        copy_entry("E", old["params"], j["materials"])
        copy_entry("nu", old["params"], j["materials"])
        copy_entry("young", old["params"], j["materials"])
        copy_entry("poisson", old["params"], j["materials"])
        copy_entry("density", old["params"], j["materials"])
        copy_entry("rho", old["params"], j["materials"])
        copy_entry("alphas", old["params"], j["materials"])
        copy_entry("mus", old["params"], j["materials"])
        copy_entry("Ds", old["params"], j["materials"])

    if "body_params" in old:
        j["materials"] = []
        for o in old["body_params"]:
            n = {}
            copy_entry("id", o, n)
            copy_entry("E", o, n)
            copy_entry("nu", o, n)
            copy_entry("rho", o, n)
            n["type"] = material_name
            j["materials"].append(n)

    # Output

    j["output"] = {}

    rename_entry("output", old, "json", j["output"])
    j["output"]["paraview"] = {}
    if "export" in old:
        if "time_sequence" in old["export"]:
            rename_entry("time_sequence",
                         old["export"], "file_name", j["output"]["paraview"])
        elif old.get("save_time_sequence", False):
            j["output"]["paraview"]["file_name"] = "sim.pvd"
        else:
            rename_entry("paraview", old["export"],
                         "file_name", j["output"]["paraview"])

        j["output"]["data"] = {}
        copy_entry("solution", old["export"], j["output"]["data"])
        copy_entry("full_mat", old["export"], j["output"]["data"])
        copy_entry("stiffness_mat", old["export"], j["output"]["data"])
        copy_entry("solution_mat", old["export"], j["output"]["data"])
        copy_entry("stress_mat", old["export"], j["output"]["data"])
        copy_entry("u_path", old["export"], j["output"]["data"])
        copy_entry("v_path", old["export"], j["output"]["data"])
        copy_entry("a_path", old["export"], j["output"]["data"])
        copy_entry("mises", old["export"], j["output"]["data"])

        copy_entry("skip_frame", old["export"], j["output"]["paraview"])
        copy_entry("high_order_mesh", old["export"], j["output"]["paraview"])
        copy_entry("volume", old["export"], j["output"]["paraview"])
        copy_entry("surface", old["export"], j["output"]["paraview"])
        copy_entry("wireframe", old["export"], j["output"]["paraview"])

        j["output"]["paraview"]["options"] = {}
        rename_entry("material_params", old["export"], "material",
                     j["output"]["paraview"]["options"])
        copy_entry("body_ids", old["export"],
                   j["output"]["paraview"]["options"])
        copy_entry("contact_forces", old["export"],
                   j["output"]["paraview"]["options"])
        copy_entry("friction_forces", old["export"],
                   j["output"]["paraview"]["options"])
        copy_entry("velocity", old["export"],
                   j["output"]["paraview"]["options"])
        copy_entry("acceleration", old["export"],
                   j["output"]["paraview"]["options"])

    j["output"]["advanced"] = {}

    copy_entry("compute_error", old, j["output"]["advanced"])
    copy_entry("curved_mesh_size", old, j["output"]["advanced"])
    copy_entry("save_solve_sequence_debug", old, j["output"]["advanced"])
    copy_entry("save_time_sequence", old, j["output"]["advanced"])
    copy_entry("save_nl_solve_sequence", old, j["output"]["advanced"])
    copy_entry("vismesh_rel_area", old, j["output"]["paraview"])

    if "export" in old:
        copy_entry("sol_on_grid", old["export"], j["output"]["advanced"])
        copy_entry("sol_at_node", old["export"], j["output"]["advanced"])
        copy_entry("vis_boundary_only", old["export"], j["output"]["advanced"])
        copy_entry("nodes", old["export"], j["output"]["advanced"])
        copy_entry("spectrum", old["export"], j["output"]["advanced"])

    # Reference
    j["output"]["reference"] = {}

    if "problem_params" in old:
        rename_entry("exact", old["problem_params"],
                     "solution", j["output"]["reference"])
        rename_entry("exact_grad", old["problem_params"],
                     "gradient", j["output"]["reference"])

    if "import" in old:

        j["input"] = {}
        j["input"]["data"] = {}
        copy_entry("u_path", old["import"], j["input"]["data"])
        copy_entry("v_path", old["import"], j["input"]["data"])
        copy_entry("a_path", old["import"], j["input"]["data"])

    # Body_ids are global and are added to volume selections
    selection_entries = ["id", "axis", 'position', 'relative',
                         'normal', 'point', 'center', 'radius', 'box', 'offset']

    if "body_ids" in old:
        for t in j["geometry"]:
            t["volume_selection"] = []

        for o in old["body_ids"]:
            n = {}
            for entry in selection_entries:
                copy_entry(entry, o, n)

            for t in j["geometry"]:
                t["volume_selection"].append(n)

    # boundary_sidesets are global and are added to surface selections

    if "boundary_sidesets" in old:
        for t in j["geometry"]:
            if "is_obstacle" in t and t["is_obstacle"]:
                continue
            t["surface_selection"] = []

        for o in old["boundary_sidesets"]:
            n = {}
            for entry in selection_entries:
                copy_entry(entry, o, n)

            for t in j["geometry"]:
                if "is_obstacle" in t and t["is_obstacle"]:
                    continue
                t["surface_selection"].append(n)

    remove_empty_dicts_from_dict(j)

    return j


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert JSON params v1 to v2')
    parser.add_argument('input_json', help='input version 1 JSON param')
    parser.add_argument('output_json', help='output version 2 JSON param')
    args = parser.parse_args()

    # read old json
    with open(args.input_json, 'r') as myfile:
        data_old = myfile.read()
    old = json.loads(data_old)

    # convert
    conv = PolyFEM_convert(old)

    # save it to file
    j = json.dumps(conv, ensure_ascii=False,
                   indent=None, separators=(',', ':'))

    j = jsbeautifier.beautify(j, opts={
        "js": {
            "allowed_file_extensions": ["json", "jsbeautifyrc"],
            "brace_style": "collapse",
            "indent_char": " ",
            "indent_size": 4
        }
    })

    with open(args.output_json, 'w', encoding='utf-8') as f:
        f.write(j)
