
import argparse
import json
import os


def update_field(config, field_lst, new_val, create_new=False):
    local_config = config
    for field in field_lst[:-1]:
        if not field in local_config.keys():
            if not create_new:
                return
            else:
                local_config[field] = {}
        local_config = local_config[field]
        
    local_config[field_lst[-1]] = new_val

    
def get_old_dt(sim_config):
    if not "time" in sim_config.keys():
        return 1
    
    if "dt" in sim_config["time"].keys():
        return sim_config["time"]["dt"]
    
    return sim_config["time"]["tend"] / sim_config["time"]["time_steps"]
    

def get_old_characteristic_length(sim_config):
    if "units" in sim_config.keys() and "characteristic_length" in sim_config["units"].keys():
        return sim_config["units"]["characteristic_length"]
    else:
        return 1  
    

def get_old_tol(sim_config, tol_name, in_json_directory, solver_type):
    in_advanced = tol_name == "derivative_along_delta_x"
    in_line_search = tol_name == "use_grad_norm_tol"

    if solver_type == "nonlinear":
        try:
            nl_config = sim_config["solver"]["nonlinear"]
        except KeyError:
            nl_config = {}
    elif solver_type == "augmented_lagrangian":
        try:
            nl_config = sim_config["solver"]["augmented_lagrangian"]["nonlinear"]
        except KeyError:
            nl_config = {}

    if in_advanced and "advanced" in nl_config.keys() and tol_name in nl_config["advanced"].keys():
        return nl_config["advanced"][tol_name], True
    elif in_line_search and "line_search" in nl_config.keys() and tol_name in nl_config["line_search"].keys():
        return nl_config["line_search"][tol_name], True
    elif tol_name in nl_config.keys():
        return nl_config[tol_name], True
    
    if not "common" in sim_config.keys():
        if tol_name == "derivative_along_delta_x":
            return 0, False
        elif tol_name == "f_delta":
            return 0, False
        elif tol_name == "x_delta":
            return 0, False
        elif tol_name == "grad_norm":
            return 1e-8, False
        elif tol_name == "use_grad_norm_tol":
            return 1e-6, False
        elif tol_name == "first_grad_norm_tol":
            return 1e-10, False
        else:
            assert False
    
    current_dir = "/".join(in_json_directory.split("/")[:-1])
    common_json_path = os.path.join(current_dir, sim_config["common"])
    with open(common_json_path, "r") as f:
        return get_old_tol(json.load(f), tol_name, common_json_path, solver_type)
    

def has_specified_augmented_lagrangian_params(sim_config, in_json_directory):
    if "solver" in sim_config.keys() and "augmented_lagrangian" in sim_config["solver"].keys():
        return True
    elif not "common" in sim_config.keys():
        return False
    else:
        current_dir = "/".join(in_json_directory.split("/")[:-1])
        common_json_path = os.path.join(current_dir, sim_config["common"])
        with open(common_json_path, "r") as f:
            return has_specified_augmented_lagrangian_params(json.load(f), common_json_path)


def update_json(in_json_path, out_json_path, is_common_file):

    with open(in_json_path, "r") as f:
        sim_config = json.load(f)

    if not is_common_file:
        update_field(sim_config, ["solver", "advanced", "characteristic_force_density"], 1, create_new=True)
        update_field(sim_config, ["solver", "advanced", "characteristic_length"], 1, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "norm_type"], "Euclidean", create_new=True)

        dt = get_old_dt(sim_config)
        old_characteristic_length = get_old_characteristic_length(sim_config)
        old_scale = dt * old_characteristic_length

        old_first_grad_norm_tol = get_old_tol(sim_config, "first_grad_norm_tol", in_json_path, "nonlinear")[0] * old_scale
        old_grad_norm_tol = get_old_tol(sim_config, "grad_norm", in_json_path, "nonlinear")[0] * old_scale
        old_x_delta_tol = get_old_tol(sim_config, "x_delta", in_json_path, "nonlinear")[0] * old_scale
        old_f_delta_tol = get_old_tol(sim_config, "f_delta", in_json_path, "nonlinear")[0] * old_scale
        old_derivative_along_delta_x_tol = get_old_tol(sim_config, "derivative_along_delta_x", in_json_path, "nonlinear")[0] * old_scale
        old_use_grad_norm_tol = get_old_tol(sim_config, "use_grad_norm_tol", in_json_path, "nonlinear")[0] * old_scale
        
        update_field(sim_config, ["solver", "nonlinear", "rel_grad_norm_tol"], 0, create_new=True)
        if old_first_grad_norm_tol != 1e-12:
            update_field(sim_config, ["solver", "nonlinear", "first_grad_norm_tol"], old_first_grad_norm_tol, create_new=True)
        
        if old_grad_norm_tol != 1e-10:
            update_field(sim_config, ["solver", "nonlinear", "grad_norm_tol"], old_grad_norm_tol, create_new=True)

        if old_x_delta_tol != 0:
            update_field(sim_config, ["solver", "nonlinear", "x_delta_tol"], old_x_delta_tol, create_new=True)

        if old_f_delta_tol != 0:
            update_field(sim_config, ["solver", "nonlinear", "advanced", "f_delta_tol"], old_f_delta_tol, create_new=True)
        
        if old_derivative_along_delta_x_tol != 0:
            update_field(sim_config, ["solver", "nonlinear", "advanced", "derivative_along_delta_x_tol"], old_derivative_along_delta_x_tol, create_new=True)
        
        if old_use_grad_norm_tol != 1e-6:
            update_field(sim_config, ["solver", "nonlinear", "line_search", "use_grad_norm_tol"], old_use_grad_norm_tol, create_new=True)
        
        if has_specified_augmented_lagrangian_params(sim_config, in_json_path):
            update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "rel_grad_norm_tol"], 0, create_new=True)
            old_al_first_grad_norm_tol = get_old_tol(sim_config, "first_grad_norm_tol", in_json_path, "augmented_lagrangian")
            if old_al_first_grad_norm_tol[1] and old_al_first_grad_norm_tol[0] * old_scale != 1e-12:
                update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "first_grad_norm_tol"], old_al_first_grad_norm_tol[0] * old_scale, create_new=True)

            old_al_grad_norm_tol = get_old_tol(sim_config, "grad_norm", in_json_path, "augmented_lagrangian")
            if old_al_grad_norm_tol[1] and old_al_grad_norm_tol[0] * old_scale != 1e-10:
                update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "grad_norm_tol"], old_al_grad_norm_tol[0] * old_scale, create_new=True)

            old_al_x_delta_tol = get_old_tol(sim_config, "x_delta", in_json_path, "augmented_lagrangian")
            if old_al_x_delta_tol[1] and old_al_x_delta_tol[0] * old_scale != 0:
                update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "x_delta_tol"], old_al_x_delta_tol[0] * old_scale, create_new=True)

            old_al_f_delta_tol = get_old_tol(sim_config, "f_delta", in_json_path, "augmented_lagrangian")
            if old_al_f_delta_tol[1] and old_al_f_delta_tol[0] * old_scale != 0:
                update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "advanced", "f_delta_tol"], old_al_f_delta_tol[0] * old_scale, create_new=True)

            old_al_derivative_along_delta_x_tol = get_old_tol(sim_config, "derivative_along_delta_x", in_json_path, "augmented_lagrangian")
            if old_al_derivative_along_delta_x_tol[1] and old_al_derivative_along_delta_x_tol[0] * old_scale != 0:
                update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "advanced", "derivative_along_delta_x"], old_al_derivative_along_delta_x_tol[0] * old_scale, create_new=True)

            old_al_use_grad_norm_tol = get_old_tol(sim_config, "use_grad_norm_tol", in_json_path, "augmented_lagrangian")
            if old_al_use_grad_norm_tol[1] and old_al_use_grad_norm_tol[0] * old_scale != 1e-6:
                update_field(sim_config, ["solver", "augmented_lagrangian", "nonlinear", "line_search", "use_grad_norm_tol"], old_al_use_grad_norm_tol[0] * old_scale, create_new=True)

    if "solver" in sim_config.keys() and "nonlinear" in sim_config["solver"].keys():
        sim_config["solver"]["nonlinear"].pop("grad_norm", None)
        sim_config["solver"]["nonlinear"].pop("x_delta", None)
        sim_config["solver"]["nonlinear"].pop("f_delta", None)
        if "advanced" in sim_config["solver"]["nonlinear"].keys():
            sim_config["solver"]["nonlinear"]["advanced"].pop("derivative_along_delta_x", None)

    if "solver" in sim_config.keys() and "augmented_lagrangian" in sim_config["solver"].keys() and "nonlinear" in sim_config["solver"]["augmented_lagrangian"].keys():
        sim_config["solver"]["augmented_lagrangian"]["nonlinear"].pop("grad_norm", None)
        sim_config["solver"]["augmented_lagrangian"]["nonlinear"].pop("x_delta", None)
        sim_config["solver"]["augmented_lagrangian"]["nonlinear"].pop("f_delta", None)
        if "advanced" in sim_config["solver"]["augmented_lagrangian"]["nonlinear"].keys():
            sim_config["solver"]["augmented_lagrangian"]["nonlinear"]["advanced"].pop("derivative_along_delta_x", None)

    with open(out_json_path, 'w') as f:
        json.dump(sim_config, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", help="directory containing old json scripts", required=True)
    parser.add_argument("-o", "--output_directory", help="directory where updated jsons will be saved", required=True)
    parser.add_argument("-c", "--common_file", help="common json file (if it exists)", required=False)
    args = parser.parse_args()

    common_files = []
    for root, dirs, files in os.walk(args.input_directory):
        for file_name in files:
            if file_name.endswith(".json"):
                input_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, args.input_directory)

                output_dir_path = os.path.join(args.output_directory, relative_path)
                os.makedirs(output_dir_path, exist_ok=True)
                output_path = os.path.join(output_dir_path, file_name)
                
                if args.common_file == file_name:
                    common_files.append((input_path, output_path))
                    continue

                print(f"Updating file {file_name} in {relative_path}")
                update_json(
                    input_path, 
                    output_path,
                    False
                )

    for (input_path, output_path) in common_files:
        update_json(
            input_path,
            output_path,
            True
        )


if __name__ == "__main__":
    main()