
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

    if "solver" in sim_config.keys() and \
    solver_type in sim_config["solver"].keys():
        if in_advanced and "advanced" in sim_config["solver"][solver_type].keys() \
        and tol_name in sim_config["solver"][solver_type]["advanced"].keys():
            return sim_config["solver"][solver_type]["advanced"][tol_name]
        elif tol_name in sim_config["solver"][solver_type].keys():
            return sim_config["solver"][solver_type][tol_name]
    
    current_config = sim_config
    current_dir = "/".join(in_json_directory.split("/")[:-1])
    while True:
        if "solver" in current_config and solver_type in current_config["solver"]:
            nl_solver = current_config["solver"][solver_type]
            
            if in_advanced:
                if "advanced" in nl_solver and tol_name in nl_solver["advanced"]:
                    return nl_solver["advanced"][tol_name]
            elif tol_name in nl_solver:
                return nl_solver[tol_name]
        
        if "common" in current_config:
            new_common_path = os.path.join(current_dir, current_config["common"])
            with open(new_common_path, "r") as f:
                current_config = json.load(f)
            current_dir = "/".join(new_common_path.split("/")[:-1])
        else:
            break 

    # 3. If the loop finishes without returning, apply the defaults
    if in_advanced:
        return 0
    elif tol_name == "f_delta":
        return 0
    elif tol_name == "x_delta":
        return 0
    elif tol_name == "grad_norm":
        return 1e-8
    else:
        assert False


def get_old_use_grad_norm_tol(sim_config, in_json_path, solver_type):
    assert False


def update_json(in_json_path, out_json_path, is_common_file, in_json_directory):

    with open(in_json_path, "r") as f:
        sim_config = json.load(f)

    update_field(sim_config, ["solver", "advanced", "characteristic_force_density"], 1, create_new=True)
    update_field(sim_config, ["solver", "advanced", "characteristic_length"], 1, create_new=True)
    update_field(sim_config, ["solver", "nonlinear", "norm_type"], "Euclidean", create_new=True)

    if not is_common_file:
        dt = get_old_dt(sim_config)
        old_characteristic_length = get_old_characteristic_length(sim_config)
        old_scale = dt * old_characteristic_length
        
        update_field(sim_config, ["solver", "nonlinear", "grad_norm_tol"], get_old_tol(sim_config, "grad_norm", in_json_path, "nonlinear") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "x_delta_tol"], get_old_tol(sim_config, "x_delta", in_json_path, "nonlinear") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "advanced", "f_delta_tol"], get_old_tol(sim_config, "f_delta", in_json_path, "nonlinear") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "newton_decrement_tol"], get_old_tol(sim_config, "derivative_along_delta_x", in_json_path, "nonlinear") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "line_search", "use_grad_norm_tol"], get_old_use_grad_norm_tol(sim_config, in_json_path, "nonlinear") * old_characteristic_length, create_new=True)
        
        update_field(sim_config, ["solver", "augmented_lagrangian", "grad_norm_tol"], get_old_tol(sim_config, "grad_norm", in_json_path, "augmented_lagrangian") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "augmented_lagrangian", "x_delta_tol"], get_old_tol(sim_config, "x_delta", in_json_path, "augmented_lagrangian") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "augmented_lagrangian", "advanced", "f_delta_tol"], get_old_tol(sim_config, "f_delta", in_json_path, "augmented_lagrangian") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "augmented_lagrangian", "newton_decrement_tol"], get_old_tol(sim_config, "derivative_along_delta_x", in_json_path, "augmented_lagrangian") * old_scale, create_new=True)
        update_field(sim_config, ["solver", "augmented_lagrangian", "line_search", "use_grad_norm_tol"], get_old_use_grad_norm_tol(sim_config, in_json_path, "augmented_lagrangian") * old_characteristic_length, create_new=True)

    sim_config["solver"]["nonlinear"].pop("grad_norm", None)
    sim_config["solver"]["nonlinear"].pop("x_delta", None)
    sim_config["solver"]["nonlinear"].pop("f_delta", None)
    sim_config["solver"]["augmented_lagrangian"].pop("grad_norm", None)
    sim_config["solver"]["augmented_lagrangian"].pop("x_delta", None)
    sim_config["solver"]["augmented_lagrangian"].pop("f_delta", None)
    if "advanced" in sim_config["solver"]["nonlinear"].keys():
        sim_config["solver"]["nonlinear"]["advanced"].pop("derivative_along_delta_x", None)
    if "advanced" in sim_config["solver"]["augmented_lagrangian"].keys():
        sim_config["solver"]["augmented_lagrangian"]["advanced"].pop("derivative_along_delta_x", None)

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
                    False, 
                    args.input_directory
                )

    for (input_path, output_path) in common_files:
        update_json(
            input_path,
            output_path,
            True,
            args.input_directory
        )


if __name__ == "__main__":
    main()