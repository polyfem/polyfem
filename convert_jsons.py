
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
    

def get_old_tol(sim_config, tol_name, in_json_directory):

    in_advanced = tol_name == "derivative_along_delta_x"

    if "solver" in sim_config.keys() and \
    "nonlinear" in sim_config["solver"].keys():
        if in_advanced and "advanced" in sim_config["solver"]["nonlinear"].keys() \
        and tol_name in sim_config["solver"]["nonlinear"]["advanced"].keys():
            return sim_config["solver"]["nonlinear"]["advanced"][tol_name]
        elif tol_name in sim_config["solver"]["nonlinear"].keys():
            return sim_config["solver"]["nonlinear"][tol_name]
    
    common_path = sim_config["common"]
    with open(os.path.join(in_json_directory, common_path), "r") as f:
        common_config = json.load(f)

    if in_advanced:
        return common_config["solver"]["nonlinear"]["advanced"][tol_name]
    else:
        return common_config["solver"]["nonlinear"][tol_name]


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
        
        update_field(sim_config, ["solver", "nonlinear", "grad_norm_tol"], get_old_tol(sim_config, "grad_norm", in_json_directory) * old_scale, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "x_delta_tol"], get_old_tol(sim_config, "x_delta", in_json_directory) * old_scale, create_new=True)
        update_field(sim_config, ["solver", "nonlinear", "advanced", "derivative_along_delta_x_tol"], get_old_tol(sim_config, "derivative_along_delta_x", in_json_directory) * old_scale, create_new=True)

    sim_config["solver"]["nonlinear"].pop("grad_norm", None)
    sim_config["solver"]["nonlinear"].pop("x_delta", None)
    sim_config["solver"]["nonlinear"]["advanced"].pop("derivative_along_delta_x", None)

    with open(out_json_path, 'w') as f:
        json.dump(sim_config, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", help="directory containing old json scripts", required=True)
    parser.add_argument("-o", "--output_directory", help="directory where updated jsons will be saved", required=True)
    parser.add_argument("-c", "--common_file", help="common json file (if it exists)", required=False)
    args = parser.parse_args()

    for file_name in os.listdir(args.input_directory):
        if file_name.endswith(".json"):
            print(f"Updating file {file_name}")
            update_json(
                os.path.join(args.input_directory, file_name), 
                os.path.join(args.output_directory, file_name),
                file_name == args.common_file, 
                args.input_directory
            )


if __name__ == "__main__":
    main()