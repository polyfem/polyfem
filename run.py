import os
import json
import glob
import subprocess
import tempfile

if __name__ == '__main__':
    polyfem_exe = os.path.join("C:/Users/zizhou/Documents/GitHub/polyfem/build/Release", "PolyFEM_bin")
    vtu_folder = "vtu"
    json_folder = "out"

    discr_order = 1
    p_discr_order = 1
    viscosities = [1, 1e-3]

    n_refs = [0, 1, 2]
    base_ref = 0
    base_steps = [32]

    ext = "obj"

    folder_path = "meshes"
    current_folder = cwd = os.getcwd()

    with open("OperatorSplitting.json", 'r') as f:
        json_data = json.load(f)

    for viscosity in viscosities:
        json_data["problem_params"]["viscosity"] = viscosity
        json_data["viscosity"] = viscosity

        for base_step in base_steps:
            for mesh in glob.glob(os.path.join(folder_path, "*." + ext)):
                basename = os.path.splitext(os.path.basename(mesh))[0]
                mesh = os.path.join(current_folder, mesh)
                title = basename

                json_data["mesh"] = mesh

                json_data["discr_order"] = discr_order
                json_data["pressure_discr_order"] = p_discr_order

                for ref in n_refs:
                    fname = title + "_" + str(base_step) + "_" + str(viscosity) + "_" + str(ref)

                    if title == "2pi":
                        json_data["n_refs"] = ref+base_ref
                    else:
                        json_data["n_refs"] = ref+base_ref+2
                    json_data["time_steps"] = int(base_step * 2**ref)

                    json_data["output"] = os.path.join(current_folder, json_folder, fname + ".json")

                    with tempfile.NamedTemporaryFile(suffix=".json",delete=False) as tmp_json:
                        with open(tmp_json.name, 'w') as f:
                            f.write(json.dumps(json_data, indent=4))

                        tempFile = tmp_json.name

                        args = [polyfem_exe,
                                '--json', tmp_json.name,
                                '--cmd',
                                '--log_level', '2']

                        subprocess.run(args)

                    os.remove(tempFile)
