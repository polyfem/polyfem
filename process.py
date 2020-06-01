import os
import json
import glob
from math import sqrt


if __name__ == '__main__':
    json_folder = "out"

    viscosities = {}

    for json_f in glob.glob(os.path.join(json_folder, "*.json")):
        with open(json_f, "r") as f:
            jdata = json.load(f)

        tmp = json_f.split("_")
        visc = tmp[-2]
        base_step = tmp[-3]
        if visc in viscosities:
            if base_step not in viscosities[visc]:
                viscosities[visc][base_step] = []
        else:
            viscosities[visc] = {}
            viscosities[visc][base_step] = []
        res = viscosities[visc][base_step]

        is_p = jdata["count_simplex"] > 0
        if is_p:
            N = jdata["count_simplex"]
        else:
            N = jdata["count_regular"] + jdata["count_regular_boundary"]

        letter = "P" if is_p else "Q"
        discr = jdata["discr_order"]
        p_discr = jdata["args"]["pressure_discr_order"]

        tend = jdata["args"]["tend"]
        time_steps = jdata["args"]["time_steps"]

        title = "{0:}{1:}{0:}{2:} Particle {3:}".format(
            letter, discr, p_discr, base_step)
        res.append({
            "N": int(sqrt(N)),
            "h": jdata["average_edge_length"],
            "dt": tend/time_steps,
            "time": jdata["time_solving"],
            "time_steps": time_steps,
            "error": jdata["err_l2"],
            "mem": jdata["peak_memory"],
            "title": title
        })

    for visc in viscosities:
        for base_step in viscosities[visc]:
            name = "Particle_v{}_nt{}".format(visc, base_step)
            res = viscosities[visc][base_step]
            title = res[0]["title"]

            with open(name + ".json", "w") as f:
                tmp = {"name": title, "data": res}
                f.write(json.dumps(tmp, indent=4))


