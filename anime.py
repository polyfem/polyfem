import numpy as np
import json, os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# idx = 0
# for dirichlet in range(0, 40):
#     with open('homo-dirichlet.json') as f:
#         args = json.load(f)
#     for arg in args["boundary_conditions"]["dirichlet_boundary"]:
#         arg["value"] = dirichlet / 20.0 * np.sign(arg["value"])
        

#     print(args)
#     with open('homo-dirichlet-tmp.json','w') as f:
#         json.dump(args, f, indent = 6, cls=NumpyEncoder)
    
#     os.system("./build/release/PolyFEM_bin -j homo-dirichlet-tmp.json")
#     os.system("mv homo.vtu step_" + str(idx) + ".vtu")
#     idx += 1

idx = 0
for def_grad in range(0, 50):
    with open('homo.json') as f:
        args = json.load(f)
        args["materials"]["def_grad"] = [[0,0],[0,-def_grad/100.0]]
        

    print(args)
    with open('homo-tmp.json','w') as f:
        json.dump(args, f, indent = 6, cls=NumpyEncoder)
    
    os.system("./build/release/homogenization -j homo-tmp.json --ns")
    os.system("mv homo.vtu step_" + str(idx) + ".vtu")
    idx += 1