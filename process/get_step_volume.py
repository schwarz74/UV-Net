import os

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from tqdm import tqdm
import json

if __name__ == "__main__":
    dir = "../datasets/v_mock/test_step"
    save_name = "test_volume.json"
    file_volume = {}
    for file in tqdm(os.listdir(dir)):
        reader = STEPControl_Reader()
        reader.ReadFile(dir + "/" + file)
        reader.TransferRoots()
        shape = reader.OneShape()

        system = GProp_GProps()
        brepgprop.VolumeProperties(shape, system)
        volume = system.Mass()

        file_volume[file] = volume
    with open(save_name, 'w') as file:
        file.write(json.dumps(file_volume))