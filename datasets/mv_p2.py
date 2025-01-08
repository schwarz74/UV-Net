import pathlib
import string
import json
import os
from operator import itemgetter

import torch
from sklearn.model_selection import train_test_split


from datasets.base import BaseDataset

class MV_P2(BaseDataset):
    
    def __init__(self, root_dir, split="train", center_and_scale=True, random_rotate=False,):
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)
        self.path = path
        self.random_rotate = random_rotate
                
        if split in ("train", "val"):
            self.cad_dict = self._get_cad_time_name_dict(root_dir, "train")
            file_paths = self._get_filenames(path, list(self.cad_dict.keys()))

            train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)
            if split == "train":
                file_paths = train_files
            elif split == "val":
                file_paths = val_files
        elif split == "test":
            self.cad_dict = self._get_cad_time_name_dict(root_dir, "test")
            file_paths = self._get_filenames(path, list(self.cad_dict.keys()))
        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)
        print("Done loading {} files".format(len(self.data)))
    
    def _get_cad_time_name_dict(self, root_dir, split):
        with open(f"{root_dir}/{split}_cad_time.json", 'r') as file:
            return json.load(file)
    
    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally get the label from the filename and store it in the sample dict
        sample["label"] = torch.tensor([self.cad_dict[file_path.stem]]).float()
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        return collated
    
    def _get_filenames(self, root_dir, filelist):
        files = list(
            x
            for x in root_dir.rglob(f"*.bin")
            if x.stem in filelist
        )
        return files