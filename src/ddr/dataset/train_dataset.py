from dataclasses import dataclass, field
import logging
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class Hydrofabric:
    leakance_attributes: Union[torch.Tensor, None] = field(default=None)


class GeneralDataset(torch.utils.data.Dataset):
    """GeneralDataset class for handling dataset operations for training dMC models"""

    def __init__(self):
        data_path = "/Users/taddbindas/projects/ddr/data/SRB.gpkg"
        self.network = gpd.read_file(data_path, layer="network")
        self.divides = gpd.read_file(data_path, layer="divides")
        self.flowpath_attr = gpd.read_file(data_path, layer="flowpath-attributes-ml")
        self.flowpaths = gpd.read_file(data_path, layer="flowpaths")
        self.nexus = gpd.read_file(data_path, layer="nexus")
        
        self.flowpaths_sorted = self.flowpaths.sort_values('areasqkm')
        self.idx_mapper = {_id: idx for idx, _id in enumerate(self.flowpaths_sorted["id"])}
        self.network = np.zeros([len(self.idx_mapper), len(self.idx_mapper)])
        
        for idx, _id in enumerate(self.flowpaths_sorted["id"]):
            to_id = self.flowpaths_sorted.iloc[idx]["toid"]
            next_id = self.nexus[self.nexus["id"] == to_id]["toid"]
            for __id in next_id:
                col = idx
                row = self.idx_mapper[__id]
                self.network[row, col] = 1      
                
        self.length = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["Length_m"]] * 1000 for _id in self.flowpaths_sorted["id"])  
        self.slope = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["So"]] for _id in self.flowpaths_sorted["id"])      
        self.width = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["TopWdth"]] for _id in self.flowpaths_sorted["id"])      
    

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return 1

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=self.length,
            slope=self.slope,
            dates=self.dates,
            mapping=self.idx_mapper,
            network=self.network,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=hydrofabric_observations,
        )
