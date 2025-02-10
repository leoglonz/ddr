from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

from ddr.dataset.Dates import Dates
from ddr.dataset.observations import ZarrUSGSReader
from ddr.dataset.statistics import set_statistics
from ddr.dataset.utils import read_coo

log = logging.getLogger(__name__)


@dataclass
class Hydrofabric:
    spatial_attributes: Union[torch.Tensor, None] = field(default=None)
    length: Union[torch.Tensor, None] = field(default=None)
    slope: Union[torch.Tensor, None] = field(default=None)
    dates: Union[Dates, None] = field(default=None)
    normalized_spatial_attributes: Union[torch.Tensor, None] = field(default=None)
    observations: Union[xr.Dataset, None] = field(default=None)

    
def create_hydrofabric_observations(
    dates: Dates,
    gage_ids: np.ndarray,
    observations: xr.Dataset,
) -> xr.Dataset:
    ds = observations.sel(time=dates.batch_daily_time_range, gage_id=gage_ids)
    return ds


class train_dataset(torch.utils.data.Dataset):
    """train_dataset class for handling dataset operations for training dMC models"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dates = Dates(**self.cfg.train)

        self.obs_reader = ZarrUSGSReader(cfg=self.cfg)
        self.observations = self.obs_reader.read_data(dates=self.dates)
        self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])

        self.network = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="network")
        self.divides = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divides").set_index("id")
        self.divide_attr = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divide-attributes").set_index("divide_id")
        self.flowpath_attr = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpath-attributes-ml").set_index("id")
        self.flowpaths = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpaths").set_index("id")
        self.nexus = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="nexus")

         # TODO add logic for multiple gauges
        self.adjacency_matrix, self.order = read_coo(Path(cfg.data_sources.network), self.gage_ids[0])
        
        ordered_index = [f"wb-{_id}" for _id in self.order]
        self.divides_sorted = self.divides.reindex(ordered_index)
        self.divide_attr_sorted = self.divide_attr.reindex(self.divides_sorted["divide_id"])
        self.flowpaths_sorted = self.flowpaths.reindex(ordered_index)
        self.flowpath_attr = self.flowpath_attr[~self.flowpath_attr.index.duplicated(keep='first')]
        self.flowpath_attr_sorted = self.flowpath_attr.reindex(ordered_index)
        # self.idx_mapper = {_id: idx for idx, _id in enumerate(self.divides_sorted.index)}
        # self.catchment_mapper = {_id : idx for idx, _id in enumerate(self.divides_sorted["divide_id"])}
        
        self.length = torch.tensor(self.flowpath_attr["Length_m"].values, dtype=torch.float32)
        self.slope = torch.tensor(self.flowpath_attr["So"].values, dtype=torch.float32)
        self.width = torch.tensor(self.flowpath_attr["TopWdth"].values, dtype=torch.float32)
        self.x = torch.tensor(self.flowpath_attr["MusX"].values, dtype=torch.float32)
    
        self.attribute_stats = set_statistics(self.cfg)

        # Convert to tensor after collecting all valid data        
        self.means = torch.tensor([self.attribute_stats[attr].iloc[2] for attr in self.cfg.kan.input_var_names], device=self.cfg.device, dtype=torch.float32).unsqueeze(1)  # Mean is always idx 2
        self.stds = torch.tensor([self.attribute_stats[attr].iloc[3] for attr in self.cfg.kan.input_var_names], device=self.cfg.device, dtype=torch.float32).unsqueeze(1)  # Mean is always idx 3

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return 1

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        self.dates.calculate_time_period()
        
        spatial_attributes = torch.tensor(
            np.array([self.divide_attr[attr].values for attr in self.cfg.kan.input_var_names]),
            device=self.cfg.device,
            dtype=torch.float32
        )
        
        normalized_spatial_attributes = (spatial_attributes - self.means) / self.stds
        
        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=self.gage_ids,
            observations=self.observations,
        )
        
        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=self.length,
            slope=self.slope,
            dates=self.dates,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=hydrofabric_observations,
        )
