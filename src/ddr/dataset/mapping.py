import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple, Union

import binsparse
import numpy as np
import numpy.typing as npt
import torch
import zarr
from tqdm import tqdm

from dMC.conf.config import Config
from dMC.dataset_modules.utils.Dates import Dates
from dMC.dataset_modules.utils.Network import FullZoneNetwork, LargeGageNetwork, Network

log = logging.getLogger(__name__)

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)


def get_zone_indices(edge_order, global_to_zone_mapping) -> DefaultDict[str, List[int]]:
    """Gets the zone indices of the MERIT TM data

    Parameters
    ----------
    edge_order : List[int]
        The edge order of the network
    global_to_zone_mapping : Dict[int, str]
        The global to zone mapping of the network

    Returns
    -------
    DefaultDict[str, List[int]]
        A dictionary of the zone indices
    """
    zone_indices = [global_to_zone_mapping[global_idx] for global_idx in edge_order]
    list_zone_ids = defaultdict(list)
    [list_zone_ids[zone].append(idx) for zone, idx in zone_indices]
    return list_zone_ids


class MeritMap:
    """A class to map the MERIT TM data to the network matrix"""

    def __init__(self, cfg: Config, dates: Dates, network: Union[FullZoneNetwork, Network, LargeGageNetwork]) -> None:
        super().__init__()
        self.cfg = cfg
        self.dates = dates
        self.network = network
        self.comid_indices_dict = {}
        self.comid_map: Dict[Tuple[str, np.int16]] = {}
        self.tm = torch.empty(0)
        self.write_mapping()

    def get_indices(
        self, zone: str, zone_idx: List[int]
    ) -> Tuple[
        npt.NDArray[np.int16],
        npt.NDArray[np.int16],
        npt.NDArray[np.int16],
    ]:
        """Gets the indices of the MERIT TM data

        Parameters
        ----------
        zone : str
            The zone of the MERIT TM data
        zone_idx : List[int]
            The zone index of the MERIT TM data

        Returns
        -------
        Tuple[npt.NDArray[np.int16], npt.NDArray[np.int16], npt.NDArray[np.int16]]
            The sorted zone index, the comid indices, and the edge indices
        """
        _zone_idx_array = np.array(zone_idx)
        _zone_attributes = self.network.global_attributes[zone]
        _segment_idx = _zone_attributes.segment_sorting_index[zone_idx]
        _sorted_indices = np.argsort(_zone_idx_array)
        _sorted_zone_idx = _zone_idx_array[_sorted_indices]
        edge_indices = np.zeros_like(_sorted_indices)
        edge_indices[_sorted_indices] = np.arange(len(_sorted_indices))
        comid_indices = np.sort(np.unique(_segment_idx))

        return _sorted_zone_idx, comid_indices, edge_indices

    def read_data(
        self,
        zone: str,
        sorted_idx: npt.NDArray[np.int16],
        comid_indices: npt.NDArray[np.int16],
        edge_indices: npt.NDArray[np.int16],
    ) -> torch.Tensor:
        """Reads the MERIT TM data and returns a data mapping

        Parameters
        ----------
        zone : str
            The zone of the MERIT TM data
        sorted_idx : npt.NDArray[np.int16]
            The sorted index of the zone
        comid_indices : npt.NDArray[np.int16]
            The comid indices of the zone
        edge_indices : npt.NDArray[np.int16]
            The edge indices of the zone

        Returns
        -------
        torch.Tensor
            A torch tensor of the MERIT TM data
        """
        try:
            zarr_group = zarr.open_group(
                Path(f"{self.cfg.data_sources.MERIT_TM}/MERIT_FLOWLINES_{zone}"),
                mode="r",
            )
        except FileNotFoundError as e:
            msg = f"Cannot find the MERIT TM {self.cfg.data_sources.MERIT_TM}/MERIT_FLOWLINES_{zone}."
            log.exception(msg=msg)
            raise FileNotFoundError(msg) from e
        comid_indices_reshaped = comid_indices.reshape(-1, 1)
        merit_to_edge_tm = zarr_group.TM.vindex[comid_indices_reshaped, sorted_idx][:, edge_indices]
        return torch.tensor(merit_to_edge_tm, dtype=torch.float64)

    def read_sparse_data(
        self,
        zone: str,
        sorted_idx: npt.NDArray[np.int16],
        comid_indices: npt.NDArray[np.int16],
        edge_indices: npt.NDArray[np.int16],
    ) -> torch.Tensor:
        """Reads the sparse MERIT TM data and returns a torch sparse tensor

        Parameters
        ----------
        zone : str
            The zone of the MERIT TM data
        sorted_idx : npt.NDArray[np.int16]
            The sorted index of the zone
        comid_indices : npt.NDArray[np.int16]
            The comid indices of the zone
        edge_indices : npt.NDArray[np.int16]
            The edge indices of the zone

        Returns
        -------
        torch.Tensor
            A torch sparse tensor of the sparse MERIT TM data
        """
        try:
            zarr_group = zarr.open_group(
                Path(f"{self.cfg.data_sources.MERIT_TM}/sparse_MERIT_FLOWLINES_{zone}"),
                mode="r",
            )
        except FileNotFoundError as e:
            msg = f"Cannot find the MERIT TM {self.cfg.data_sources.MERIT_TM}/sparse_MERIT_FLOWLINES_{zone}."
            log.exception(msg=msg)
            raise FileNotFoundError(msg) from e
        comid_indices_reshaped = comid_indices.reshape(-1, 1)
        sparse_zone = binsparse.read(zarr_group["TM"])
        subset_sparse = sparse_zone[comid_indices_reshaped, sorted_idx][:, edge_indices]
        return torch.sparse_csr_tensor(
            crow_indices=torch.tensor(subset_sparse.indptr),
            col_indices=torch.tensor(subset_sparse.indices),
            values=torch.tensor(subset_sparse.data),
            size=subset_sparse.shape,
        ).to_dense()

    def write_mapping(self) -> None:
        """Maps the MERIT TM to the network matrix"""
        list_zone_ids = get_zone_indices(self.network.edge_order, self.network.global_to_zone_mapping)
        tms = defaultdict(torch.Tensor)
        for zone, _zone_idx in tqdm(list_zone_ids.items(), desc="\rReading MERIT TM", ncols=140, ascii=True):
            _sorted_zone_idx, comid_indices, edge_indices = self.get_indices(zone, _zone_idx)
            tms[zone] = self.read_sparse_data(zone, _sorted_zone_idx, comid_indices, edge_indices)
            # tms[zone] = self.read_data(
            #     zone, _sorted_zone_idx, comid_indices, edge_indices
            # )
            self.comid_indices_dict[zone] = comid_indices
        comid_indices_flattened = [
            (zone, comid_idx) for zone, comid_indices in self.comid_indices_dict.items() for comid_idx in comid_indices
        ]
        self.comid_map = {comid_indices_flattened[i]: i for i in range(len(comid_indices_flattened))}
        self.tm = torch.zeros(
            (len(comid_indices_flattened), len(self.network.edge_order)),
        )
        for zone, _zone_idx in tqdm(
            list_zone_ids.items(),
            desc="\rMapping merit tm to matrix",
            ncols=140,
            ascii=True,
        ):
            zone_tuple = [(zone, item) for item in _zone_idx]
            global_idx = [self.network.zone_to_global_mapping[tuple_] for tuple_ in zone_tuple]
            subset_idx = np.array([self.network.global_to_subset_mapping[item] for item in global_idx])
            rows = np.array([self.comid_map[(zone, comid)] for comid in self.comid_indices_dict[zone]]).reshape(-1, 1)
            self.tm[rows, subset_idx] = tms[zone].to(torch.float32)

        self.tm = self.tm.to_sparse_coo()
