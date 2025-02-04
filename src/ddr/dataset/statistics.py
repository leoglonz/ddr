import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

def set_statistics(cfg) -> pd.DataFrame:
    """Creating the necessary statistics for normalizing atributes

    Parameters
    ----------
    cfg: Config
        The configuration object containing the path to the data sources.
    attributes: zarr.Group
        The zarr.Group object containing attributes.

    Returns
    -------
      pl.DataFrame: A polars DataFrame containing the statistics for normalizing attributes.
    """
    attributes_name = "hydrofabric_v2.2"
    statistics_path = Path(cfg.data_sources.statistics)
    statistics_path.mkdir(exists=True)
    stats_file = statistics_path / f"attribute_statistics_{attributes_name}.json"
    if stats_file.exists():
        log.info(f"Reading Attribute Statistics from file: {stats_file.name}")
        df = pd.read_csv(str(stats_file))
    else:
        gdf = gpd.read_file(cfg.data_sources.conus_hydrofabric, layer="divide-attributes")
        data = []
        for attribute in cfg.params.attributes:
            data.append(gdf[attribute].values)
        
        json_ = {}
        for i, attr in enumerate(cfg.params.attributes):
            json_[attr] = {
                "min": np.min(data, axis=0)[i],
                "max": np.max(data, axis=0)[i],
                "mean": np.mean(data, axis=0)[i],
                "std": np.std(data, axis=0)[i],
                "p10": np.percentile(data, 10, axis=0)[i],
                "p90": np.percentile(data, 90, axis=0)[i],
            }
        df = pd.DataFrame(json_).T
        df.to_csv(str(stats_file))
    return df
