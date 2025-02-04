import logging
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

from ddr.nn.kan import kan
from ddr.routing.dmc import dmc
from ddr.dataset.utils import downsample
from ddr.dataset.streamflow import MeritReader as streamflow
from ddr.dataset.train_dataset import train_dataset

log = logging.getLogger(__name__)

def train(cfg, flow, routing_model, nn, optimizer):
    
    dataset = train_dataset(cfg)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    
    for epoch in range(0, cfg.train.epochs + 1):
        for i, hydrofabric in enumerate(dataloader, start=0):
            streamflow_predictions = flow(cfg=cfg, hydrofabric=hydrofabric)
            q_prime = streamflow_predictions["streamflow"] @ hydrofabric.mapping.tm

            dmc_kwargs = {
                "hydrofabric": hydrofabric,
                "spatial_parameters": nn(
                    inputs=hydrofabric.normalized_spatial_attributes.to(cfg.device)
                ),
                "streamflow": q_prime,
            }
            dmc_output = routing_model(**dmc_kwargs)

            num_days = len(dmc_output["runoff"][0][13 : (-11 + cfg.params.tau)]) // 24
            daily_runoff = downsample(
                dmc_output["runoff"][:, 13 : (-11 + cfg.params.tau)],
                rho=num_days,
            )

            nan_mask = hydrofabric.observations.isnull().any(dim="time")
            np_nan_mask = nan_mask.streamflow.values

            filtered_ds = hydrofabric.observations.where(~nan_mask, drop=True)
            filtered_observations = torch.tensor(filtered_ds.streamflow.values, device=cfg.device)[
                :, 1:-1
            ]  # Cutting off days to match with realigned timesteps

            filtered_predictions = daily_runoff[~np_nan_mask]

            loss = mse_loss(
                input=filtered_predictions.transpose(0, 1)[cfg.warmup:].unsqueeze(2),
                target=filtered_observations.transpose(0, 1)[cfg.warmup:].unsqueeze(2),
            )

            log.info("Running gradient-averaged backpropagation")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Loss: {loss.item}")



@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="training_config",
)
def main(cfg: DictConfig) -> None:
    try:
        start_time = time.perf_counter()
        nn = kan(**cfg.spatial_kan)
        routing_model = dmc(cfg)
        flow = streamflow(cfg)
        train(
            cfg=cfg,
            flow=flow,
            routing_model=routing_model,
            nn=nn
        )
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    
    finally:
        print("Cleaning up...")
    
        total_time = time.perf_counter() - start_time
        log.info(
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 
