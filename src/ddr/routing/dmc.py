"""Differentiable Muskingum-Cunge"""
import logging

import torch
from omegaconf import DictConfig
from torch.linalg import solve_triangular
from tqdm import tqdm

from ddr.routing.utils import (
    PatternMapper,
    RiverNetworkMatrix,
    denormalize,
)

log = logging.getLogger(__name__)

def _log_base_q(x, q):
    return torch.log(x) / torch.log(torch.tensor(q, dtype=x.dtype))

def _get_velocity(q_t, _n, _p_spatial, width, _q_spatial, _s0, velocity_lb, depth_lb) -> torch.Tensor:
    """Calculate flow velocity using Manning's equation.

    Parameters
    ----------
    q_t : torch.Tensor
        Discharge at time t.
    _n : torch.Tensor
        Manning's roughness coefficient.
    _q_spatial : torch.Tensor
        Spatial discharge parameter.
    _s0 : torch.Tensor
        Channel slope.
    p_spatial : torch.Tensor
        Spatial parameter for width calculation.

    Returns
    -------
    torch.Tensor
        Celerity (wave speed) of the flow.

    Notes
    -----
    The function first calculates flow depth using Manning's equation, then
    computes velocity and finally celerity. The celerity is clamped between
    0.3 and 15 m/s and scaled by 5/3 according to kinematic wave theory.
    """
    depth = _log_base_q(width/_p_spatial, _q_spatial)
    v = torch.div(1, _n) * torch.pow(depth, (2 / 3)) * torch.pow(_s0, (1 / 2))
    c_ = torch.clamp(v, velocity_lb, 15)
    c = c_ * 5 / 3
    return c

class dmc(torch.nn.Module):
    """
    dMC is a differentiable implementation of the Muskingum Cunge River rouing function

    This class implements the forward pass for the dMC model, including the setup of
    various parameters and the handling of reservoir routing if needed.
    """

    def __init__(
        self,
        cfg: dict[str, any] | DictConfig, 
        device: str | None = "cpu"
    ):
        super().__init__()
        self.cfg = cfg
        
        self.device_num = device

        self.t = torch.tensor(
            3600.0,
            device=self.device_num,
        )
        
        # Base routing parameters
        self.n = None
        self.q_spatial = None
        self.p_spatial = None

        # Routing state
        self.length = None
        self.slope = None
        self.velocity = None
        self._discharge_t = None
        self.adjacency_matrix = None

        self.parameter_bounds = self.cfg.params.parameter_ranges.range
        self.velocity_lb = torch.tensor(self.cfg.params.attribute_minimums.velocity, device=self.device_num)
        self.depth_lb = torch.tensor(self.cfg.params.attribute_minimums.depth, device=self.device_num)
        self.discharge_lb = torch.tensor(self.cfg.params.attribute_minimums.discharge, device=self.device_num)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """The forward pass for the dMC model

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the runoff, zeta, and reservoir storage tensors
        """
        # Setup solver and get hydrofabric data

        hydrofabric = kwargs["hydrofabric"]
        q_prime = kwargs["streamflow"].to(self.device_num)
        observations = hydrofabric.observations.gage_id
        # gage_information = hydrofabric.network.gage_information
        # TODO: create a dynamic gauge look up
        gage_indices = torch.tensor([-1])
        self.adjacency_matrix = hydrofabric.adjacency_matrix

        # Set up base parameters
        self.n = denormalize(value=kwargs["spatial_parameters"]["n"], bounds=self.parameter_bounds["n"])
        self.q_spatial = denormalize(
            value=kwargs["spatial_parameters"]["q_spatial"],
            bounds=self.parameter_bounds["q_spatial"],
        )
        self.p_spatial = denormalize(
            value=kwargs["spatial_parameters"]["p_spatial"],
            bounds=self.parameter_bounds["p_spatial"],
        )

        # Initialize discharge
        self._discharge_t = q_prime[0].to(self.device_num)

        # Setup output tensors
        output = torch.zeros(
            size=[observations.shape[0], q_prime.shape[0]],
            device=torch.device(self.device_num),
        )

        # Initialize mapper
        matrix_dims = self.network.shape[0]
        mapper = PatternMapper(self.fill_op, matrix_dims, device=self.device_num)

        # Set initial output values
        if len(self._discharge_t) != 0:
            for i, gage_idx in enumerate(gage_indices):
                output[i, 0] = torch.sum(self._discharge_t[gage_idx])
        else:
            for i, gage_idx in enumerate(gage_indices):
                output[i, 0] = q_prime[0, gage_idx]
        output[:, 0] = torch.clamp(input=output[:, 0], min=self.discharge_lb)

        # Get spatial attributes
        length = hydrofabric.length.to(self.device_num).to(torch.float32)
        slope = torch.clamp(
            hydrofabric.slope.to(self.device_num).to(torch.float32),
            min=self.cfg.params.attribute_minimums.slope,
        )
        width = hydrofabric.length.to(self.device_num).to(torch.float32)
        x_storage = hydrofabric.length.to(self.device_num).to(torch.float32)

        desc = "Running dMC Routing"
        for timestep in tqdm(
            range(1, len(q_prime)),
            desc=f"\r{desc} for"
            f"Epoch: {self.epoch} | "
            f"Mini Batch: {self.mini_batch} | ",
            ncols=140,
            ascii=True,
        ):
            q_prime_sub = q_prime[timestep - 1].clone()
            q_prime_clamp = torch.clamp(q_prime_sub, min=self.cfg.params.attribute_minimums.discharge)
            velocity = _get_velocity(
                q_t=self._discharge_t,
                _n=self.n,
                _q_spatial=self.q_spatial,
                _s0=slope,
                _p_spatial=self.p_spatial,
                width=width,
                velocity_lb=self.velocity_lb,
                depth_lb=self.depth_lb,
            )
            k = torch.div(length, velocity)
            denom = (2.0 * k * (1.0 - x_storage)) + self.t
            c_2 = (self.t + (2.0 * k * x_storage)) / denom
            c_3 = ((2.0 * k * (1.0 - x_storage)) - self.t) / denom
            c_4 = (2.0 * self.t) / denom
            i_t = torch.matmul(self.adjacency_matrix, self._discharge_t)
            q_l = q_prime_clamp

            b_array = (c_2 * i_t) + (c_3 * self._discharge_t) + (c_4 * q_l)
            b = b_array.unsqueeze(-1)
            c_1 = (self.t - (2.0 * k * x_storage)) / denom
            c_1_ = c_1 * -1
            c_1_[0] = 1.0
            A_values = mapper.map(c_1_)
            A_csr = RiverNetworkMatrix.apply(A_values, mapper.crow_indices, mapper.col_indices)

            try:
                x = solve_triangular(A_csr, b, upper=False)
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError from e
            sol = x.squeeze()
            q_t1 = torch.clamp(sol, min=self.discharge_lb)
            
            for i, gage_idx in enumerate(gage_indices):
                output[i, timestep] = torch.sum(q_t1[gage_idx])

            self._discharge_t = q_t1

        output_dict = {
            "runoff": output,
        }

        return output_dict
