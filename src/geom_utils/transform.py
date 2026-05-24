"""
This file has been heavily modified, mostly in terms
of syntax and nicer flow.

2022.11.08
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform

from .so3 import sample_vec, score_vec
from .geometry import kabsch_torch, axis_angle_to_matrix

# ------ PyG UTILS -------

class NoiseTransform(BaseTransform):
    """
        Apply translation, rotation, latent noise
    """
    def __init__(self, args):
        self.noise_schedule = NoiseSchedule(args)
        self.all_atom = (args.resolution == "atom")
        self.translation=args.translation
        self.rotation=args.rotation
        self.latent=args.latent

    def __call__(self, data):
        t = np.random.uniform()
        t_tr, t_rot, t_latent = t, t, t
        data = self.apply_noise(data, t_tr, t_rot, t_latent)
        return data

    def forward(self, data):
        return self.__call__(data)

    def apply_noise(self, data, t_tr, t_rot, t_latent,
                    tr_update=None,
                    rot_update=None,
                    latent_update=None):
        tr_s, rot_s, latent_s = self.noise_schedule(t_tr, t_rot, t_latent)
        set_time(data, t_tr, t_rot, t_latent, 1, device=None)

        # sample updates if not provided
        if tr_update is None:
            tr_update = torch.normal(mean=0, std=tr_s, size=(1, 3))
        if rot_update is None:
            rot_update = sample_vec(eps=rot_s)
            rot_update = torch.from_numpy(rot_update).float()
        if latent_update is None:
            latent_update = torch.normal(mean=0, std=latent_s, size=(1, 8))

        self.apply_updates(data, tr_update, rot_update,latent_update)
        self.get_score(data, tr_update, tr_s, rot_update, rot_s,
                       latent_update, latent_s)

        return data

    def apply_updates(self, data, tr_update, rot_update,latent_update):

        
        if self.translation and self.rotation: 
            com = torch.mean(data["ligand"].pos, dim=0, keepdim=True)
            rot_mat = axis_angle_to_matrix(rot_update.squeeze())
            rigid_new_pos = (
            (data["ligand"].pos - com) @ rot_mat.T + tr_update + com
            )
            data["ligand"].pos = rigid_new_pos

        if latent_update is not None and self.latent:
            if hasattr(data["ligand"], "z"):
                data["ligand"].z = data["ligand"].z + latent_update
            else:
                data["ligand"].z = latent_update

        return data

    def get_score(self, data,
                  tr_update, tr_s,
                  rot_update, rot_s,
                  latent_update, latent_s):
        # translation score
        data.tr_score = -tr_update / tr_s**2
        # rotation score
        rot_score = score_vec(vec=rot_update, eps=rot_s)
        data.rot_score = rot_score.unsqueeze(0)
        # latent score (Euclidean, matches tr style)
        data.latent_score = -latent_update / latent_s**2
        return data


class NoiseSchedule:
    """
        Transforms t into scaled sigmas
    """
    def __init__(self, args):
        self.tr_s_min      = args.tr_s_min
        self.tr_s_max      = args.tr_s_max
        self.rot_s_min     = args.rot_s_min
        self.rot_s_max     = args.rot_s_max        
        self.latent_s_min  = args.latent_s_min
        self.latent_s_max  = args.latent_s_max

    def __call__(self, t_tr, t_rot, t_latent):
        tr_s     = self.tr_s_min     ** (1 - t_tr)     * self.tr_s_max     ** t_tr
        rot_s    = self.rot_s_min    ** (1 - t_rot)    * self.rot_s_max    ** t_rot
        latent_s = self.latent_s_min ** (1 - t_latent) * self.latent_s_max ** t_latent
        return tr_s, rot_s, latent_s


def set_time(complex_graphs, t_tr, t_rot, t_latent,
             batch_size: int, device=None):
    lig_size = complex_graphs["ligand"].num_nodes
    complex_graphs["ligand"].node_t = {
        "tr":     t_tr     * torch.ones(lig_size).to(device),
        "rot":    t_rot    * torch.ones(lig_size).to(device),
        "latent": t_latent * torch.ones(lig_size).to(device),
    }
    rec_size = complex_graphs["receptor"].num_nodes
    complex_graphs["receptor"].node_t = {
        "tr":     t_tr     * torch.ones(rec_size).to(device),
        "rot":    t_rot    * torch.ones(rec_size).to(device),
        "latent": t_latent * torch.ones(rec_size).to(device),
    }
    complex_graphs.complex_t = {
        "tr":     t_tr     * torch.ones(batch_size).to(device),
        "rot":    t_rot    * torch.ones(batch_size).to(device),
        "latent": t_latent * torch.ones(batch_size).to(device),
    }