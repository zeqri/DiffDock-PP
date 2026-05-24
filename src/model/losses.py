import torch
import torch.nn as nn
import torch.nn.functional as F

from geom_utils import NoiseSchedule, score_norm


class DiffusionLoss(nn.Module):
    def __init__(self, args):
        super(DiffusionLoss, self).__init__()
        self.args = args
        self.tr_weight = args.tr_weight
        self.rot_weight = args.rot_weight
        self.latent_weight = args.latent_weight
       
        #bool sargs to include tr, rot and latent
        self.translation=args.translation
        self.rotation=args.rotation
        self.latent=args.latent
        
        self.noise_schedule = NoiseSchedule(args)
        self.eps = 1e-5

    def forward(self, data, outputs, apply_mean=True):
        # extract outputs
        tr_pred     = outputs["tr_pred"]
        rot_pred    = outputs["rot_pred"]
        latent_pred = outputs["latent_pred"]
        device = tr_pred.device

        # gather t values
        complex_t = []
        for noise_type in ["tr", "rot", "latent"]:
            if torch.cuda.is_available() and self.args.num_gpu == 1:
                cur_t = data.complex_t[noise_type]
            else:
                cur_t = torch.cat([d.complex_t[noise_type] for d in data])
            complex_t.append(cur_t)

        # convert to sigmas
        tr_s, rot_s, latent_s = self.noise_schedule(*complex_t)
        mean_dims = (0, 1) if apply_mean else 1

        # translation component
        tr_score = (
            torch.cat([d.tr_score for d in data], dim=0)
            if device.type == "cuda" and self.args.num_gpu > 1
            else data.tr_score.cpu()
        )
        tr_s = tr_s.unsqueeze(-1).cpu()
        tr_loss      = ((tr_pred.cpu() - tr_score) ** 2 * tr_s**2).mean(dim=mean_dims)
        tr_base_loss = (tr_score**2 * tr_s**2).mean(dim=mean_dims).detach()

        # rotation component
        rot_score = (
            torch.cat([d.rot_score for d in data], dim=0)
            if device.type == "cuda" and self.args.num_gpu > 1
            else data.rot_score.cpu()
        )
        rot_score_norm = score_norm(rot_s.cpu()).unsqueeze(-1)
        rot_loss      = (((rot_pred.cpu() - rot_score) / (rot_score_norm + self.eps)) ** 2).mean(dim=mean_dims)
        rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

        # latent component (Euclidean, matches tr_loss style)
        latent_score = (
            torch.cat([d.latent_score for d in data], dim=0)
            if device.type == "cuda" and self.args.num_gpu > 1
            else data.latent_score.cpu()
        )
        latent_s = latent_s.unsqueeze(-1).cpu()
        latent_loss      = ((latent_pred.cpu() - latent_score) ** 2 * latent_s**2).mean(dim=mean_dims)
        latent_base_loss = (latent_score**2 * latent_s**2).mean(dim=mean_dims).detach()


        if not self.translation and not self.rotation and self.latent: # latnet only
            loss =  latent_loss * self.latent_weight
            losses = {
            "loss": loss,
            "latent_loss": latent_loss,
            "latent_base_loss": latent_base_loss,
            }    

        elif self.translation and  self.rotation and not self.latent: #  #rigid body
            loss = tr_loss * self.tr_weight + rot_loss * self.rot_weight
            losses = {
            "loss": loss,
            "tr_loss": tr_loss,
            "rot_loss": rot_loss,
            "tr_base_loss": tr_base_loss,
            "rot_base_loss": rot_base_loss,
        }
            
        elif self.translation and  self.rotation and self.latent:
            loss = tr_loss * self.tr_weight + rot_loss * self.rot_weight+ latent_loss * self.latent_weight
            losses = {
            "loss": loss,
            "tr_loss": tr_loss,
            "rot_loss": rot_loss,
            "latent_loss": latent_loss,
            "tr_base_loss": tr_base_loss,
            "rot_base_loss": rot_base_loss,
            "latent_base_loss": latent_base_loss,
           }
                     
        return losses


        

        