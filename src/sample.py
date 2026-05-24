

"""
    Inference script including latent
"""

import os
import gc
import sys
import copy

import numpy as np
import torch
from torch_geometric.loader import DataLoader, DataListLoader
from scipy.spatial.transform import Rotation as R

from utils import printt
from geom_utils import set_time, NoiseTransform

def sample(data_list, model, args, epoch=0, visualize_first_n_samples=0,
           visualization_dir="./visualization", in_batch_size=None):
    """
        Run reverse process
    """
    if in_batch_size is None:
        in_batch_size = args.batch_size
    # switch to eval mode
    model.eval()

    # stores various noise-related utils
    transform = NoiseTransform(args)

    # diffusion timesteps
    timesteps = get_timesteps(args.num_steps)

    # Prepare for visualizations
    visualize_first_n_samples = min(visualize_first_n_samples, len(data_list))
    graph_gts = [data_list[i] for i in range(visualize_first_n_samples)]
    visualization_values = [data_list.get_visualization_values(index=i) for i in range(visualize_first_n_samples)]
    four_letter_pdb_names = [get_four_letters_pdb_identifier(graph_gt.name) for graph_gt in graph_gts]
    visualization_dirs = create_visualization_directories(visualization_dir, epoch, four_letter_pdb_names)

    # randomize original position and COPY data_list
    data_list = randomize_position(data_list, args)

    # For visualization
    for i in range(visualize_first_n_samples):
        write_pdb(visualization_values[i], graph_gts[i], "receptor",
              f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-receptor.pdb")
        write_pdb(visualization_values[i], graph_gts[i], "ligand",
              f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-gt.pdb")
        write_pdb(visualization_values[i], data_list[i], "ligand",
              f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-0.pdb")

    # sample
    for t_idx in range(args.num_steps):
        # create new loader with current step graphs
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        test_loader = loader(data_list, batch_size=args.batch_size)
        new_data_list = []  # updated every step

        # DiffDock uses same schedule for all noise
        cur_t = timesteps[t_idx]
        if t_idx == args.num_steps - 1:
            dt = cur_t
        else:
            dt = cur_t - timesteps[t_idx + 1]

        for com_idx, complex_graphs in enumerate(test_loader):
            # move to CUDA
            if torch.cuda.is_available() and args.num_gpu == 1:
                complex_graphs = complex_graphs.cuda(args.gpu)

            # this MAY differ from args.batch_size
            # based on # GPUs and last batch
            if type(complex_graphs) is list:
                batch_size = len(complex_graphs)
            else:
                batch_size = complex_graphs.num_graphs

            # convert to sigma space and save time
            tr_s, rot_s, latent_s = transform.noise_schedule(cur_t, cur_t, cur_t)

            device_for_set_time = complex_graphs["ligand"]["pos"].device if torch.cuda.is_available() and args.num_gpu == 1 else None
            if type(complex_graphs) is list:
                for g in complex_graphs:
                    set_time(g, cur_t, cur_t, cur_t, 1, device_for_set_time)
            else:
                set_time(complex_graphs, cur_t, cur_t, cur_t, batch_size, device_for_set_time)

            with torch.no_grad():
                outputs = model(complex_graphs)

            tr_score     = outputs["tr_pred"].cpu()
            rot_score    = outputs["rot_pred"].cpu()
            latent_score = outputs["latent_pred"].cpu()

            # translation gradient
            tr_scale = torch.sqrt(
                2 * torch.log(torch.tensor(args.tr_s_max / args.tr_s_min)))
            tr_g = tr_s * tr_scale

            # rotation gradient
            rot_scale = torch.sqrt(
                torch.log(torch.tensor(args.rot_s_max / args.rot_s_min)))
            rot_g = 2 * rot_s * rot_scale

            # latent gradient (mirrors tr)
            # latent_scale = torch.sqrt(
            #     2 * torch.log(torch.tensor(args.latent_s_max / args.latent_s_min)))

            latent_scale = torch.sqrt(
                2 * torch.log(torch.tensor(args.tr_s_max / args.tr_s_min)))    
            latent_g = latent_s * latent_scale

            # actual update
            if args.ode:
                tr_update     = (0.5 * tr_g**2     * dt * tr_score)
                rot_update    = (0.5 * rot_score    * dt * rot_g**2)
                latent_update = (0.5 * latent_g**2 * dt * latent_score)
            else:
                if args.no_final_noise and t_idx == args.num_steps - 1:
                    tr_z     = torch.zeros((batch_size, 3))
                    rot_z    = torch.zeros((batch_size, 3))
                    latent_z = torch.zeros((batch_size, 8))
                elif args.no_random:
                    tr_z     = torch.zeros((batch_size, 3))
                    rot_z    = torch.zeros((batch_size, 3))
                    latent_z = torch.zeros((batch_size, 8))
                else:
                    tr_z     = torch.normal(0, 1, size=(batch_size, 3))
                    rot_z    = torch.normal(0, 1, size=(batch_size, 3))
                    latent_z = torch.normal(0, 1, size=(batch_size, 8))

                tr_update     = (tr_g**2     * dt * tr_score)     + (tr_g     * np.sqrt(dt) * tr_z)
                rot_update    = (rot_score   * dt * rot_g**2)     + (rot_g    * np.sqrt(dt) * rot_z)
                latent_update = (latent_g**2 * dt * latent_score) + (latent_g * np.sqrt(dt) * latent_z)

            if args.temp_sampling != 1.0:
                tr_sigma_data = np.exp(
                    args.temp_sigma_data_tr * np.log(args.tr_s_max) +
                    (1 - args.temp_sigma_data_tr) * np.log(args.tr_s_min))
                lambda_tr = (tr_sigma_data + tr_s) / (tr_sigma_data + tr_s / args.temp_sampling)
                tr_update = (
                    tr_g**2 * dt * (lambda_tr + args.temp_sampling * args.temp_psi / 2) * tr_score.cpu() +
                    tr_g * np.sqrt(dt * (1 + args.temp_psi)) * tr_z
                ).cpu()

                rot_sigma_data = np.exp(
                    args.temp_sigma_data_rot * np.log(args.rot_s_max) +
                    (1 - args.temp_sigma_data_rot) * np.log(args.rot_s_min))
                lambda_rot = (rot_sigma_data + rot_s) / (rot_sigma_data + rot_s / args.temp_sampling)
                rot_update = (
                    rot_g**2 * dt * (lambda_rot + args.temp_sampling * args.temp_psi / 2) * rot_score.cpu() +
                    rot_g * np.sqrt(dt * (1 + args.temp_psi)) * rot_z
                ).cpu()

                latent_sigma_data = np.exp(
                    args.temp_sigma_data_latent * np.log(args.latent_s_max) +
                    (1 - args.temp_sigma_data_latent) * np.log(args.latent_s_min))
                lambda_latent = (latent_sigma_data + latent_s) / (latent_sigma_data + latent_s / args.temp_sampling)
                latent_update = (
                    latent_g**2 * dt * (lambda_latent + args.temp_sampling * args.temp_psi / 2) * latent_score.cpu() +
                    latent_g * np.sqrt(dt * (1 + args.temp_psi)) * latent_z
                ).cpu()

            # apply transformations
            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to("cpu").to_data_list()
            for i, data in enumerate(complex_graphs):
                new_graph = transform.apply_updates(
                    data,
                    tr_update[i:i+1],
                    rot_update[i:i+1].squeeze(0),
                    latent_update[i:i+1],
                )
                new_data_list.append(new_graph)
            # === end of batch ===

        for i in range(visualize_first_n_samples):
            write_pdb(visualization_values[i], new_data_list[i], "ligand",
                      f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-{t_idx + 1}.pdb")

        # update starting point for next step
        assert len(new_data_list) == len(data_list)
        data_list = new_data_list
        printt(f"Completed {t_idx} out of {args.num_steps} steps")

        # Cut last diffusion steps short because they tend to overfit
        if t_idx >= args.actual_steps - 1:
            break
        # === end of timestep ===

    return data_list


def create_visualization_directories(top_visualization_dir, epoch, pdb_names):
    visualization_dirs = [f"{top_visualization_dir}/epoch-{epoch}/{pdb_name}" for pdb_name in pdb_names]
    for directory in visualization_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return visualization_dirs

def get_four_letters_pdb_identifier(pdb_name):
    return pdb_name.split('/')[-1].split('.')[0]

def write_pdb(item, graph, part, path):
    lines = to_pdb_lines(item, graph, part)
    with open(path, "w") as file:
        file.writelines(lines)


def to_pdb_lines(visualization_values, graph, part):
    assert part in ("ligand", "receptor", "both"), "Part should be ligand or receptor"
    parts = ["ligand", "receptor"] if part == "both" else [part]

    lines = []
    for part in parts:
        this_vis_values = visualization_values[part]
        this_vis_values = {k: v.strip() if type(v) is str else v for k, v in this_vis_values.items()}
        for i, resname in enumerate(this_vis_values["resname"]):
            xyz = graph[part].pos[i]

            chain_raw = this_vis_values["chain"][i]
            chain_id = str(chain_raw).replace("<Chain id=", "").replace(">", "").strip()
            chain_id = chain_id[0] if chain_id else " "

            atom_name = this_vis_values["atom_name"][i].strip()
            resname = resname.strip()
            resseq_raw = str(this_vis_values["residue"][i]).strip()
            resseq = int(resseq_raw) if resseq_raw else i + 1
            element = this_vis_values["element"][i].strip()

            line = (
                f"ATOM  {i + 1:>5} {atom_name:<4} {resname:<3} {chain_id}{resseq:>4}    "
                f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
                f"  1.00  0.00          {element:>2}\n"
            )
            lines.append(line)

    return lines

def get_timesteps(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def randomize_position(data_list, args):
    """
        Modify COPY of data_list objects
    """
    data_list = copy.deepcopy(data_list)

    for i, complex_graph in enumerate(data_list):
        pos = complex_graph["ligand"].pos
        center = torch.mean(pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix())
        pos = (pos - center) @ random_rotation.T.float()

        # random translation
        tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
        pos = pos + tr_update
        complex_graph["ligand"].pos = pos

        # initialize latent z with maximum noise (mirrors tr initialization)
        complex_graph["ligand"].z = torch.normal(0, args.tr_s_max, size=(1, 8))

        data_list.set_graph(i, complex_graph)

    return data_list