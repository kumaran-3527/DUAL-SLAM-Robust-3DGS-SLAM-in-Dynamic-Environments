import numpy as np
import torch
import lietorch
import droid_backends
import src.geom.ba
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F

from src.modules.droid_net import cvx_upsample
import src.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from src.utils.dyn_uncertainty import mapping_utils as map_utils
from src.utils.dyn_uncertainty.mapping_utils import compute_dino_regularization_loss


class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer, uncer_network=None):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.metric_depth_reg = cfg['tracking']['backend']['metric_depth_reg']
        if not self.metric_depth_reg:
            self.printer.print(f"Metric depth for regularization is not activated.",FontColor.INFO)
            self.printer.print(f"This should not happen for WildGS-SLAM unless you are doing ablation study",FontColor.INFO)
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8
        self.slice_h = slice(self.down_scale // 2 - 1, ht//self.down_scale*self.down_scale+1, self.down_scale)
        self.slice_w = slice(self.down_scale // 2 - 1, wd//self.down_scale*self.down_scale+1, self.down_scale)
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        # To save gpu ram, we put images to cpu as it is never used
        self.images = torch.zeros(buffer, 3, ht, wd, device='cpu', dtype=torch.float32)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_mask_up = torch.ones(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.printer = printer
        
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.uncer_network = uncer_network
        if self.uncertainty_aware:
            n_features = self.cfg["mapping"]["uncertainty_params"]['feature_dim']
            
            # This check is to ensure the size of self.dino_feats
            if self.cfg["mono_prior"]["feature_extractor"] not in ["dinov2_reg_small_fine", "dinov2_small_fine","dinov2_vits14", "dinov2_vits14_reg"]:
                raise ValueError("You are using a new feature extractor, make sure the downsample factor is 14")
            
            # The followings are in cpu to save memory
            self.dino_feats = torch.zeros(buffer, ht//14, wd//14, n_features, device='cpu', dtype=torch.float).share_memory_()
            self.dino_feats_resize = torch.zeros(buffer, n_features, ht//self.down_scale, wd//self.down_scale, device='cpu', dtype=torch.float).share_memory_()
            self.uncertainties_inv = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            self.mapping_uncertainties_inv = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            
            
            # DINOv2 warping Semantic Consistency Error (SCE) per keyframe at 1/8 resolution.
            # Aggregated across all BA edges touching each frame during ba().
            self.dino_warp_scores = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            # Experience Replay: geometric pseudo-labels from tracker (sentinel: -1.0 = not yet computed)
            self.geom_uncertainty_labels = -1.0 * torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            # SCE scheduling: run on new-keyframe events and every SCE_STRIDE BA calls.
            self._sce_ba_call_count = 0
            self._sce_new_frame_pending = False
            self.SCE_STRIDE = 2           # recompute every N BA calls (tune: 2-8)
            self.SCE_MIN_GAP = 0     # skip stereo pairs and adjacent frames
            self.SCE_MAX_GAP = 4   # since this counts keyframes not frames larger values can lead to huge viewpoint changes
            
            self.use_sce_init_gate = cfg['tracking']["uncertainty_params"].get('use_sce_init_gate', False)
            self.cached_sce_f = None
            self.cached_ii_filtered = None
            self.cached_jj_filtered = None

            from src.utils.dyn_uncertainty.uncertainty_model import LoRATrackerNet
            lora_rank = cfg['tracking']['uncertainty_params'].get('lora_rank',8)
            lora_lr = cfg['tracking']['uncertainty_params'].get('lora_lr',4e-7) #0.0000001 or 4e-7, 4e-8
            # With lambda_prior = 0.0, we need a moderate AdamW weight decay (1.0) 
            # to act as a gentle parameter-space "forgetting" mechanism, but we keep it small (0.05)
            # because the inactive edges provide plenty of natural stability.
            lora_wd = cfg['tracking']['uncertainty_params'].get('lora_wd', 0.1)
            self.tracker_net = LoRATrackerNet(base_net=self.uncer_network.net_fast, rank=lora_rank).to(self.device)
            # Only optimize the LoRA parameters (A and B matrices)
            lora_params = [p for n, p in self.tracker_net.named_parameters() if 'lora_' in n]
            self.tracker_optimizer = torch.optim.AdamW(
                lora_params, lr=lora_lr, weight_decay=lora_wd
            )
          

            # --- NEW SINGLE-LAYER NETWORK ---
            # import torch.nn as nn
            # self.tracker_net = nn.Sequential(
            #     nn.Linear(384, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 1),
            #     nn.Softplus()
            # ).to(self.device)
            
            # lora_lr = cfg['tracking']['uncertainty_params'].get('lora_lr', 1e-3)
            # lora_wd = cfg['tracking']['uncertainty_params'].get('lora_wd', 1e-2)
            # self.tracker_optimizer = torch.optim.Adam(
            #     self.tracker_net.parameters(), lr=lora_lr, weight_decay=lora_wd
            # )
            # # --------------------------------
            self._tracker_net_trained = False
            self.is_initialized = False
            self.is_initialized_fully = False

        else:
            self.dino_feats = None
            self.dino_feats_resize = None

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1].cpu()

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            mono_depth = item[4][self.slice_h,self.slice_w]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            self.mono_disps_up[index] = torch.where(item[4]>0, 1.0/item[4], 0)
            # self.disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.dino_feats[index] = item[9].cpu()

            if len(item[9].shape) == 3:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(2,0,1).unsqueeze(0),
                                                            self.disps.shape[-2:], 
                                                            mode='bilinear').squeeze().cpu()
            else:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(0,3,1,2),
                                                            self.disps.shape[-2:], 
                                                            mode='bilinear').cpu()

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def project_images_with_mask(self, images, pixel_positions, masks=None):
        """ 
            Project images/depths from the input pixel positions using bilinear interpolation.
            This function will automatically return the mask where the given pixel positions are out of the images
        Args:
            images (torch.Tensor): A tensor of shape [B, C, H, W] representing the images/depths.
            pixel_positions (torch.Tensor): A tensor of shape [B, H, W, 2] containing float 
                                            pixel positions for interpolation. Note that [:,:,:,0]
                                            is width and [:,:,:,1] is height.
            masks (torch.Tensor, optional): A boolean tensor of shape [B, H, W]. If provided, 
                                            specifies valid pixels. Default is None, which 
                                            results in all pixels being valid at the begining.
        
        Returns:
            torch.Tensor: A tensor of shape [B, C, H, W] containing the projected images/depths, 
                        where invalid pixels are set to 0.
            torch.Tensor: The combined mask that filters out invalid positions and applies
                      the original mask.
        """
        B, C, H, W = images.shape
        device = images.device

        # If masks are not provided, create a mask of all ones (True) with the same shape as the images
        if masks is None:
            masks = torch.ones(B, H, W, dtype=torch.bool, device=device)
        
        # Normalize pixel positions to range [-1, 1]
        grid = pixel_positions.clone()
        grid[..., 0] = 2.0 * (grid[..., 0] / (W - 1)) - 1.0
        grid[..., 1] = 2.0 * (grid[..., 1] / (H - 1)) - 1.0

        projected_image = F.grid_sample(images, grid, mode='bilinear', align_corners=True)

        # Mask out invalid positions where x or y are out of bounds and combine it with the initial mask
        valid_mask = (pixel_positions[..., 0] >= 0) & (pixel_positions[..., 0] < W) & \
                    (pixel_positions[..., 1] >= 0) & (pixel_positions[..., 1] < H)
        valid_mask &= masks

        # Apply the combined mask: set to 0 where combined mask is False
        projected_image = projected_image.permute(0, 2, 3, 1)  # conver to [B, H, W, C]
        projected_image = projected_image * valid_mask.unsqueeze(-1)
        
        return projected_image.permute(0, 3, 1, 2), valid_mask  # Return to [B, C, H, W]

    @torch.no_grad()
    def filter_high_err_mono_depth(self, idx, ii, jj):
        nb_frame = self.cfg['tracking']['nb_ref_frame_metric_depth_filtering']

        jj = jj[ii==idx]
        for j in torch.arange(idx-1, max(0,idx-nb_frame)-1, -1):
            if jj.shape[0] >= nb_frame:
                break
            if j not in jj:
                torch.cat((jj, j.unsqueeze(0).to(jj.device)))

        ii = torch.tensor(idx).repeat(jj.shape[0])

        # all frames share the same intrinsics
        X0, _ = pops.iproj(self.mono_disps_up[jj].unsqueeze(0), 
                      self.intrinsics[0].unsqueeze(0).repeat(1,jj.shape[0],1)*self.down_scale, 
                      jacobian=False)
        Gs = lietorch.SE3(self.poses[None])
        Gji = Gs[:,ii] * Gs[:,jj].inv()
        X1, _ = pops.actp(Gji, X0, jacobian=False)
        x1, _ = pops.proj(X1, self.intrinsics[0].unsqueeze(0).repeat(1,jj.shape[0],1)*self.down_scale, jacobian=False, return_depth=True)

        i_disp = self.mono_disps_up[idx]
        accurate_count = torch.zeros_like(i_disp)
        inaccurate_count = torch.zeros_like(i_disp)

        x1_rounded = torch.round(x1[..., :2]).long()
        # x1 is the 3d poisition (x,y,z)
        # projected point is valid only if its inside the image range and the depth is greater than 0
        valid_mask = (x1_rounded[..., 1] >= 0) & (x1_rounded[..., 1] < x1.shape[2]) & \
                    (x1_rounded[..., 0] >= 0) & (x1_rounded[..., 0] < x1.shape[3]) & (x1[...,2]>0)
        
        i_dino = F.interpolate(self.dino_feats[idx].permute(2,0,1).unsqueeze(0),
                                self.disps_up.shape[-2:], 
                                mode='bilinear').to(self.device).squeeze()
        for j_id in range(jj.shape[0]):
            projected_j_to_i = x1[0, j_id]
            x_coords, y_coords = x1_rounded[0, j_id, ..., 0], x1_rounded[0, j_id, ..., 1]
            
            # Select valid coordinates and their Dino features
            j_dino = F.interpolate(self.dino_feats[jj[j_id]].permute(2,0,1).unsqueeze(0),
                                    self.disps_up.shape[-2:], 
                                    mode='bilinear').to(self.device).squeeze()
            valid_x, valid_y = x_coords[valid_mask[0, j_id]], y_coords[valid_mask[0, j_id]]
            src_y, src_x = torch.where(valid_mask[0, j_id])

            # Compute cosine similarity for each valid position in chunks to avoid CUDA OOM
            num_valid = valid_x.shape[0]
            similarity = torch.zeros(num_valid, device=self.device, dtype=torch.float32)
            pixel_chunk_size = 50000
            for chunk_start in range(0, num_valid, pixel_chunk_size):
                chunk_end = min(num_valid, chunk_start + pixel_chunk_size)
                
                j_dino_chunk = j_dino[:, src_y[chunk_start:chunk_end], src_x[chunk_start:chunk_end]]
                i_dino_chunk = i_dino[:, valid_y[chunk_start:chunk_end], valid_x[chunk_start:chunk_end]]
                
                sim_chunk = F.normalize(j_dino_chunk, p=2, dim=0).mul(F.normalize(i_dino_chunk, p=2, dim=0)).sum(dim=0)
                similarity[chunk_start:chunk_end] = sim_chunk

            matching_mask = similarity > 0.9

            # Update projected disparity and counts based on the similarity check
            j_projected_disp = torch.zeros_like(self.mono_disps_up[idx])
            matched_disp = projected_j_to_i[valid_mask[0, j_id]][matching_mask]
            matched_x, matched_y = valid_x[matching_mask], valid_y[matching_mask]
            j_projected_disp[matched_y, matched_x] = matched_disp[..., 2]

            # Error calculation and count updates
            error = torch.abs(1 / j_projected_disp[matched_y, matched_x] - 1 / i_disp[matched_y, matched_x]) * j_projected_disp[matched_y, matched_x]
            correct_mask = error < 0.02

            # Batch update correct and bad counts
            accurate_count[matched_y[correct_mask], matched_x[correct_mask]] += 1
            inaccurate_count[matched_y[~correct_mask], matched_x[~correct_mask]] += 1

        # Clean the gpu memory
        torch.cuda.empty_cache()

        self.mono_disps_mask_up[idx][(accurate_count<=1)&(inaccurate_count>0)&(self.mono_disps_up[idx]>0)] = False
        

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1,
           motion_only=False, global_ba=False):

        if self.uncertainty_aware:
            if not self._tracker_net_trained or motion_only:
                # Use Mapper uncertainty during warmup, OR when filling non-keyframes 
                # (motion_only=True) since the Tracker's LoRA state is ephemeral and local.
                u_inv_i = self.mapping_uncertainties_inv[ii][None, :, :, :, None]
                weight *= u_inv_i
            else:
                u_inv_i = self.uncertainties_inv[ii][None, :, :, :, None] 
                weight *=  u_inv_i

            # Init-phase edge-level SCE gate (Optional)
            if self.use_sce_init_gate and self.cached_sce_f is not None and not self.is_initialized_fully:
                H_s = self.ht // self.down_scale
                W_s = self.wd // self.down_scale
                E_ba = len(ii)

                match = (
                    (ii[:, None] == self.cached_ii_filtered[None, :]) &
                    (jj[:, None] == self.cached_jj_filtered[None, :])
                )
                has_match = match.any(dim=1)
                edge_sce = torch.zeros(E_ba, H_s, W_s, device=self.device)
                if has_match.any():
                    match_idx = match.float().argmax(dim=1)
                    edge_sce[has_match] = self.cached_sce_f[match_idx[has_match]]

                # Conservative SCE gate: down-weight max by 50% instead of dropping completely
                sce_gate = (1.0 - 0.5 * edge_sce)[None, ..., None]
                weight *= sce_gate

            elif self.cached_sce_f is not None and self.is_initialized_fully:
                self.cached_sce_f = None
                self.cached_ii_filtered = None
                self.cached_jj_filtered = None


        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()

            if not self.metric_depth_reg:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False)
            else:
                mono_valid_mask = self.mono_disps_mask_up[:,self.slice_h,self.slice_w].clone().to(self.device)
                
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.mono_disps*mono_valid_mask,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False)
            
            self.disps.clamp_(min=1e-5)

            # ── Semantic Consistency Error (SCE) via DINOv2 Warping ──
            # After BA updates poses/disps, we warp the semantic DINOv2 features
            # to check for rigid-semantic inconsistency. Moving objects violate 
            # the static world assumption, causing high SCE at object boundaries.
            #
            # Scheduling: since motion_only is never actually True in the call chain
            # (frontend and backend always pass motion_only=False), we gate on:
            #   (a) a new keyframe was just confirmed (_sce_new_frame_pending=True), OR
            #   (b) every SCE_STRIDE ba calls elapsed.
            if self.uncertainty_aware:

                _run_sce = not global_ba
                if _run_sce:
                    self._sce_ba_call_count = 0
                    self._sce_new_frame_pending = False
                    with torch.no_grad():
                        # We consider ALL edges (both active and inactive) passed to ba()
                        # as long as they fall within the valid initialized frame range [0, t1)
                        window_mask = (ii >= 0) & (jj >= 0) & (ii < t1) & (jj < t1)
                        ii_active = ii[window_mask]
                        jj_active = jj[window_mask]
                        
                        if len(ii_active) > 0:
                            gap = (ii_active - jj_active).abs()
                            # temporal_gap_mask = (gap >= self.SCE_MIN_GAP)

                            ii_filtered = ii_active
                            jj_filtered = jj_active
                        else:
                            ii_filtered = torch.empty(0, dtype=torch.long, device=self.device)
                            jj_filtered = torch.empty(0, dtype=torch.long, device=self.device)


                        if len(ii_filtered) == 0 or self.dino_feats_resize is None:
                            pass
                        else:
                            W_s = self.wd // self.down_scale
                            H_s = self.ht // self.down_scale

                            # ── Condition 2: Geometric Validity ──
                            # Re-project source pixels into 3D and warp into target frame.
                            # We need the 3D point in the target frame (X1) to check
                            # the MIN_DEPTH condition.  We replicate the projective ops
                            # inline so we can access the intermediate Z value.
                            MIN_DEPTH_SCE = 0.25

                            # Lift source pixels to 3D: X0 = [X, Y, 1, disp] in source cam
                            X0, _ = pops.iproj(
                                self.disps[None, ii_filtered],
                                self.intrinsics[None, ii_filtered],
                            )  # [1, E_f, H_s, W_s, 4]

                            # Relative pose: T_jj * T_ii^{-1}
                            poses_se3 = lietorch.SE3(self.poses[None])
                            Gij = poses_se3[:, jj_filtered] * poses_se3[:, ii_filtered].inv()
                            # Perturb self-edges to prevent zero-division
                            Gij.data[:, ii_filtered == jj_filtered] = torch.as_tensor(
                                [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=self.device
                            )

                            # Apply rigid transform: X1 = Gij * X0  [1, E_f, H_s, W_s, 4]
                            X1, _ = pops.actp(Gij, X0)

                            # Project X1 into target image: x1  [1, E_f, H_s, W_s, 2]
                            x1, _ = pops.proj(X1, self.intrinsics[None, jj_filtered])

                            X1 = X1.squeeze(0)  # [E_f, H_s, W_s, 4]
                            x1 = x1.squeeze(0)  # [E_f, H_s, W_s, 2]

                            # Z-depth in the target camera frame
                            Z_target = X1[..., 2]  # [E_f, H_s, W_s]

                            # Pixel-level geometric validity: depth floor + image boundary
                            u = x1[..., 0]  # x-coord in target image
                            v = x1[..., 1]  # y-coord in target image
                            geom_valid = (
                                (Z_target >= MIN_DEPTH_SCE) &
                                (u >= 0.0) & (u <= W_s - 1.0) &
                                (v >= 0.0) & (v <= H_s - 1.0)
                            )  # [E_f, H_s, W_s]

                            # Build normalized grid for grid_sample from the projected coords
                            # (reuse x1 which is already in pixel-space)
                            grid = x1.clone()  # [E_f, H_s, W_s, 2]
                            grid[..., 0] = 2.0 * grid[..., 0] / (W_s - 1) - 1.0
                            grid[..., 1] = 2.0 * grid[..., 1] / (H_s - 1) - 1.0

                            # ── Condition 3: Cosine Similarity with Clamping & Threshold ──
                            E_f = len(ii_filtered)
                            chunk_size = 16
                            sce_f = torch.zeros(E_f, H_s, W_s, device=self.device, dtype=torch.float32)

                            for chunk_start in range(0, E_f, chunk_size):
                                chunk_end = min(E_f, chunk_start + chunk_size)

                                jj_chunk = jj_filtered[chunk_start:chunk_end].cpu()
                                ii_chunk = ii_filtered[chunk_start:chunk_end].cpu()

                                feats_jj_chunk = self.dino_feats_resize[jj_chunk].to(self.device)
                                feats_ii_chunk = self.dino_feats_resize[ii_chunk].to(self.device)

                                # L2-normalize along the channel dim for true cosine similarity
                                feats_jj_chunk = F.normalize(feats_jj_chunk, p=2, dim=1)
                                feats_ii_chunk = F.normalize(feats_ii_chunk, p=2, dim=1)

                                grid_chunk = grid[chunk_start:chunk_end]

                                # Bilinear interpolation of target features at projected coords
                                warped_chunk = F.grid_sample(
                                    feats_jj_chunk, grid_chunk,
                                    mode='bilinear', padding_mode='zeros', align_corners=True
                                )

                                # Cosine similarity = dot product (already L2-normalized)
                                cos_sim = (feats_ii_chunk * warped_chunk).sum(dim=1)  # [chunk, H_s, W_s]

                                # Clamp to [0, 1] to reject outlier negative similarities
                                cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)

                               # Hard threshold: zero out low-confidence matches (<= 0.5)
                                cos_sim = torch.where(
                                    cos_sim >= 0.50, cos_sim, torch.zeros_like(cos_sim)
                                )
                        

                                # SCE = 1 - cosine_similarity (high SCE = semantically inconsistent)
                                sce_f[chunk_start:chunk_end] = 1.0 - cos_sim

                            # Zero out pixels that failed geometric validity
                            sce_f[~geom_valid] = 0.0

                            # ── Fully Vectorized Aggregation (CUDA) ──────────────────────
                            # We replace all Python dictionary groupings and for-loops with 
                            # PyTorch scatter operations. This avoids launching hundreds of 
                            # tiny CUDA kernels and pushes the entire aggregation to the GPU.
                            
                            N = self.counter.value
                            delta = torch.clamp(torch.abs(ii_filtered - jj_filtered), min=1)
                            edge_weights = (1.0 / delta.float()).view(-1, 1, 1)  # [E, 1, 1]
                            
                            # Valid weights and masked SCE
                            valid_mask = geom_valid.float()  # [E, H_s, W_s]
                            weighted_sce = sce_f * edge_weights * valid_mask
                            weight_contrib = edge_weights * valid_mask

                            # Scatter add aggregation
                            out_sce = torch.zeros(N, H_s, W_s, device=self.device)
                            out_weight = torch.zeros(N, H_s, W_s, device=self.device)
                            
                            idx = ii_filtered.view(-1, 1, 1).expand(-1, H_s, W_s)
                            out_sce.scatter_add_(0, idx, weighted_sce)
                            out_weight.scatter_add_(0, idx, weight_contrib)
                            
                            frame_sce = out_sce / out_weight.clamp(min=1e-6)

                            clamped_sce = torch.clamp(frame_sce, 0.0, 1.0)

                            # ONLY update frames that were actively evaluated in this BA window
                            # This preserves historical scores for frames that weren't touched
                            active_srcs = torch.unique(ii_filtered)
                            if len(active_srcs) > 0:
                                self.dino_warp_scores[active_srcs] = clamped_sce[active_srcs]

                            if self.use_sce_init_gate and not self.is_initialized_fully:
                                self.cached_sce_f = sce_f.detach().clone()
                                self.cached_ii_filtered = ii_filtered.clone()
                                self.cached_jj_filtered = jj_filtered.clone()

                            with torch.enable_grad():
                                # ONLY train the tracker after the graph has stabilized (warmup is complete)
                                if self.is_initialized_fully and hasattr(self, 'tracker_net') and len(ii_filtered) > 0:
                                    self.tracker_net.train()
                                    unique_srcs = torch.unique(ii_filtered)
                                    total_loss = 0.0
                                    total_reg_loss = 0.0
                                    n_frames = 0
                                    for src_idx in unique_srcs:
                                        src = src_idx.item()
                                        edge_mask = (ii_filtered == src_idx)
                                        feats = self.dino_feats_resize[src].to(self.device)
                                        feats_hw = feats.permute(1, 2, 0)
                                        
        
                                        with torch.no_grad():
                                            u_map = self.uncer_network.net_fast(feats_hw).view(H_s, W_s)
                                        
                                        # 2. Get final combined prediction from Tracker (which internally adds LoRA to base)
                                        u_final = self.tracker_net(feats_hw).view(H_s, W_s)
                                        
                                        # 4. Filter for valid edges
                                        sce_src = sce_f[edge_mask]
                                        valid_src = geom_valid[edge_mask]
                                        
                                        # Broadcast uncertainty maps along the edge dimension [1, H_s, W_s]
                                        u_final_b = u_final.unsqueeze(0)
                                        u_map_b = u_map.unsqueeze(0)
                                        
                                        # 5. Observation Likelihood (Geometry / SCE)
                                        L_obs = (sce_src / (u_final_b**2 + 1e-6)) + torch.log(u_final_b + 1e-6)
                                        
                                        # 6. Prior Likelihood (Log-Normal Photometric Anchor)
                                        # lambda_prior controls how strictly the Tracker must adhere to the Mapper's belief
                                        lambda_prior = 0.0
                                        L_prior = lambda_prior * (torch.log(u_final_b + 1e-6) - torch.log(u_map_b + 1e-6))**2
                                        
                                        # 7. Total NLL
                                        nll = (L_obs + L_prior) * valid_src
                                        n_valid = valid_src.sum().clamp(min=1.0)
                                        
                                        total_loss += nll.sum() / n_valid
                                        n_frames += 1

                                    if n_frames > 0:
                                        # Sum across frames (no / n_frames). Each frame's NLL is already
                                        # per-pixel via / n_valid. More frames = more evidence = stronger update.
                                        # L2 weight decay on the A,B matrices acts as the native regularizer against the mapper.
                                        self.tracker_optimizer.zero_grad()
                                        total_loss.backward()
                                        self.tracker_optimizer.step()
                                    
                                    self.tracker_net.eval()
                                    self._tracker_net_trained = True

                                    # Experience Replay: save geometric pseudo-labels and immediate inference
                                    if hasattr(self, 'geom_uncertainty_labels') and self.is_initialized_fully:
                                        with torch.no_grad():
                                            for src_idx in unique_srcs:
                                                src = src_idx.item()
                                                feats = self.dino_feats_resize[src].to(self.device)
                                                feats_hw = feats.permute(1, 2, 0)
                                                
                                                u_final = self.tracker_net(feats_hw).view(H_s, W_s)
                                                
                                                # Store raw σ for distillation (same space as Mapper's u_map)
                                                self.geom_uncertainty_labels[src] = u_final
                                                
                                                # # Optional periodic print statement
                                                # self._print_uncer_counter = getattr(self, '_print_uncer_counter', 0) + 1
                                                # if self._print_uncer_counter % 10 == 0:
                                                #     from src.utils.Printer import FontColor
                                                #     print(f"{FontColor.INFO}Tracker Stats - Min: {u_final.min().item():.3f}, Max: {u_final.max().item():.3f}, Mean: {u_final.mean().item():.3f}")
                                                
                                                # # Gaussian precision: 0.5/u² (unified with Mapper's convention)
                                                # u ≤ 0.707 → weight 1.0 (background safe)
                                                # u = 1.0   → weight 0.50
                                                # u = 2.0   → weight 0.125 (dynamic crushed)
                                                # To get ABSOLUTE ZERO weights for dynamic objects, we drop the asymptotic inverse formula
                                                # and use a direct linear mapping with a hard clamp.
                                                # u <= 0.6 (Background) -> Weight = 1.0
                                                # u >= 1.0 (Dynamic)    -> Weight = 0.0
                                                # Map: u <= 0.7 -> u_scaled <= 0.1 (Background safe)
                                                #      u >= 1.1 -> u_scaled >= 10.0 (Dynamic crushed)
                                                m_scale = 45.0
                                                c_scale = 35.5
                                                # Denominator MUST be clamped to 0.1, otherwise background (negative values) breaks
                                                u_final_scaled = torch.clamp(m_scale * u_final - c_scale, min=0.1)
                                                # Simple inverse-square effectively crushes dynamic weights to 0.01
                                                w_uncer = torch.clamp(0.5 / (u_final_scaled**2 + 1e-6), min=0.01, max=1.0)
                                                
                                                self.uncertainties_inv[src] = w_uncer


    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            if self.metric_depth_reg:
                est_disp = self.mono_disps_up[index].clone().to(device)  # [h, w]
                est_depth = torch.where(est_disp>0.0, 1.0 / (est_disp), 0.0)
                depth_mask = torch.ones_like(est_disp,dtype=torch.bool).to(device)
                c2w = self.get_pose(index,device)
            else:
                est_disp = self.disps_up[index].clone().to(device)  # [h, w]
                est_depth = 1.0 / (est_disp)
                depth_mask = self.valid_depth_mask[index].clone().to(device)
                c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    @torch.no_grad()
    def update_all_uncertainty_mask(self):
        if not self.uncertainty_aware:
            # we only estimate uncertainty when we activate the mode
            raise Exception('This function should not be called if uncertainty aware is not activated')

        i = 0
        while i*20 < self.counter.value:
            batch = slice(i*20, min((i+1)*20, self.counter.value))
            dino_feat_batch = self.dino_feats[batch].to(self.device)
            
            with Lock():
                u_map_raw, _ = self.uncer_network(dino_feat_batch)
                u_map = u_map_raw.clone()
    
            train_frac = self.cfg['mapping']['uncertainty_params']['train_frac_fix']

            h = self.images.shape[2]
            w = self.images.shape[3]
            
            u_map = torch.clip(u_map, min=0.1) + 1e-3
            u_map = u_map.unsqueeze(1)
            u_map = F.interpolate(u_map, size=(h, w), mode="bilinear", align_corners=False).squeeze(1).detach()
            data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
            u_map = u_map[:, self.slice_h, self.slice_w]
            u_map = (u_map - 0.1) * data_rate + 0.1
            self.mapping_uncertainties_inv[batch,:,:] = torch.clamp(0.5/u_map**2, 0.0, 1.0)
        

            i += 1

    @torch.no_grad()
    def update_uncertainty_mask_given_index(self, idxs):
        if not self.uncertainty_aware:
            # we only estimate uncertainty when we activate the mode
            raise Exception('This function should not be called if uncertainty aware is not activated')

        dino_feat_batch = self.dino_feats[idxs,:,:,:].to(self.device)
        
        with Lock():
            u_map_raw, _ = self.uncer_network(dino_feat_batch)
            u_map = u_map_raw.clone()

        train_frac = self.cfg['mapping']['uncertainty_params']['train_frac_fix']

        h = self.images.shape[2]
        w = self.images.shape[3]
        u_map = torch.clip(u_map, min=0.1) + 1e-3

        if u_map.dim() == 2:
            u_map = u_map.unsqueeze(0)

        u_map = u_map.unsqueeze(1)
        u_map = F.interpolate(u_map, size=(h, w), mode="bilinear", align_corners=False).squeeze(1).detach()
        data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
        
        u_map = u_map[:, self.slice_h, self.slice_w]
        u_map = (u_map - 0.1) * data_rate + 0.1

        self.mapping_uncertainties_inv[idxs,:,:] = torch.clamp(0.5/u_map**2, 0.0, 1.0)
        

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        
        save_dict = {
            "poses": poses,
            "depths": depths,
            "timestamps": timestamps,
            "valid_depth_masks": valid_depth_masks
        }
        if self.uncertainty_aware:
            self.update_all_uncertainty_mask()
            save_dict["tracking_uncertainty"] = self.uncertainties_inv[:self.counter.value].cpu().numpy()
            save_dict["mapping_uncertainty"] = self.mapping_uncertainties_inv[:self.counter.value].cpu().numpy()

        np.savez(path, **save_dict)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)


    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        """This is from splat-slam, not used in WildGS-SLAM
        """
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()