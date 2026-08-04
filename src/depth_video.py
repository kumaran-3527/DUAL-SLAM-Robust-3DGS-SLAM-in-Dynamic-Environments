import numpy as np
import torch
import lietorch
import droid_backends
import mfc_backends
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
            

            
            self.geom_uncertainty_labels = -1.0 * torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            self._sce_ba_call_count = 0
            self._sce_new_frame_pending = False
            
            self.use_sce_init_gate = cfg['tracking']["uncertainty_params"].get('use_sce_init_gate', False)
            self.cached_sce_f = None
            self.cached_ii_filtered = None
            self.cached_jj_filtered = None

            from src.utils.dyn_uncertainty.uncertainty_model import LoRATrackerNet
            lora_rank = cfg['tracking']['uncertainty_params'].get('lora_rank', 8)
            lora_alpha = cfg['tracking']['uncertainty_params'].get('lora_alpha', 8.0)
            lora_lr = cfg['tracking']['uncertainty_params'].get('lora_lr', 4e-7)
            lora_wd = cfg['tracking']['uncertainty_params'].get('lora_wd', 0.1)
            
            self.tracker_net = LoRATrackerNet(base_net=self.uncer_network.net_fast, rank=lora_rank, alpha=lora_alpha).to(self.device)
            # Only optimize the LoRA parameters (A and B matrices)
            lora_params = [p for n, p in self.tracker_net.named_parameters() if 'lora_' in n]
            self.tracker_optimizer = torch.optim.AdamW(
                lora_params, lr=lora_lr, weight_decay=lora_wd
            )
        

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

        # We need jj features. jj is a tensor of indices.
        # feats_jj shape: [E, 384, H_feat, W_feat]
        feats_jj = self.dino_feats[jj.cpu()].permute(0, 3, 1, 2).contiguous().to(self.device)
        feats_jj.div_(feats_jj.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
        
        # We need ii features. ii is idx repeated E times.
        feats_ii = self.dino_feats[ii.cpu()].permute(0, 3, 1, 2).contiguous().to(self.device)
        feats_ii.div_(feats_ii.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))

        # x1 is [1, E, H_full, W_full, 3] -> [E, H_full, W_full, 3]
        x1_b = x1[0].contiguous()
        valid_mask_b = valid_mask[0].contiguous()

        # Execute fused zero-VRAM CUDA operation to get similarity > 0.9 mask!
        # Remember: we pass feats_jj first (sampled at px,py) and feats_ii second (sampled at w,h)
        matching_mask = mfc_backends.sce_feature_warp_full_res(
            feats_jj, 
            feats_ii, 
            x1_b,
            valid_mask_b
        )

        # matching_mask is [E, H, W]
        # For each E, we have a set of matched pixels in frame i.
        # The coordinates in frame i are x1_rounded[..., 0] and x1_rounded[..., 1].
        x1_rounded = torch.round(x1_b[..., :2]).long()
        
        for j_id in range(jj.shape[0]):
            m_mask = matching_mask[j_id] & valid_mask_b[j_id]
            
            # Extract the 2D spatial planes for coordinates and depth
            x_coords = x1_rounded[j_id, ..., 0]
            y_coords = x1_rounded[j_id, ..., 1]
            depth_plane = x1_b[j_id, ..., 2]
            
            # Apply the boolean mask to get valid 1D arrays
            valid_x = x_coords[m_mask]
            valid_y = y_coords[m_mask]
            depth_j = depth_plane[m_mask]
            
            # We compare depth_j against i_disp at (valid_x, valid_y)
            depth_i = i_disp[valid_y, valid_x]
            
            error = torch.abs(1.0 / depth_j - 1.0 / depth_i) * depth_j
            correct_mask = error < 0.02
            
            accurate_count[valid_y[correct_mask], valid_x[correct_mask]] += 1
            inaccurate_count[valid_y[~correct_mask], valid_x[~correct_mask]] += 1

        # Clean the gpu memory
        torch.cuda.empty_cache()

        self.mono_disps_mask_up[idx][(accurate_count<=1)&(inaccurate_count>0)&(self.mono_disps_up[idx]>0)] = False
        

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1,
           motion_only=False, global_ba=False):

        if self.uncertainty_aware:
            if not getattr(self, 'is_initialized', False) or motion_only or not self._tracker_net_trained:
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
            if self.uncertainty_aware:

                _run_sce = not global_ba
                if _run_sce:
                    self._sce_ba_call_count = 0
                    self._sce_new_frame_pending = False
                    with torch.no_grad():
                        # We consider ALL edges (both active and inactive) passed to ba()
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
                            
                            # -- NEW OCCLUSION CHECK --
                            use_occlusion = self.cfg['tracking']['uncertainty_params'].get('use_occlusion_check', False)
                            if use_occlusion:
                                # Sample the actual disparity of the target frame (jj) at the projected coordinates
                                disp_jj = self.disps[jj_filtered].unsqueeze(1)  # [E_f, 1, H_s, W_s]
                                sampled_disp_jj = torch.nn.functional.grid_sample(
                                    disp_jj, grid, mode='bilinear', padding_mode='zeros', align_corners=True
                                ).squeeze(1)  # [E_f, H_s, W_s]
                                
                                # Convert actual target inverse depth to metric depth
                                actual_Z_target = 1.0 / (sampled_disp_jj + 1e-6)
                                
                                occlusion_thresh = self.cfg['tracking']['uncertainty_params'].get('occlusion_thresh', 1.15)
                                # If the projected depth (Z_target) is much larger than the actual depth surface,
                                # the point is occluded by something closer in the target frame.
                                not_occluded = Z_target <= (actual_Z_target * occlusion_thresh)
                                
                                # Invalidating occluded pixels safely sets SCE = 0.0, avoiding false positives.
                                geom_valid = geom_valid & not_occluded


                            jj_cpu = jj_filtered.cpu()
                            ii_cpu = ii_filtered.cpu()

                            feats_jj = self.dino_feats_resize[jj_cpu].contiguous().to(self.device)
                            feats_ii = self.dino_feats_resize[ii_cpu].contiguous().to(self.device)

                            feats_jj.div_(feats_jj.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
                            feats_ii.div_(feats_ii.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))

                            sce_f = mfc_backends.sce_feature_warp(
                                feats_ii, feats_jj, grid, geom_valid
                            )

                            if self.use_sce_init_gate and not self.is_initialized_fully:
                                self.cached_sce_f = sce_f.detach().clone()
                                self.cached_ii_filtered = ii_filtered.clone()
                                self.cached_jj_filtered = jj_filtered.clone()

                            with torch.enable_grad():
                                # Train the tracker dynamically after Stage 1
                                if hasattr(self, 'tracker_net') and len(ii_filtered) > 0 and self.is_initialized:
                                    
                                    if self.cfg['tracking']['uncertainty_params'].get('freeze_base', True):
                                        import copy
                                        self.tracker_net.base_net = copy.deepcopy(self.uncer_network.net_fast)
                                        self.tracker_net.base_net.requires_grad_(False)
                                        self.tracker_net.base_net.eval()

                                    self.tracker_net.train()
                                    unique_srcs = torch.unique(ii_filtered)
                                    
                                    if len(unique_srcs) > 0:
                                        feats_b = self.dino_feats_resize[unique_srcs.cpu()].to(self.device)
                                        feats_bhwc = feats_b.permute(0, 2, 3, 1)  # [N, H, W, C]
                                        u_finals_b = self.tracker_net(feats_bhwc) # [N, H_s, W_s]
                                        
                                        total_loss = 0.0
                                        for batch_idx, src_idx in enumerate(unique_srcs):
                                            edge_mask = (ii_filtered == src_idx)
                                            u_final = u_finals_b[batch_idx]
                                            
                                            sce_src = sce_f[edge_mask]
                                            valid_src = geom_valid[edge_mask]
                                            u_final_b = u_final.unsqueeze(0)
                                            
                                            L_obs = (sce_src / (u_final_b**2 + 1e-6)) + torch.log(u_final_b + 1e-6)
                                            nll = L_obs * valid_src
                                            n_valid = valid_src.sum().clamp(min=1.0)
                                            total_loss += nll.sum() / n_valid
                                            
                                        # L2 weight decay on the A,B matrices acts as the native regularizer against the mapper.
                                        self.tracker_optimizer.zero_grad()
                                        total_loss.backward()
                                        self.tracker_optimizer.step()
                                        
                                    self.tracker_net.eval()
                                    self._tracker_net_trained = True

                                    # Experience Replay: save geometric pseudo-labels and immediate inference
                                    if hasattr(self, 'geom_uncertainty_labels') and len(unique_srcs) > 0:
                                        with torch.no_grad():
                                            # --- BATCHED FORWARD PASS (INFERENCE) ---
                                            # We re-run batched inference in eval() mode just to be structurally safe
                                            u_finals_eval = self.tracker_net(feats_bhwc)
                                            
                                            m_scale = 45.0
                                            c_scale = 35.5
                                            u_finals_scaled = torch.clamp(m_scale * u_finals_eval - c_scale, min=0.1)
                                            w_uncers = torch.clamp(0.5 / (u_finals_scaled**2 + 1e-6), min=0.01, max=1.0)
                                            
                                            for batch_idx, src_idx in enumerate(unique_srcs):
                                                src = src_idx.item()
                                                self.geom_uncertainty_labels[src] = u_finals_eval[batch_idx]
                                                self.uncertainties_inv[src] = w_uncers[batch_idx]



                                                # # Visualization Hook for Convergence Timeline
                                                # target_keyframes = [i for i in range(0,12,1)] # Edit this list to specify which keyframes to visualize
                                                # if src in target_keyframes:
                                                #     if not hasattr(self, 'uncer_history'):
                                                #         self.uncer_history = {}
                                                #     if not hasattr(self, 'ba_calls_per_frame'):
                                                #         self.ba_calls_per_frame = {}
                                                    
                                                #     if src not in self.ba_calls_per_frame:
                                                #         self.ba_calls_per_frame[src] = 0
                                                #         self.uncer_history[src] = []
                                                        
                                                #     if self.ba_calls_per_frame[src] < 15:
                                                #         tracking_uncer_map = (1.0 - w_uncers[batch_idx]).clone().cpu().numpy()
                                                #         self.uncer_history[src].append(tracking_uncer_map)
                                                #         self.ba_calls_per_frame[src] += 1
                                                        
                                                #         if self.ba_calls_per_frame[src] == 15:
                                                #             import matplotlib.pyplot as plt
                                                #             import os
                                                            
                                                #             save_dir = os.path.join(self.output, "tracker_convergence_vis")
                                                #             os.makedirs(save_dir, exist_ok=True)
                                                            
                                                #             # 1 RGB image + 8 iterations (0,2,4,6,8,10,12,14)
                                                #             fig, axs = plt.subplots(1, 9, figsize=(32, 2))
                                                            
                                                #             # Plot RGB Image
                                                #             rgb_img = self.images[src].permute(1, 2, 0).cpu().numpy()
                                                #             rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
                                                #             axs[0].imshow(rgb_img)
                                                #             axs[0].axis("off")
                                                #             axs[0].set_title("RGB")
                                                            
                                                #             # Plot Iterations
                                                #             for idx, i in enumerate(range(0,15,2)):
                                                #                 axs[idx+1].imshow(self.uncer_history[src][i], cmap="jet", vmin=0, vmax=1.0)
                                                #                 axs[idx+1].axis("off")
                                                #                 axs[idx+1].set_title(f"Iter {i}")
                                                            
                                                #             plt.tight_layout()
                                                #             plt.savefig(os.path.join(save_dir, f"frame_{src}_convergence_timeline.png"), bbox_inches="tight")
                                                #             plt.close()


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