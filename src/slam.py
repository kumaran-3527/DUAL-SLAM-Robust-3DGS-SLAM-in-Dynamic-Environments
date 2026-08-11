import os
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
from munch import munchify

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.utils.datasets import RGB_NoPose
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.droid_net: DroidNet = DroidNet()

        self.printer = Printer(
            len(stream)
        )  # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            n_features = self.cfg["mapping"]["uncertainty_params"]["feature_dim"]
            self.uncer_network = generate_uncertainty_mlp(n_features)
            self.uncer_network.share_memory()
        else:
            self.uncer_network = None
            if self.cfg["tracking"]["uncertainty_params"]["activate"]:
                raise ValueError(
                    "uncertainty estimation cannot be activated on tracking while not on mapping"
                )

        self.video = DepthVideo(cfg, self.printer, uncer_network=self.uncer_network)
        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self, pipe):
        self.tracker = Tracker(self, pipe)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Tracking Starts!", FontColor.TRACKER)
        self.printer.pbar_ready()
        
        start_time = time.time()
        self.tracker.run(self.stream)
        end_time = time.time()
        
        tracking_time = getattr(self.tracker, 'active_time', end_time - start_time)
        tracking_fps = len(self.stream) / tracking_time if tracking_time > 0 else 0
        self.printer.print(f"Tracking Done! (Time: {tracking_time:.2f}s, FPS: {tracking_fps:.2f})", FontColor.TRACKER)
        
        with open(f"{self.save_dir}/fps_stats.txt", "a") as f:
            f.write(f"Tracking Thread - Frames: {len(self.stream)}, Time: {tracking_time:.4f}s, FPS: {tracking_fps:.4f}\n")

    def mapping(self, pipe, q_main2vis, q_vis2main):
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        else:
            self.mapper = Mapper(self, pipe, None, q_main2vis, q_vis2main)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Mapping Starts!", FontColor.MAPPER)
        
        start_time = time.time()
        self.mapper.run()
        end_time = time.time()
        
        mapping_time = getattr(self.mapper, 'active_time', end_time - start_time)
        mapping_fps = len(self.stream) / mapping_time if mapping_time > 0 else 0
        self.printer.print(f"Mapping Done! (Time: {mapping_time:.2f}s, FPS: {mapping_fps:.2f})", FontColor.MAPPER)
        
        with open(f"{self.save_dir}/fps_stats.txt", "a") as f:
            f.write(f"Mapping Thread - Frames: {len(self.stream)}, Time: {mapping_time:.4f}s, FPS: {mapping_fps:.4f}\n")

        # Calculate Whole Pipeline FPS BEFORE the offline terminate/evaluation functions
        pipeline_time = end_time - start_time
        pipeline_fps = len(self.stream) / pipeline_time if pipeline_time > 0 else 0
        self.printer.print(f"Whole Pipeline Done! (Total Time: {pipeline_time:.2f}s, Overall FPS: {pipeline_fps:.2f})", FontColor.INFO)
        
        with open(f"{self.save_dir}/fps_stats.txt", "a") as f:
            f.write(f"Whole Pipeline - Frames: {len(self.stream)}, Time: {pipeline_time:.4f}s, FPS: {pipeline_fps:.4f}\n")

        self.terminate()

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        metric_depth_reg_activated = self.video.metric_depth_reg
        if metric_depth_reg_activated:
            self.video.metric_depth_reg = False

        self.ba = Backend(self.droid_net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

        if metric_depth_reg_activated:
            self.video.metric_depth_reg = True

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""

        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            self.video.save_video(f"{self.save_dir}/video.npz")
            if not isinstance(self.stream, RGB_NoPose):
                try:
                    ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj/before_final_ba",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)

            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="before_refine",
            )

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, RGB_NoPose):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(str(e), FontColor.ERROR)

        if self.cfg['mapping']['masked_render_eval'] :
            self.mapper.eval_mapping_metrics(suffix="before_refine")

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.mapper.final_refine(
                iters=self.cfg["mapping"]["final_refine_iters"]
            )  # this performs a set of optimizations with RGBD loss to correct

        # Evaluate the metrics
        self.mapper.save_all_kf_figs(
            self.save_dir,
            iteration="after_refine",
        )
        
        if self.cfg['mapping']['masked_render_eval'] :
            self.mapper.eval_mapping_metrics(suffix="after_refine")

        ## Not used, see head comments of the function
        # self._eval_depth_all(ate_statistics, global_scale, r_a, t_a)

        # Regenerate feature extractor for non-keyframes
        self.traj_filler.setup_feature_extractor()
        traj_est_not_align, _, _, dino_feats = full_traj_eval(
            self.traj_filler,
            self.mapper,
            f"{self.save_dir}/traj",
            "full_traj",
            self.stream,
            self.logger,
            self.printer,
            self.cfg['fast_mode'],
        )

        if self.cfg["data"]["colmap"]["export"]:
            self.save_colmap_format_kf_dynrm(traj_est_not_align, dino_feats, self.cfg["data"]["colmap"]["keyframe_only"])

        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            torch.save(
                self.mapper.uncer_network.state_dict(),
                self.save_dir + "/uncertainty_mlp_weight.pth",
            )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

        # Cleanup: Delete the heavy mono_priors disk cache to free up memory
        import shutil
        mono_priors_dir = os.path.join(self.save_dir, "mono_priors")
        if os.path.exists(mono_priors_dir):
            shutil.rmtree(mono_priors_dir)
            self.printer.print("Cleaned up mono_priors disk cache.", FontColor.INFO)

    def save_colmap_format_kf_dynrm(self, traj_est, dino_feats=None, keyframe_only=True):
        import shutil
        import cv2
        import torch
        import numpy as np
        from scipy.spatial.transform import Rotation
        from src.utils.dyn_uncertainty import mapping_utils as map_utils
        from thirdparty.gaussian_splatting.gaussian_renderer import render
        from src.utils.camera_utils import Camera
        
        colmap_dir = os.path.join(self.save_dir, "colmap")
        sparse_dir = os.path.join(colmap_dir, "sparse", "0")
        images_dir = os.path.join(colmap_dir, "images")
        
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # 1. cameras.txt
        with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {self.W} {self.H} {self.fx} {self.fy} {self.cx} {self.cy}\n")
            
        # 2. images.txt & Copy/Mask images
        with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            if keyframe_only:
                indices_to_process = sorted(list(self.mapper.cameras.keys()))
            else:
                indices_to_process = range(len(traj_est))
            
            for colmap_img_id, idx in enumerate(indices_to_process):
                if keyframe_only:
                    viewpoint = self.mapper.cameras[idx]
                    actual_frame_idx = int(self.video.timestamp[idx])
                else:
                    actual_frame_idx = idx
                    # Create temporary camera for non-keyframe pose
                    c2w = torch.from_numpy(traj_est[actual_frame_idx]).to(self.device).float()
                    w2c = torch.linalg.inv(c2w)
                    
                    camera_data = {
                        "idx": actual_frame_idx,
                        "gt_color": torch.zeros((3, self.H, self.W), device=self.device),
                        "est_depth": np.zeros((self.H, self.W)),
                        "est_pose": w2c,
                        "features": None,
                    }
                    
                    viewpoint = Camera.init_from_dataset(
                        self.mapper.frame_reader,
                        camera_data,
                        self.mapper.projection_matrix,
                        full_resol=self.cfg["mapping"]["full_resolution"],
                    )
                    viewpoint.update_RT(w2c[:3, :3], w2c[:3, 3])

                # Render the image from the Gaussian model
                render_pkg = render(
                    viewpoint, self.mapper.gaussians, self.mapper.pipeline_params, self.mapper.background
                )
                rendered_img = render_pkg["render"].detach() ## c,h,w
                rendered_img = torch.clamp(rendered_img, 0.0, 1.0) ## c,h,w
                img = (rendered_img.cpu().numpy().transpose((1, 2, 0)) * 255.0).astype(np.uint8) # h,w,c
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                img_name = f"{actual_frame_idx:05d}.png"
                dst_img_path = os.path.join(images_dir, img_name)
                cv2.imwrite(dst_img_path, img)
                
                c2w = traj_est[actual_frame_idx] ## c2w is wrt camera frame/original frame
                w2c = np.linalg.inv(c2w) ## w2c is wrt world frame/colmap frame
                R = w2c[:3, :3]
                T = w2c[:3, 3]
                quat = Rotation.from_matrix(R).as_quat() # x, y, z, w
                qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
                
                f.write(f"{colmap_img_id+1} {qw} {qx} {qy} {qz} {T[0]} {T[1]} {T[2]} 1 {img_name}\n\n") 
                
        # 3. points3D.txt
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            xyz = self.mapper.gaussians.get_xyz.detach().cpu().numpy()
            features = self.mapper.gaussians.get_features.detach().cpu().numpy()
            f_dc = features[:, 0, :]
            SH0 = 0.28209479177387814
            colors = np.clip(f_dc * SH0 + 0.5, 0, 1) * 255
            colors = colors.astype(np.uint8)
            
            for i in range(xyz.shape[0]):
                f.write(f"{i+1} {xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]} 0.0\n")

        self.printer.print(f"Saved COLMAP format in {colmap_dir}!", FontColor.INFO)


    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        """From Splat-SLAM. Not used in WildGS-SLAM evaluation, but might be useful in the future."""
        # Evaluate depth error
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,  # How much of each frame uses depth from droid (the rest from Omnidata)
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        # Write to the file
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

    def run(self):
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,q_main2vis,q_vis2main)),
        ]
        self.num_running_thread += len(processes)
        if self.cfg['gui']:
            self.num_running_thread += 1
            
        start_time = time.time()
        for p in processes:
            p.start()

        if self.cfg['gui']:
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            self.all_trigered += 1


        for p in processes:
            p.join()

        end_time = time.time()
        pipeline_time_with_eval = end_time - start_time
        self.printer.print(f"System Shutdown. (Total Time including offline eval: {pipeline_time_with_eval:.2f}s)", FontColor.INFO)

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()


def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
