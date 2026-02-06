#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui, render_gaussian_shapes
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from os import makedirs
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
import torchvision
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, file_name):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    #scene = Scene(dataset, gaussians)
    scene = Scene(dataset, gaussians)#,load_iteration=5000)
    if dataset.npcs > 30000:#30000:    global: 5000, R: 30000
        # SMoE
        opt.rotation_lr = 0.01
        # GS
        #opt.rotation_lr = 0.001
        
    else:
        # SMoE
        opt.rotation_lr = 0.1
        # GS
        #opt.rotation_lr = 0.1
    # if dataset.npcs <= 4000:
    #     #opt.rotation_lr = 0.01
    #     opt.rotation_lr = 0.005
    # else:
    #     opt.rotation_lr = 0.001
    #opt.rotation_lr = 0.001
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    current_loss = 1000000
    losses = []
    times = []
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        if dataset.npcs > 10000:# and iteration > 4000:    global: 1000, R: 10000
            gaussians.update_learning_rate(iteration)
        #if iteration % 5000 == 0:
        #    gaussians.update_learning_rate(iteration)
        #print(gaussians.optimizer)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        #print('before: ',gaussians._features_dc[3])
        #print('Image max: ',image.max())
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        #print(gt_image.shape,gt_image[:,100,100])
        Ll1 = l1_loss(image, gt_image)
        Ll2 = l2_loss(image, gt_image)

        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll2 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()
        #print('xyz: ',gaussians._xyz.max(), '\nrotation: ',gaussians._rotation.max(),'\nfeature: ',gaussians._features_dc.max())
        #print('xyz: ',gaussians._xyz.min(), '\nrotation: ',gaussians._rotation.min(),'\nfeature: ',gaussians._features_dc.min())
        #print('feature: ',gaussians._features_dc.max(),gaussians._features_dc[3], '\ngrad: ',gaussians._features_dc.grad.max(),gaussians._features_dc.grad.min())
        
        #print('rotation: ',gaussians._rotation[3], '\ngrad: ',gaussians._rotation.grad[3])
        #gaussians._xyz -= gaussians._xyz.grad
        #if iteration == 10:
            #break
        with torch.no_grad():
            # Progress bar
            #ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_for_log = loss.item()
            losses.append(loss.item())
            times.append(progress_bar.format_dict['elapsed'])
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                
            if iteration == opt.iterations:
                training_time = progress_bar.format_dict['elapsed']
                progress_bar.close()
                

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),dataset)
            if (iteration in saving_iterations) and ema_loss_for_log < current_loss:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                current_loss = ema_loss_for_log


            # Densification
            #if iteration < opt.densify_until_iter:
            if False:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    np.save(scene.model_path + "/loss.npy", losses)
    np.save(scene.model_path + "/time.npy", times)

    render_path = os.path.join(scene.model_path, 'train', "ours_{}".format(opt.iterations), "renders")
    gts_path = os.path.join(scene.model_path, 'train', "ours_{}".format(opt.iterations), "gt")
    render_points_path = os.path.join(scene.model_path, 'train', "ours_{}".format(opt.iterations), "points_overlay")
    gauss_render_path = os.path.join(scene.model_path, 'train', "ours_{}".format(opt.iterations), "gauss_renders")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_points_path, exist_ok=True)
    makedirs(gauss_render_path, exist_ok=True)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    rendering_time = 0
    full_dict = {}
    ssims = []
    psnrs = []
    lpipss = []
    viewpoint_stack = scene.getTrainCameras().copy()
    for idx, view in enumerate(tqdm(viewpoint_stack, desc="Rendering progress")):
        iter_start.record()
        rendering = torch.clamp(render(viewpoint_cam, gaussians, pipe, bg)["render"], 0.0, 1.0)
        iter_end.record()
        torch.cuda.synchronize()
        rendering_time += iter_start.elapsed_time(iter_end)
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        #psnr_v = psnr(rendering, gt).mean().double()
        #print(psnr_v)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        ssims.append(ssim(rendering, gt))
        psnrs.append(psnr(rendering, gt).mean())
        lpipss.append(lpips(rendering, gt, net_type='vgg'))

        # --- 呼び出し側の例 (Iterationの最後など) ---
        point_render_path = os.path.join(render_points_path, '{0:05d}_pts.png'.format(idx))
        save_points_overlay(rendering, gaussians, viewpoint_cam, point_render_path)
        print(f"Point overlay saved to {point_render_path}")
        black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        gauss_shapes = torch.clamp(render_gaussian_shapes(viewpoint_cam, gaussians, pipe, black_bg), 0.0, 1.0)
        save_points_overlay(gauss_shapes, gaussians, viewpoint_cam, os.path.join(gauss_render_path, '{0:05d}_gauss.png'.format(idx)))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict[scene.model_path] = {}
    full_dict[scene.model_path].update({"SSIM": torch.tensor(ssims).mean().item(),
                                  "PSNR": torch.tensor(psnrs).mean().item(),
                                  "LPIPS": torch.tensor(lpipss).mean().item(),
                                  "Rendering Time": rendering_time / len(viewpoint_stack),
                                  "Training Time": training_time})
    
    with open("all_result/{}.json".format(file_name), 'a') as fp:
        json.dump(full_dict, fp, indent=True)
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def save_points_overlay(render_img, gaussians, viewpoint_cam, save_path):
    """
    レンダリング画像の上にガウス関数の中心点をプロットして保存する
    """
    # 1. データの準備
    img_np = render_img.permute(1, 2, 0).detach().cpu().numpy()
    h, w, _ = img_np.shape
    
    # 3D中心点 (xyz) をスクリーン座標 (2D) に投影
    # R-SMoEでは z=0 平面に配置されている前提だが、レンダラーの投影行列を使うのが確実
    # ここではシンプルに現在のxyzから2D位置を取得
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    # R-SMoEの正規化範囲 (-1 to 1) をピクセル座標に変換
    # scene/__init__.py の初期化ロジックと合わせる
    points_x = (xyz[:, 0] / 2 + 0.5) * viewpoint_cam.image_width
    points_y = (xyz[:, 1] / 2 + 0.5) * viewpoint_cam.image_height
    
    # 不透明度によるフィルタリング (ほぼ透明な点を除外)
    opacities = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    valid_indices = (opacities > 0.05) & \
                (points_x >= 0) & (points_x < viewpoint_cam.image_width) & \
                (points_y >= 0) & (points_y < viewpoint_cam.image_height)
    # 2. 描画
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(img_np)
    ax.scatter(points_x[valid_indices], points_y[valid_indices], 
            s=1, c='lime', marker='o', alpha=0.8, edgecolors='none')
    
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    # 3. 保存
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,dataset):
    #print(tb_writer)
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                l1_noise_test = 0.0
                psnr_noise_test = 0.0
                ssim_noise_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    #noise_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    #gt_image = Image.open(os.path.join(dataset.source_path,'images',viewpoint.image_name+'.jpg'))
                    #res = (int(gt_image.size[0]), int(gt_image.size[1]))
                    #gt_image = PILtoTorch(gt_image, res).clamp(0.0, 1.0).to(viewpoint.data_device)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    #l1_noise_test += l1_loss(noise_image, gt_image).mean().double()
                    #psnr_noise_test += psnr(noise_image, gt_image).mean().double()
                    #ssim_noise_test += ssim(noise_image, gt_image).mean().double()


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                #psnr_noise_test /= len(config['cameras'])
                #l1_noise_test /= len(config['cameras'])
                #ssim_noise_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.2")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--file_name", type=str, default = 'new_dataset')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args.npcs)
    #lp.npcs = args.npcs
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.file_name)

    # All done
    print("\nTraining complete.")
