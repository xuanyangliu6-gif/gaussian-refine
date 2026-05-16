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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    从某个相机视角渲染当前 GaussianModel。

    这是 3DGS 的可微前向渲染入口，对应论文里的 tile-based differentiable rasterization。
    Python 侧主要准备参数；真正高性能的投影、排序、alpha blending 在
    diff_gaussian_rasterization CUDA 扩展里完成。
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # 创建一个与 3D 点数量相同的“屏幕空间占位张量”。
    # CUDA rasterizer 会把每个 3D Gaussian 投影到屏幕，反向传播时这里能拿到 2D 均值的梯度；
    # train.py 正是用这个梯度判断哪些高斯需要 densify。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 根据当前相机的 FoV、分辨率、外参/投影矩阵，构造 rasterizer 的配置。
    # tanfovx/tanfovy 用于把相机空间坐标投影到 NDC/屏幕空间。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Gaussian 参数。注意 get_opacity/get_scaling/get_rotation 都会做激活：
    # opacity 用 sigmoid 保证 0..1，scale 用 exp 保证为正，rotation 做 normalize 保证四元数有效。
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 3D 协方差 Σ 可由 scaling + rotation 组成。
    # 默认把 scale/rotation 交给 CUDA rasterizer 计算；如果开启 compute_cov3D_python，则在 Python 中预先算好。
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 颜色表示是球谐系数 SH：DC 项给基础颜色，高阶项表达视角相关外观。
    # 默认把 SH->RGB 交给 rasterizer；convert_SHs_python=True 时在 Python 中按视线方向求颜色。
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 每个高斯到相机中心的方向，就是球谐函数求值的方向。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize：把可见高斯 splat 到图像平面，并做前到后的 alpha compositing。
    # 返回 radii 是每个高斯在屏幕上的近似半径；radii==0 代表被视锥裁剪或没有贡献。
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # 可学习曝光矩阵：用于处理 train/test exposure 模式或图像间曝光差异。
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # visibility_filter 只保留本视角中实际可见的高斯。
    # densification 统计只更新这些高斯，避免不可见点的无意义梯度污染。
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
