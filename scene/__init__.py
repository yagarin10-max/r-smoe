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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, BasicPointCloud
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import scipy

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            npcs = scene_info.point_cloud.points.shape[0]
            # random init
            select_index = np.random.randint(0,npcs,args.npcs)
            new_points = scene_info.point_cloud.points[select_index]
            new_points = np.random.rand(args.npcs,3) * 2 - 1
            # # seg init
            # label = scipy.io.loadmat('./seg/20/kodim_noisy_{}/{}.mat'.format(args.var, args.source_path.split('/')[-1]))['label']
            # #init_loc = np.zeros(label.shape)
            # k_per_seg = int(args.npcs / label.max())
            # select_xs = []
            # select_ys = []
            # for l in range(label.max()+1):
            #     candidate_index = np.where(label == l+1)
            #     select_index = np.random.choice(len(candidate_index[0]), np.min([len(candidate_index[0]),k_per_seg]))

            #     select_x = candidate_index[0][select_index]
            #     select_y = candidate_index[1][select_index]

            #     select_xs += list(select_x)
            #     select_ys += list(select_y)
            #     #init_loc[select_x,select_y] = l+1
            # new_points = np.random.rand(len(select_xs),3) * 2 - 1
            # train_camera = self.getTrainCameras()[0]
            # new_points[:,0] = np.array(select_ys) / train_camera.image_width * 2 - 1
            # new_points[:,1] = np.array(select_xs) / train_camera.image_height * 2 - 1
            # print('-------------------- new_points:  ', new_points.shape, np.random.rand(args.npcs,3).shape)
            # color 
            #print(new_points)
            train_camera = self.getTrainCameras()[0]
            original_image = train_camera.original_image.cuda()
            new_colors = scene_info.point_cloud.colors[select_index]
            #new_colors = scene_info.point_cloud.colors[np.arange(len(select_xs))]
            #print(original_image[:,100:110,0])
            print(original_image.shape,train_camera.image_height,train_camera.image_width)
            #print(new_colors[0:10])
            for i,c in enumerate(new_colors):
                #print(new_points[i,0],new_points[i,1])
                new_colors[i] = original_image[:,int((new_points[i,1] / 2 + 0.5) * train_camera.image_height) ,int((new_points[i,0] / 2 + 0.5)* train_camera.image_width) ].cpu()
                #print(original_image[:,int(new_points[i,0] / 2 + 0.5) * train_camera.image_width,int(new_points[i,1] / 2 + 0.5) * train_camera.image_height])
                #print(int(new_points[i,0] / 2 + 0.5) * train_camera.image_width,int(new_points[i,1] / 2 + 0.5) * train_camera.image_height)
            # #print(new_colors[0:10])
            new_normals = scene_info.point_cloud.normals[select_index]
            print(new_points.shape)
            new_pcs = BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals)
            #print('cameras_extent: ', self.cameras_extent)
            # if args.npcs > 30000:
            #     rot_scale = 1
            # else:
            #     rot_scale = 5
            #if args.npcs > 30000:
            if args.npcs > 30000:#30000
                rot_scale = 1
            else:
                rot_scale = 5#5#5
            #rot_scale = 5
            self.gaussians.create_from_pcd(new_pcs, self.cameras_extent, rot_scale)
            #self.gaussians.create_from_pcd_3d(scene_info.point_cloud, self.cameras_extent)
            #self.gaussians.load_ply(os.path.join(args.source_path,"sparse","0","points3D.ply"))

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]