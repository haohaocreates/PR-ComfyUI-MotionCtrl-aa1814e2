import argparse
import datetime
import glob
import json
import math
import os
import tempfile
import folder_paths

import imageio
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision
## note: decord should be imported after torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from .lvdm.models.samplers.ddim import DDIMSampler
from .main.evaluation.motionctrl_prompts_camerapose_trajs import (
    both_prompt_camerapose_traj, cmcm_prompt_camerapose, omom_prompt_traj)
from .main.evaluation.motionctrl_inference import motionctrl_sample,save_images,load_camera_pose,load_trajs,load_model_checkpoint,post_prompt,DEFAULT_NEGATIVE_PROMPT
from .utils.utils import instantiate_from_config
from .gradio_utils.traj_utils import process_points,get_flow
from PIL import Image, ImageFont, ImageDraw
from .gradio_utils.utils import vis_camera
from io import BytesIO

def process_camera(camera_pose_str,frame_length):
    RT=json.loads(camera_pose_str)
    for i in range(frame_length):
        if len(RT)<=i:
            RT.append(RT[len(RT)-1])
    
    if len(RT) > frame_length:
        RT = RT[:frame_length]
    
    RT = np.array(RT).reshape(-1, 3, 4)
    return RT


def process_camera_list(camera_pose_str,frame_length):
    RT=json.loads(camera_pose_str)
    for i in range(frame_length):
        if len(RT)<=i:
            RT.append(RT[len(RT)-1])
    
    if len(RT) > frame_length:
        RT = RT[:frame_length]
        
    RT = np.array(RT).reshape(-1, 3, 4)
    return RT

    
def process_traj(points_str,frame_length):
    points=json.loads(points_str)
    for i in range(frame_length):
        if len(points)<=i:
            points.append(points[len(points)-1])
    xy_range = 1024
    #points = process_points(points,frame_length)
    points = [[int(256*x/xy_range), int(256*y/xy_range)] for x,y in points]
    
    optical_flow = get_flow(points,frame_length)
    # optical_flow = torch.tensor(optical_flow).to(device)

    return optical_flow
    
def save_results(video, fps=10,traj="[]",draw_traj_dot=False,cameras=[],draw_camera_dot=False):
    
    # b,c,t,h,w
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, h, w*n, 3]
    
    path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    outframes=[]
    
    #writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for i in range(grid.shape[0]):
        img = grid[i].numpy()
        image=Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        #draw.ellipse((0,0,255,255),fill=(255,0,0), outline=(255,0,0))
        if draw_traj_dot:
            traj_list=json.loads(traj)
            
            #print(traj_point)
            size=3
            for j in range(grid.shape[0]):
                traj_point=traj_list[len(traj_list)-1]
                if len(traj_list)>j:
                    traj_point=traj_list[j]
                if i==j:
                    draw.ellipse((traj_point[0]/4-size,traj_point[1]/4-size,traj_point[0]/4+size,traj_point[1]/4+size),fill=(255,0,0), outline=(255,0,0))
                else:
                    draw.ellipse((traj_point[0]/4-size,traj_point[1]/4-size,traj_point[0]/4+size,traj_point[1]/4+size),fill=(255,255,255), outline=(255,255,255))
            
        if draw_traj_dot:
            fig = vis_camera(cameras,1,i)
            camimg=Image.open(BytesIO(fig.to_image('png',256,256)))
            image.paste(camimg,(0,0),camimg.convert('RGBA'))
        
        image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)  # Convert back to CxHxW
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        outframes.append(image_tensor_out)
        #writer.append_data(img)

    #writer.close()
    return torch.cat(tuple(outframes), dim=0).unsqueeze(0)

MOTION_CAMERA_OPTIONS = ["U", "D", "L", "R", "O", "O_0.2x", "O_0.4x", "O_1.0x", "O_2.0x", "O_0.2x", "O_0.2x", "Round-RI", "Round-RI_90", "Round-RI-120", "Round-ZoomIn", "SPIN-ACW-60", "SPIN-CW-60", "I", "I_0.2x", "I_0.4x", "I_1.0x", "I_2.0x", "1424acd0007d40b5", "d971457c81bca597", "018f7907401f2fef", "088b93f15ca8745d", "b133a504fc90a2d1"]

MOTION_TRAJ_OPTIONS = ["curve_1", "curve_2", "curve_3", "curve_4", "horizon_2", "shake_1", "shake_2", "shaking_10"]

        
def read_points(file, video_len=16, reverse=False):
    with open(file, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(',')
        points.append((int(x)*4, int(y)*4))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]
    
    return points
    
class LoadMotionCameraPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_camera": (MOTION_CAMERA_OPTIONS,),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_camera_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_camera_preset(self, motion_camera):
        data="[]"
        comfy_path = os.path.dirname(folder_paths.__file__)
        with open(f'{comfy_path}/custom_nodes/ComfyUI-MotionCtrl/examples/camera_poses/test_camera_{motion_camera}.json') as f:
            data = f.read()
        
        return (data,)
        

class LoadMotionTrajPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_traj": (MOTION_TRAJ_OPTIONS,),
                "frame_length": ("INT", {"default": 16}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_traj_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_traj_preset(self, motion_traj, frame_length):
        comfy_path = os.path.dirname(folder_paths.__file__)
        points = read_points(f'{comfy_path}/custom_nodes/ComfyUI-MotionCtrl/examples/trajectories/{motion_traj}.txt',frame_length)
        return (json.dumps(points),)

MODE = ["control camera poses", "control object trajectory", "control both camera and object motion"]
class MotionctrlLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "motionctrl.pth"}),
                "frame_length": ("INT", {"default": 16}),
            }
        }
        
    RETURN_TYPES = ("MOTIONCTRL", "EMBEDDER", "VAE", "SAMPLER",)
    RETURN_NAMES = ("model","clip","vae","ddim_sampler",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "motionctrl"

    def load_checkpoint(self, ckpt_name, frame_length):
        gpu_num=1
        gpu_no=0
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        comfy_path = os.path.dirname(folder_paths.__file__)
        config_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/configs/inference/config_both.yaml')
        args={"ckpt_path":f"{ckpt_path}","adapter_ckpt":None,"base":f"{config_path}","condtype":"both","prompt_dir":None,"n_samples":1,"ddim_steps":50,"ddim_eta":1.0,"bs":1,"height":256,"width":256,"unconditional_guidance_scale":1.0,"unconditional_guidance_scale_temporal":None,"seed":1234,"cond_T":800}
        
        config = OmegaConf.load(args["base"])
        OmegaConf.update(config, "model.params.unet_config.params.temporal_length", frame_length)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(gpu_no)
        assert os.path.exists(args["ckpt_path"]), f'Error: checkpoint {args["ckpt_path"]} Not Found!'
        print(f'Loading checkpoint from {args["ckpt_path"]}')
        model = load_model_checkpoint(model, args["ckpt_path"], args["adapter_ckpt"])
        model.eval()

        ddim_sampler = DDIMSampler(model)

        return (model,model.cond_stage_model,model.first_stage_model,ddim_sampler,)


class MotionctrlCond:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOTIONCTRL",),
                "prompt": ("STRING", {"multiline": True, "default":"a rose swaying in the wind"}),
                "camera": ("STRING", {"multiline": True, "default":"[[1,0,0,0,0,1,0,0,0,0,1,0.2]]"}),
                "traj": ("STRING", {"multiline": True, "default":"[[117, 102]]"}),
                "infer_mode": (MODE, {"default":"control both camera and object motion"}),
                "context_overlap": ("INT", {"default": 0, "min": 0, "max": 32}),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING","TRAJ_LIST","RT_LIST","TRAJ_FEATURES","RT","NOISE_SHAPE","INT")
    RETURN_NAMES = ("positive", "negative","traj_list","rt_list","traj","rt","noise_shape","context_overlap")
    FUNCTION = "load_cond"
    CATEGORY = "motionctrl"

    def load_cond(self, model, prompt, camera, traj,infer_mode,context_overlap):
        comfy_path = os.path.dirname(folder_paths.__file__)
        camera_align_file = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/camera.json')
        traj_align_file = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/traj.json')
        frame_length=model.temporal_length

        camera_align=json.loads(camera)
        for i in range(frame_length):
            if len(camera_align)<=i:
                camera_align.append(camera_align[len(camera_align)-1])
        camera=json.dumps(camera_align)
        traj_align=json.loads(traj)
        for i in range(frame_length):
            if len(traj_align)<=i:
                traj_align.append(traj_align[len(traj_align)-1])
        traj=json.dumps(traj_align)

        if context_overlap>0:
            if os.path.exists(camera_align_file):
                with open(camera_align_file, 'r') as file:
                    pre_camera_align=json.load(file)
                    camera_align=pre_camera_align[:context_overlap]+camera_align[:-context_overlap]

            if os.path.exists(traj_align_file):
                with open(traj_align_file, 'r') as file:
                    pre_traj_align=json.load(file)
                    traj_align=pre_traj_align[:context_overlap]+traj_align[:-context_overlap]

            with open(camera_align_file, 'w') as file:
                json.dump(camera_align, file)

            with open(traj_align_file, 'w') as file:
                json.dump(traj_align, file)
        
        prompts = prompt
        RT = process_camera(camera,frame_length).reshape(-1,12)
        RT_list = process_camera_list(camera,frame_length)
        traj_flow = process_traj(traj,frame_length).transpose(3,0,1,2)
        print(prompts)
        print(RT.shape)
        print(traj_flow.shape)

        height=256
        width=256

        ## run over data
        assert (height % 16 == 0) and (width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        
        ## latent noise shape
        h, w = height // 8, width // 8
        channels = model.channels
        frames = model.temporal_length
        #frames = frame_length
        noise_shape = [1, channels, frames, h, w]

        if infer_mode == MODE[0]:
            camera_poses = RT
            camera_poses = torch.tensor(camera_poses).float()
            camera_poses = camera_poses.unsqueeze(0)
            trajs = None
            if torch.cuda.is_available():
                camera_poses = camera_poses.cuda()
        elif infer_mode == MODE[1]:
            trajs = traj_flow
            trajs = torch.tensor(trajs).float()
            trajs = trajs.unsqueeze(0)
            camera_poses = None
            if torch.cuda.is_available():
                trajs = trajs.cuda()
        else:
            camera_poses = RT
            trajs = traj_flow
            camera_poses = torch.tensor(camera_poses).float()
            trajs = torch.tensor(trajs).float()
            camera_poses = camera_poses.unsqueeze(0)
            trajs = trajs.unsqueeze(0)
            if torch.cuda.is_available():
                camera_poses = camera_poses.cuda()
                trajs = trajs.cuda()
        
        batch_size = noise_shape[0]
        prompts=prompt
        ## get condition embeddings (support single prompt only)
        if isinstance(prompts, str):
            prompts = [prompts]

        for i in range(len(prompts)):
            prompts[i] = f'{prompts[i]}, {post_prompt}'

        cond = model.get_learned_conditioning(prompts)
        if camera_poses is not None:
            RT = camera_poses[..., None]
        else:
            RT = None

        traj_features = None
        if trajs is not None:
            traj_features = model.get_traj_features(trajs)
        else:
            traj_features = None
        
        uc = None
        prompts = batch_size * [DEFAULT_NEGATIVE_PROMPT]
        uc = model.get_learned_conditioning(prompts)
        if traj_features is not None:
            un_motion = model.get_traj_features(torch.zeros_like(trajs))
        else:
            un_motion = None
        uc = {"features_adapter": un_motion, "uc": uc}

        return (cond,uc,traj,RT_list,traj_features,RT,noise_shape,context_overlap)



class MotionctrlSampleSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOTIONCTRL",),
                "clip": ("EMBEDDER",),
                "vae": ("VAE",),
                "ddim_sampler": ("SAMPLER",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "traj_list": ("TRAJ_LIST",),
                "rt_list": ("RT_LIST",),
                "traj": ("TRAJ_FEATURES",),
                "rt": ("RT",),
                "steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 1234}),
                "noise_shape":("NOISE_SHAPE",),
                "context_overlap": ("INT", {"default": 0, "min": 0, "max": 32}),
            },
            "optional": {
                "traj_tool": ("STRING",{"multiline": False, "default": "https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html"}),
                "draw_traj_dot": ("BOOLEAN", {"default": False}),#, "label_on": "draw", "label_off": "not draw"
                "draw_camera_dot": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"

    def run_inference(self,model,clip,vae,ddim_sampler,positive, negative,traj_list,rt_list,traj,rt,steps,seed,noise_shape,context_overlap,traj_tool="https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html",draw_traj_dot=False,draw_camera_dot=False):
        frame_length=model.temporal_length
        device = model.betas.device
        print(f'frame_length{frame_length}')
        #noise_shape = [1, 4, 16, 32, 32]
        unconditional_guidance_scale = 7.5
        unconditional_guidance_scale_temporal = None
        n_samples = 1
        ddim_steps= steps
        ddim_eta=1.0
        cond_T=800
        #seed = args["seed"]

        if n_samples < 1:
            n_samples = 1
        if n_samples > 4:
            n_samples = 4

        seed_everything(seed)

        batch_images=[]
        batch_variants = []
        intermediates = {}

        x0=None
        x_T=None
        pre_x0=None
        pre_x_T=None

        comfy_path = os.path.dirname(folder_paths.__file__)
        pred_x0_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/pred_x0.pt')
        x_inter_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/x_inter.pt')
        
        if context_overlap>0:
            if os.path.exists(pred_x0_path):
                pre_x0=torch.load(pred_x0_path)
            if os.path.exists(x_inter_path):
                pre_x_T=torch.load(x_inter_path)

            pre_x0_np=pre_x0[-1].detach().cpu().numpy()
            pre_x_T_np=pre_x_T[-1].detach().cpu().numpy()
            
            randt=torch.randn([noise_shape[0],noise_shape[1],frame_length-context_overlap,noise_shape[3],noise_shape[4]], device=device)
            randt_np=randt.detach().cpu().numpy()

            pre_x0_np_overlap = np.concatenate((pre_x0_np[:,:,-context_overlap:], randt_np), axis=2)
            x0=torch.tensor(pre_x0_np_overlap, device=device)
            pre_x_T_np_overlap = np.concatenate((pre_x_T_np[:,:,-context_overlap:], randt_np), axis=2)
            x_T=torch.tensor(pre_x_T_np_overlap, device=device)

        for _ in range(n_samples):
            if ddim_sampler is not None:
                samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                conditioning=positive,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=False,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=negative,
                                                eta=ddim_eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                                features_adapter=traj,
                                                pose_emb=rt,
                                                cond_T=cond_T,
                                                x0=x0,
                                                x_T=x_T
                                                )        
            #print(f'{samples}')
            ## reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
            '''
            batch_images = model.decode_first_stage(intermediates['pred_x0'][0])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['pred_x0'][1])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['pred_x0'][2])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['x_inter'][0])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['x_inter'][1])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['x_inter'][2])
            batch_variants.append(batch_images)
            '''
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants, dim=1)
        batch_variants = batch_variants[0]
        
        torch.save(intermediates['x_inter'], x_inter_path)
        torch.save(intermediates['pred_x0'], pred_x0_path)
        ret = save_results(batch_variants, fps=10,traj=traj_list,draw_traj_dot=draw_traj_dot,cameras=rt_list,draw_camera_dot=draw_camera_dot)
        #print(ret)
        return ret
        

class MotionctrlSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default":"a rose swaying in the wind"}),
                "camera": ("STRING", {"multiline": True, "default":"[[1,0,0,0,0,1,0,0,0,0,1,0.2]]"}),
                "traj": ("STRING", {"multiline": True, "default":"[[117, 102]]"}),
                "frame_length": ("INT", {"default": 16}),
                "steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 1234}),
            },
            "optional": {
                "traj_tool": ("STRING",{"multiline": False, "default": "https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html"}),
                "draw_traj_dot": ("BOOLEAN", {"default": False}),#, "label_on": "draw", "label_off": "not draw"
                "draw_camera_dot": ("BOOLEAN", {"default": False}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "motionctrl.pth"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"
        
    def run_inference(self,prompt,camera,traj,frame_length,steps,seed,traj_tool="https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html",draw_traj_dot=False,draw_camera_dot=False,ckpt_name="motionctrl.pth"):
        gpu_num=1
        gpu_no=0
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        comfy_path = os.path.dirname(folder_paths.__file__)
        config_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/configs/inference/config_both.yaml')
        args={"savedir":f'./output/both_seed20230211',"ckpt_path":f"{ckpt_path}","adapter_ckpt":None,"base":f"{config_path}","condtype":"both","prompt_dir":None,"n_samples":1,"ddim_steps":50,"ddim_eta":1.0,"bs":1,"height":256,"width":256,"unconditional_guidance_scale":1.0,"unconditional_guidance_scale_temporal":None,"seed":1234,"cond_T":800,"save_imgs":True,"cond_dir":"./custom_nodes/ComfyUI-MotionCtrl/examples/"}
        
        prompts = prompt
        RT = process_camera(camera,frame_length).reshape(-1,12)
        RT_list = process_camera_list(camera,frame_length)
        traj_flow = process_traj(traj,frame_length).transpose(3,0,1,2)
        print(prompts)
        print(RT.shape)
        print(traj_flow.shape)
        
        args["savedir"]=f'./output/{args["condtype"]}_seed{args["seed"]}'
        config = OmegaConf.load(args["base"])
        OmegaConf.update(config, "model.params.unet_config.params.temporal_length", frame_length)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(gpu_no)
        assert os.path.exists(args["ckpt_path"]), f'Error: checkpoint {args["ckpt_path"]} Not Found!'
        print(f'Loading checkpoint from {args["ckpt_path"]}')
        model = load_model_checkpoint(model, args["ckpt_path"], args["adapter_ckpt"])
        model.eval()
       
        ## run over data
        assert (args["height"] % 16 == 0) and (args["width"] % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        
        ## latent noise shape
        h, w = args["height"] // 8, args["width"] // 8
        channels = model.channels
        frames = model.temporal_length
        #frames = frame_length
        noise_shape = [args["bs"], channels, frames, h, w]

        savedir = os.path.join(args["savedir"], "samples")
        os.makedirs(savedir, exist_ok=True)
        
        #noise_shape = [1, 4, 16, 32, 32]
        unconditional_guidance_scale = 7.5
        unconditional_guidance_scale_temporal = None
        n_samples = 1
        ddim_steps= steps
        ddim_eta=1.0
        cond_T=800
        #seed = args["seed"]

        if n_samples < 1:
            n_samples = 1
        if n_samples > 4:
            n_samples = 4

        seed_everything(seed)
        
        camera_poses = RT
        trajs = traj_flow
        camera_poses = torch.tensor(camera_poses).float()
        trajs = torch.tensor(trajs).float()
        camera_poses = camera_poses.unsqueeze(0)
        trajs = trajs.unsqueeze(0)
        if torch.cuda.is_available():
            camera_poses = camera_poses.cuda()
            trajs = trajs.cuda()
        
        ddim_sampler = DDIMSampler(model)
        batch_size = noise_shape[0]
        prompts=prompt
        ## get condition embeddings (support single prompt only)
        if isinstance(prompts, str):
            prompts = [prompts]

        for i in range(len(prompts)):
            prompts[i] = f'{prompts[i]}, {post_prompt}'

        cond = model.get_learned_conditioning(prompts)
        if camera_poses is not None:
            RT = camera_poses[..., None]
        else:
            RT = None

        traj_features = None
        if trajs is not None:
            traj_features = model.get_traj_features(trajs)
        else:
            traj_features = None
            
        uc = None
        if unconditional_guidance_scale != 1.0:
            # prompts = batch_size * [""]
            prompts = batch_size * [DEFAULT_NEGATIVE_PROMPT]
            uc = model.get_learned_conditioning(prompts)
            if traj_features is not None:
                un_motion = model.get_traj_features(torch.zeros_like(trajs))
            else:
                un_motion = None
            uc = {"features_adapter": un_motion, "uc": uc}
        else:
            uc = None
        
        batch_images=[]
        batch_variants = []
        for _ in range(n_samples):
            if ddim_sampler is not None:
                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=False,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                                features_adapter=traj_features,
                                                pose_emb=RT,
                                                cond_T=cond_T
                                                )        
            #print(f'{samples}')
            ## reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants, dim=1)
        batch_variants = batch_variants[0]
        
        ret = save_results(batch_variants, fps=10,traj=traj,draw_traj_dot=draw_traj_dot,cameras=RT_list,draw_camera_dot=draw_camera_dot)
        #print(ret)
        return ret
        
        
class ImageSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "motionctrl"

    def run(self, images: torch.Tensor, selected_indexes: str):
        shape = images.shape
        len_first_dim = shape[0]

        selected_index: list[int] = []
        total_indexes: list[int] = list(range(len_first_dim))
        for s in selected_indexes.strip().split(','):
            try:
                if ":" in s:
                    _li = s.strip().split(':', maxsplit=1)
                    _start = _li[0]
                    _end = _li[1]
                    if _start and _end:
                        selected_index.extend(
                            total_indexes[int(_start):int(_end)]
                        )
                    elif _start:
                        selected_index.extend(
                            total_indexes[int(_start):]
                        )
                    elif _end:
                        selected_index.extend(
                            total_indexes[:int(_end)]
                        )
                else:
                    x: int = int(s.strip())
                    if x < len_first_dim:
                        selected_index.append(x)
            except:
                pass

        if selected_index:
            print(f"ImageSelector: selected: {len(selected_index)} images")
            return (images[selected_index, :, :, :], )

        print(f"ImageSelector: selected no images, passthrough")
        return (images, )


NODE_CLASS_MAPPINGS = {
    "Motionctrl Sample":MotionctrlSample,
    "Motionctrl Sample Simple":MotionctrlSampleSimple,
    "Load Motion Camera Preset":LoadMotionCameraPreset,
    "Load Motion Traj Preset":LoadMotionTrajPreset,
    "Select Image Indices": ImageSelector,
    "Load Motionctrl Checkpoint": MotionctrlLoader,
    "Motionctrl Cond": MotionctrlCond,
}