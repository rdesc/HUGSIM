import torch
from torch import nn

class Camera(nn.Module):
    def __init__(self, width, height, image, K, c2w,
                 image_name, data_device="cuda", 
                 semantic2d=None, depth=None, mask=None, timestamp=-1, optical_image=None, dynamics={}
                 ):
        super(Camera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
            
        self.width = width
        self.height = height
        self.image_name = image_name
        self.timestamp = timestamp
        self.K = torch.from_numpy(K).float().cuda()
        self.c2w = torch.from_numpy(c2w).float().cuda()
        self.dynamics = dynamics

        self.original_image = torch.from_numpy(image).permute(2,0,1).float().clamp(0.0, 1.0).to(self.data_device)
        if semantic2d is not None:
            self.semantic2d = semantic2d.to(self.data_device)
        else:
            self.semantic2d = None
        if depth is not None:
            self.depth = depth.to(self.data_device)
        else:
            self.depth = None
        if mask is not None:
            self.mask = torch.from_numpy(mask).bool().to(self.data_device)
        else:
            self.mask = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if optical_image is not None:
            self.optical_gt = torch.from_numpy(optical_image).to(self.data_device)
        else:
            self.optical_gt = None


def loadCam(args, cam_info):

    if cam_info.semantic2d is not None:
        semantic2d = torch.from_numpy(cam_info.semantic2d).long()[None, ...]
    else:
        semantic2d = None

    optical_image = cam_info.optical_image
    mask = cam_info.mask
    depth = cam_info.depth

    gt_image = cam_info.image[..., :3] / 255.

    return Camera(K=cam_info.K, c2w=cam_info.c2w, width=cam_info.width, height=cam_info.height,
                  image=gt_image, image_name=cam_info.image_name, data_device=args.model.data_device,
                  semantic2d=semantic2d, depth=depth, mask=mask,
                  timestamp=cam_info.timestamp, optical_image=optical_image, dynamics=cam_info.dynamics)

def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for c in cam_infos:
        camera_list.append(loadCam(args, c))

    return camera_list
