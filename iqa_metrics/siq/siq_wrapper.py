from iqa_metrics.siq.siq import get_siq
from utils.image_processing.image_tools import ensure3d


import numpy as np
import torch
import torchvision.transforms.functional as functional

import cv2
from PIL import Image

# singleton model to not reinitialize every time SIQ is called
siq_model_instance = None


def init_instance_siq(**kwargs):
    use_gpu = kwargs.pop("use_gpu", True)
    custom_model_path = kwargs.pop("custom_model_path", None)
    custom_model_path = "C:/Users/achuba/code/cnns/output/1599796129-TID2013-1000e-200b/best.pth"

    global siq_model_instance

    siq_model_instance = get_siq(use_gpu, custom_model_path)
    siq_model_instance.eval()

    print("Initialized SIQ instance.")


def compute_siq(img1, img2, **kwargs):
    global siq_model_instance

    data_range = kwargs.pop("data_range", 1.0)

    def cv2_to_tensor(img, data_range):
        img = ensure3d(255 * img / data_range).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return torch.reshape(functional.to_tensor(im_pil), [1, 3, 64, 64])

    img1 = cv2_to_tensor(img1, data_range)
    img2 = cv2_to_tensor(img2, data_range)

    # Compute distance
    q = float(siq_model_instance(img1, img2))

    return q

