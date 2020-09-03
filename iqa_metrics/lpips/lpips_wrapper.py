from .PerceptualSimilarity.util import util
from .PerceptualSimilarity.models import dist_model as dm

from utils.image_processing.image_tools import ensure3d

import torch

# singleton model to not reinitialize every time LPIPS is called
lpips_model_instance = None


def init_instance_lpips(**kwargs):
    use_gpu = kwargs.pop("use_gpu", True)
    model = kwargs.pop("model", "net-lin")
    net = kwargs.pop("net", "alex")
    attention_type = kwargs.pop("attention_type", None)

    global lpips_model_instance

    # Initializing the model
    if lpips_model_instance is None:
        # if not default LPIPS, need to be in train mode so that full model structure is initialized with weights
        is_train = model == "a2" or attention_type == "SAGAN" or net == "resnet50"

        lpips_model_instance = dm.DistModel()
        lpips_model_instance.initialize(use_gpu=use_gpu,
                                        model=model,
                                        net=net,
                                        is_train=is_train,
                                        attention_type=attention_type)

        if is_train:
            lpips_model_instance.restore_checkpoint(lpips_model_instance.model_path,
                                                    device=torch.device("cuda" if use_gpu else "cpu"))

            lpips_model_instance.eval()

    print("Initialized LPIPS instance.")


def compute_lpips(img1, img2, **kwargs):
    global lpips_model_instance

    data_range = kwargs.pop("data_range", 1.0)

    img1_t = util.im2tensor(ensure3d(img1), factor=data_range / 2.)
    img2_t = util.im2tensor(ensure3d(img2), factor=data_range / 2.)

    # Compute distance
    return 1 - float(lpips_model_instance.forward(img1_t, img2_t)[0])
