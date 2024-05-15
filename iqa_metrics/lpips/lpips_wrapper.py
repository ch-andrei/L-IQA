from iqa_metrics.lpips.PerceptualSimilarity.util import util
from iqa_metrics.lpips.PerceptualSimilarity.models import dist_model as dm

from utils.image_processing.image_tools import ensure3d
from iqa_metrics.common import *

import torch

# singleton model to not reinitialize every time LPIPS is called
__lpips_model_instance = None


def init_instance_lpips(**kwargs):
    use_gpu = kwargs.pop("use_gpu", False)
    model = kwargs.pop("model", "net-lin")
    net = kwargs.pop("net", "alex")
    attention_type = kwargs.pop("attention_type", None)

    global __lpips_model_instance

    # Initializing the model
    if __lpips_model_instance is None:
        # if not default LPIPS, need to be in train mode so that full model structure is initialized with weights
        __lpips_model_instance = dm.DistModel()
        __lpips_model_instance.initialize(use_gpu=use_gpu,
                                          model=model,
                                          net=net,
                                          is_train=False)

        __lpips_model_instance.eval()
        __lpips_model_instance.share_memory()

    print("Initialized LPIPS instance.")


def compute_lpips(img1, img2, **kwargs):
    global __lpips_model_instance

    # always rescale to [0, 1] regardless of data format
    data_range, data_format = kwargs_get_data_params(**kwargs)
    img1 = normalize(img1, data_range)
    img2 = normalize(img2, data_range)

    # LPIPS supported tensor [-1, 1]
    img1_t = util.im2tensor(ensure3d(img1), factor=1.0 / 2.0)
    img2_t = util.im2tensor(ensure3d(img2), factor=1.0 / 2.0)

    # Compute distance
    return float(__lpips_model_instance.forward(img1_t, img2_t)[0])
