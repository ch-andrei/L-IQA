import torch
from iqa_metrics.common import *
from iqa_metrics.vtamiq.common import get_vtamiq_default_params

from utils.image_processing.image_tools import ensure3d

from iqa_metrics.vtamiq.vtamiq import VTAMIQ
from iqa_metrics.vtamiq.preprocessing import transform as transform_pil
from iqa_metrics.patch_sampling import get_iqa_patches, PatchSampler


IQA_VTAMIQ_NAME = "VTAMIQ"


class VTAMIQInstance:
    model = None
    device = None
    patch_dim = None
    patch_count = None
    patch_num_scales = None
    patch_sampler = None
    parallel_runs = None
    norm_value = None


# singleton model to not reinitialize every time VTAMIQ is called
__vtamiq_instance = None


def init_instance_vtamiq(**kwargs):
    global __vtamiq_instance

    vtamiq_config, vtamiq_patch_dim = get_vtamiq_default_params()

    # configurable parameters
    use_gpu = kwargs.pop("use_gpu", True)
    model_path = kwargs.pop("model_path", None)  # for VTAMIQ to work properly, this needs to be provided
    patch_count = kwargs.pop("patch_count", 2048)  # this should be calibrated for input image resolution
    patch_num_scales = kwargs.pop("patch_num_scales", 5)  # this should be calibrated for input image resolution
    patch_sampler_config = kwargs.pop("patch_sampler_config", {})
    parallel_runs = kwargs.pop("parallel_runs", 1)

    # Initializing the model
    if __vtamiq_instance is None:
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        model = VTAMIQ(
            **vtamiq_config,
            vit_load_pretrained=False,  # no need to load pretrained VIT as we will load pretrained VTAMIQ
        )

        def get_model_checkpoint(filename, device):
            """Load model from a checkpoint"""

            if filename is None:
                return None

            print("Loading model parameters from '{}'".format(filename))
            with open(filename, "rb") as f:
                checkpoint_data = torch.load(f, map_location=device)

            return checkpoint_data

        def load_model(model_state_dict, model, info=""):
            try:
                model.load_state_dict(model_state_dict)
            except RuntimeError as re:
                print(re)
                print('WARNING: Continuing with partial load... {}'.format(info))
                model.load_state_dict(model_state_dict, strict=False)

        if not model_path:
            print("WARNING: VTAMIQ should not be used without pre-trained model weights.")
        else:
            checkpoint = get_model_checkpoint(model_path, device)

            # load pretrained VTAMIQ
            model_state_dict = checkpoint["model_state_dict"]
            load_model(model_state_dict, model, 'VTAMIQ')

        model.to(device, dtype=torch.float32)
        model.eval()
        model.share_memory()  # for multiprocessing

        # initialize all instance params
        instance = VTAMIQInstance()
        instance.model = model
        instance.device = device
        instance.patch_dim = vtamiq_patch_dim
        instance.patch_count = patch_count
        instance.patch_num_scales = patch_num_scales
        instance.patch_sampler = PatchSampler(**patch_sampler_config)
        instance.parallel_runs = parallel_runs

        # save to singleton
        __vtamiq_instance = instance

    print("Initialized VTAMIQ instance.")


def split_per_image(x, has_batch_dim=True, clone=True):
    # Note 1: clone may be needed to ensure .view() compatibility issues
    # Note 2: batch size B is optional; for K input images, x may be
    # x.shape = (B, K, N, C, P, P) -> len = 6
    # x.shape = (K, N, C, P, P) -> len = 5
    x_shape = x.shape
    num_images = x_shape[1] if has_batch_dim else x_shape[0]
    x_i = lambda i: x[:, i] if has_batch_dim else x[i].unsqueeze(0)
    x_per_image = tuple((x_i(i).clone() if clone else x_i(i)) for i in range(num_images))
    return x_per_image


def prepare_for_vtamiq(tensor, device):
    return split_per_image(tensor.to(device, dtype=torch.float32))


def get_vtamiq_inputs(img1, img2, data_range, vtamiq_instance):
    """
    Samples and returns patch-wise inputs from img1 and img2 to be used as input for VTAMIQ.
    Assumes VTAMIQ was trained with PU-encoded values normalized to [0, 1] and then to [-1, 1] during transform.
    :param img1:
    :param img2:
    :param data_range:
    :param vtamiq_instance:
    :return:
    """
    # normalize and convert cv2 images to PIL-like image format; flip from cv2 BGR to PIL-like RGB
    pil_imgs = [
        np.flip(ensure3d(normalize(img1, data_range)), axis=-1).copy(),
        np.flip(ensure3d(normalize(img2, data_range)), axis=-1).copy()
    ]

    # convert to tensors
    tensors = [transform_pil(img, None, False, False, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) for img in pil_imgs]

    # get N={parallel_runs} independent sets of sampled patches
    patches, pos, scales = [], [], []
    for i in range(vtamiq_instance.parallel_runs):
        # sample patches, positions, scales
        _patches, _ppos, _pscale = get_iqa_patches(
            pil_imgs,
            tensors,
            vtamiq_instance.patch_count,
            vtamiq_instance.patch_dim,
            vtamiq_instance.patch_sampler,
            vtamiq_instance.patch_num_scales
        )

        patches.append(_patches)
        pos.append(_ppos)
        scales.append(_pscale)

    # stack over batch dimension
    patches = torch.stack(patches)
    pos = torch.stack(pos)
    scales = torch.stack(scales)

    # split as tuples (ref, dist) and move to correct device
    patches = prepare_for_vtamiq(patches, vtamiq_instance.device)
    pos = prepare_for_vtamiq(pos, vtamiq_instance.device)
    scales = prepare_for_vtamiq(scales, vtamiq_instance.device)

    return patches, pos, scales


def compute_vtamiq(img1, img2, **kwargs):
    data_range, data_format = kwargs_get_data_params(**kwargs)

    if data_format != DATA_FORMAT_PU:
        raise ValueError("VTAMIQ requires PU-encoded inputs.")

    patches, pos, scales = get_vtamiq_inputs(img1, img2, data_range, __vtamiq_instance)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            q = __vtamiq_instance.model(patches, pos, scales)
            q = torch.mean(q, dim=0)  # average over batch dimension (parallel runs)

    return float(q.cpu())
