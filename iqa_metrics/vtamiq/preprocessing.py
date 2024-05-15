import torch
import torchvision.transforms.functional as functional


def transform(img,
              crop_params=None,
              h_flip=False,
              v_flip=False,
              norm_mean=None,
              norm_std=None,
              ):
    """
    transforms a PIL image to tensor while applying several data augmentation operations
    :param img: PIL image
    :param crop_params:
        i, j, h, w: top left corner coord [i, j] and [height, width] to extract
    :param h_flip:
    :param v_flip:
        bool, toggles for horizontal/vertical flip
    :param norm_mean:
    :param norm_std:
        None if normalization disabled; else should have 3 channels each
    :return:
    """

    # Transform to tensor
    tensor = functional.to_tensor(img)  # 3xHxW

    # crop to output dimension
    if crop_params is not None:
        i, j, h, w = crop_params
        tensor = functional.crop(tensor, i, j, h, w)

    # random flips
    if h_flip:
        tensor = functional.hflip(tensor)
    if v_flip:
        tensor = functional.vflip(tensor)

    # normalize
    if norm_mean is not None and norm_std is not None:
        tensor = functional.normalize(tensor, norm_mean, norm_std)

    return tensor


def get_transform_params(img1, img2,
                         patch_sampler,
                         patch_dim=None,
                         h_flip=None,
                         v_flip=None,
                         ):
    """
    :param img1:
    :param img2:
        PIL images
    :param patch_sampler:
    :param patch_dim:
        None (no resizing) or tuple holding image dimensions h x w for resizing the image
    :return:
    """
    if patch_dim is None:
        crop_params = None
    else:
        crop_params = patch_sampler.get_sample_params(img1, img2, patch_dim[0], patch_dim[1])

    rsamples = torch.rand(2)

    if h_flip is None:
        h_flip = rsamples[0] <= 0.5

    if v_flip is None:
        v_flip = rsamples[1] <= 0.5

    return crop_params, h_flip, v_flip


def transform_img(img,
                  patch_sampler,
                  patch_dim=None,
                  norm_mean=None,
                  norm_std=None,
                  ):
    crop_params, h_flip, v_flip = get_transform_params(img, img, patch_sampler, patch_dim)
    tensor = transform(img, crop_params, h_flip, norm_mean, norm_std)
    return tensor


def transform_img_pair(img1, img2,
                       patch_sampler,
                       patch_dim=None,
                       norm_mean=None,
                       norm_std=None
                       ):
    crop_params, h_flip, v_flip = get_transform_params(img1, img2, patch_sampler, patch_dim)
    tensor1 = transform(img1, crop_params, h_flip, v_flip, norm_mean, norm_std)
    tensor2 = transform(img2, crop_params, h_flip, v_flip, norm_mean, norm_std)
    return tensor1, tensor2
