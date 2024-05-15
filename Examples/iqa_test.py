from DisplayModels.display_model_simul import new_simul_params
from iqa_metrics.iqa_tool import IqaTool
from utils.image_processing.image_tools import imread

if __name__ == "__main__":

    SHOW_IMAGES = False

    ref_img_path = "biblioteca2_crop.jpg"
    img_ref = imread(ref_img_path, "./images", format_float=True)

    if SHOW_IMAGES:
        import cv2
        cv2.imshow("img_ref", img_ref)
        cv2.waitKey()

    iqa_tool = IqaTool()

    sp_ref = new_simul_params(illuminant=250)
    sp_test = new_simul_params(illuminant=1000)

    Q = iqa_tool.compute_iqa_custom(img_ref, img_ref, sim_params1=sp_ref, sim_params2=sp_test)
    print("Q", Q)

# example output:
# Q {'MSE': 3804.470277628302, 'PSNR': 19.693145950278122, 'SSIM': 0.9265990144354943, 'SSIM-py': 0.9265990144354777,
#    'MSSSIM': 0.9845426016437231, 'MSSSIM-py': 0.9846815689486442, 'TMQI': 0.8378354164925974, 'FSIM': 0.9898555632734,
#    'VSI': 0.9960809484591361, 'MDSI': 0.17226736057046393, 'HDR-VDP': 0.9876005954677863, 'LPIPS': 0.0250319689512252}