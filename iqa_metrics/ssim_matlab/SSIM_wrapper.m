function Q = SSIM_wrapper ( path1, path2, tag, multiscale )

img1 = load(path1);
img2 = load(path2);

if multiscale
    Q = msssim(getfield(img1, tag), getfield(img2, tag));
else
    Q = SSIM_index(getfield(img1, tag), getfield(img2, tag));
end
