function Q = MDSI_wrapper ( path1, path2, tag, use_grayscale)

img1 = load(path1);
img2 = load(path2);

Q = MDSI(getfield(img1, tag), getfield(img2, tag), use_grayscale);
