function Q = fsim_wrapper ( path1, path2, tag )

img1 = load(path1);
img2 = load(path2);

[FSIM, FSIMc] = FeatureSIM(getfield(img1, tag), getfield(img2, tag));

Q = FSIMc
