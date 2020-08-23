function Q = tmqi_wrapper ( path1, path2, tag )

img1 = load(path1);
img2 = load(path2);

Q = TMQI(getfield(img1, tag), getfield(img2, tag));
