function Q = hdrvdp_wrapper( path1, path2, tag, version, color_encoding, display_params )

img1 = load(path1);
img2 = load(path2);

reference = getfield(img1, tag);
test = getfield(img2, tag);

% compute pixels per degree
s = double(display_params{1});
w = double(display_params{2});
h = double(display_params{3});
d = double(display_params{4});
ppd = hdrvdp_pix_per_deg( s, [w, h], d );

if strcmp(version, '2.2.2')
    Q = hdrvdp( test, reference, color_encoding, ppd );
elseif strcmp(version, '3.0.6')
    Q = hdrvdp3( 'quality', test, reference, color_encoding, ppd, {} );
else
    error("HDR-VDP error: Incorrect code version specified");
end
