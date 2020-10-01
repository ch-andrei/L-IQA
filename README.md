# Perceptual Image Quality Assessment for Various Viewing Conditions and Display Systems

## Contents

This repository contains a Python implementation of our perceptual IQA framework 
capable of performing image quality assessment for various non-standard illumination 
conditions and display systems. An IQA tool allowing the user to conveniently call various 
full-reference IQA metrics with our framework is provided.

If you find this useful for your research, kindly cite the following:

<i> Perceptual Image Quality Assessment for Various Viewing Conditions and Display Systems
Andrei Chubarau, Tara Akhavan, Hyunjin Yoo, Rafał K. Mantiuk and James Clark.
Human Vision and Electronic Imaging, pp. 067-1-067-9(9), 2020, 
(<a href="http://dx.doi.org/10.2352/ISSN.2470-1173.2020.9.IQSP-067">doi</a>)
(<a href="https://www.cl.cam.ac.uk/~rkm38/pdfs/chubarau2020iqm_view_conditions.pdf">Paper PDF</a>)
</i>

## Paper abstract:

From complete darkness to direct sunlight, real-world displays operate in various viewing conditions 
often resulting in a non-optimal viewing experience. Most existing Image Quality Assessment (IQA) methods, 
however, assume ideal environments and displays, and thus cannot be used when viewing conditions differ 
from the standard. In this paper, we investigate the influence of ambient illumination level and display luminance 
on human perception of image quality. We conduct a psychophysical study to collect a novel dataset of over 
10000 image quality preference judgments performed in illumination conditions ranging from 0 lux to 20000 lux. 
We also propose a perceptual IQA framework that allows most existing image quality metrics (IQM) to accurately 
predict image quality for a wide range of illumination conditions and display parameters. Our analysis demonstrates 
strong correlation between human IQA and the predictions of our proposed framework combined with multiple prominent 
IQMs and across a wide range of luminance values.

## Perceptual IQA Framework

<img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/readme_images/iqa_framework.PNG" alt="framework" width=750>

We estimate the visual signal that reaches the observer’s eye given the 
information about the environment and the used display system, 
and apply existing IQA metrics on the resulting stimuli. 
A Display and Degradation Model (DDM) converts digital inputs from 
gamma-corrected pixel values to the physical luminance space and simulates 
the influence of the ambient illumination level. The resulting signals
are then linearized with perceptually uniform (PU) encoding [1] to account
for luminance masking and, finally, IQA is computed.

## Examples and use-cases

Note that these scripts must be run using /L-IQA as the current working directory. 

./Examples folder contains several potential use-cases of the provided tools.

1.  Perform IQA for non-standard illumination conditions and displays. Most IQA metrics assume standard 200-500 
    lux "indoors" lighting but real-world imagery can be viewed in any conditions from nighttime to direct sunlight. 
    
    ./Examples/iqa_tool_test.py 

2.  Simulate what an observer will see (how a display will appear to the observer) given display parameters 
    and ambient viewing conditions. Some examples of this include comparing i) full brightness vs dimmed displays, ii) display with 
    weak/strong (non-)uniform reflection, etc.
    
    ./Examples/structural_reflection_test.py 

    Left to right: input image, illumination map, simulated image assuming 5000 lux illumination.

    <img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/images/reflection/2/merged.jpg" alt="framework" width=750>


3. Run multiple full reference IQA metrics

DDM and IQA tool are components; one can be used with or without the other.

## Display and Degradation Model (DDM)

DDM is used to simulate the visual signal that reaches the obsever's eyes and its implementation can be found in "./DisplayModels/".

We simulate the response of the display including the influence of ambient illumination conditions using the following equation:

<img src="https://render.githubusercontent.com/render/math?math=L(V) = (L_{max} - L_{blk}) V ^ \gamma + L_{blk} + L_{refl}">

<br/>

    Lmax = maximum display luminance in cd/m2, 
    Lblk = display’s black level luminance in cd/m2, 
    V = input signal luma in the range 0-1
    Lrefl = reflected luminance approximated as


<img src="https://render.githubusercontent.com/render/math?math=L_{refl} = \frac{k}{\pi} E_{amb}">

<br/>

    k = reflectivity of the display (usually 0.01-0.05)
    Eamb is the ambient illumination level in lux

At runtime, DDM simulation parameters are specified via:

```python
iqa_simul_params = namedtuple('iqa_sim_params',
                              ['illuminant',
                               'illumination_map',
                               'illumination_map_weight_mode',
                               'use_luminance_only',
                               'apply_reflection',
                               'apply_screen_dimming',
                               ])

```

    illuminant = ambient conditions in lux
    illumination_map = reflection/illumimation map (can be i) None, ii) a luminance image, iii) RGB image)
    illumination_map_weight_mode = controls what is used for computing maximum reflection intensity, can be 'mean' or 'max' 
    use_luminance_only = toogle for only using luminance channel (must be True if doing IQA, more on this below)
    apply_reflection = toggle for applying reflection (when False, Lrefl = 0)
    apply_screen_dimming = toggle for adaptive display luminance (display luminance depends on ambient illumination)   

Additionally, a function new_simul_params() is provided to initialize iqa_simul_params:

```python
def new_simul_params(illuminant=None,
                     illumination_map=None,
                     illumination_map_weight_mode='mean',
                     use_luminance_only=True,
                     apply_reflection=True,
                     apply_screen_dimming=True
                     ):
```

##### Note on "use_luminance=True"

This should be set to True when performing IQA and in most other cases. 
Since the intended usage of this tool is to simulate the visual (photometric) 
signal that reaches the observers eyes in units cd/m2, simulated inputs to image quality metrics are 
single-channel luminance values and not the original RGB content. This is one of the restrictions of our framework, 
but it is required since PU encoding assumes Luminance inputs (and without PU encoding, the framework's predictions do 
not correlate with human IQA).

Setting "use_luminance=False" can lead to unexpected results.

## Simulating image appearance in particular conditions

You can use the DDM to simulate (approximately) what an image will appear like under particular
illumination conditions and display parameters. This can be used for visualizing reflections
on a display, to compare different display parameters, etc.

Illumination (and consequently reflection) can either be uniform or structured. Uniform reflection
is applied in equal amount to all regions of the display/image. 

Structural (non-uniform) reflection is specified via an illumination map, which can be either a single channel 
2D image with values 0-1 controlling how much illumination is applied at each pixel of the image or a 3 channel (RGB)
image further controlling the color of the reflection (if using RGB reflection, use_luminance must be set to False, 
to use color information in the simulation, else RGB will be collapsed to lum only). 

Note that the overall illumination level is still controlled via a single illuminance input, 
i.e. 1000 lux; the illumination map is then linearly scaled with the illuminance input set as its average 
(not maximum) value.

Below are some examples of simulated reflections given this method:

Illumination map:

<img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/images/reflection/1/illumination_map.jpg" alt="framework" width=250>

Input image:

<img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/images/reflection/1/reference.jpg" alt="framework" width=250>

Simulated assuming 1000 lux illumination:

<img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/images/reflection/1/test-s-1000.jpg" alt="framework" width=250>

Simulated assuming 5000 lux illumination:

<img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/images/reflection/1/test-s-5000.jpg" alt="framework" width=250>

Simulated assuming 20000 lux illumination:

<img src="https://raw.githubusercontent.com/ch-andrei/L-IQA/master/images/reflection/1/test-s-20000.jpg" alt="framework" width=250>

Note that the illuminant (in lux) must be measured via a tool such as a photometer for accurate results.
Illumination map thus only provides the structural information for the illumination/reflection and not its magnitude. 

## Using IQA metrics

We tested our IQA framework with multiple prominent image quality metrics and observed strong correlation between 
human IQA and our framework's predictions across a wide range of illumintion conditions (refer to the paper for 
detailed results). 

See ./iqa_metrics/ folder for available IQA implementations. 

## Adding your IQA metrics:

We interface between existing IQA metrics and our IQA tool via a Python wrapper.
The wrapper must implement an initialization function (called at startup) and a compute function (called when IQA 
computations are required) for each metric. 

To use a new metric with this IqaTool, add the metric's code to /iqa_metrics directory 
and add a Python wrapper with appropriate functions (see examples in ./iqa_metrics/ for implemented wrappers).

A metric is defined as follows:

    iqa_metric_variant = namedtuple('iqa_metric_variant', ['name', 'compute_function', 'init_function', 'init_kwargs'])

And a function is provided for convenient initialization:

    def new_iqa_variant(name, compute_function, init_function=None, init_kwargs=None):
        """
        use this function to initialize new IQA metrics (returns a iqa_metric_variant)
        :param name:
        :param compute_function: function to compute quality metric value Q between img1 and img2
        :param init_function: initialization function (will be called once before any calls to compute function
        :param init_kwargs: custom initialization parameters
        :return: iqa_metric_variant namedtuple
        """
        return iqa_metric_variant(name,
                                  compute_function,
                                  init_function,
                                  init_kwargs={} if init_kwargs is None else init_kwargs
                                  )

**compute_function** must adhere to input format: compute_function(img1, img2, **kwargs) where kwargs are provided at runtime.

As an example, we use the simplistic Mean-Squared Error metric:

    def compute_mse(img1, img2, **kwargs):
        return np.mean((img1 - img2) * (img1 - img2))

    iqa_mse = new_iqa_variant('MSE', compute_mse)

**init_function** will be called at initialization of IqaTool as init_function(**init_kwargs) where init_kwargs are 
provided at iqa_metric_variant declaration. 

Example with kwargs:

    iqa_hdr_vdp = new_iqa_variant('HDR-VDP', compute_hdr_vdp, init_instance_hdr_vdp, {'hdrvdp_version': "3.0.6"})

##### Note for MATLAB metrics:

Code was tested with MATLAB 19/20, the MATLAB Engine API for Python is required.

See existing examples of MATLAB metrics in ./iqa_metrics/.

Note that most of the IQA metrics have their own dependencies which must be satisfied.

To make our IQA tool work with MATLAB, we call the metric's MATLAB script from Python. Sending image data 
between Python and MATLAB processes is very slow; as a workaround, we save inputs images to two temporary .mat files, 
and have an additional MATLAB script read these files and call the appropriate MATLAB IQA code (speedup about 10-100x).

## Relevant links and dependencies:

Requires OpenCV-python (cv2), numpy, scipy.

TMQI: https://github.com/dvolgyes/TMQI

HDRVDP: http://hdrvdp.sourceforge.net/wiki/#Download
Requires MATLAB with Image Processing Toolbox and Statistics and Machine Learning Toolbox. 

LPIPS: https://github.com/richzhang/PerceptualSimilarity
requires PyTorch, see dependencies in LPIPS project page

## References:

[1] T. O. Aydın, R. K. Mantiuk, and H.-P. Seidel. Extending quality metrics
to full luminance range images. In Human Vision and Electronic
Imaging, pages 68060B–10. Spie, 2008.

