# cpabDiffeo
CPAB transformations are simple, fast, and highly-expressive finite-dimensional diffeomorphisms. They are derived from parametric, continuously-defined, velocity fields.

This implementation is based on our Journal paper, 
[\[Freifeld et al., TPAMI '17\]](https://www.cs.bgu.ac.il/~orenfr/papers/freifeld_etal_PAMI_2017)
and its earlier conference version 
[\[Freifeld et al., ICCV '15\]](http://people.csail.mit.edu/freifeld/publications.htm).

The current implementation is written in **Python**+**CUDA**. 
For a **Tensorflow/PyTorch** implementation (as well as pure NumPy version), see [libcpab](https://github.com/SkafteNicki/libcpab).

You may also want to try a [partial implementation in Julia](https://github.com/angel8yu/cpab-diffeo-julia) written by Angel Yu. 

## Author of this software

Oren Freifeld (email: freifeld@csail.mit.edu)

## License

This software is released under the MIT License (included with the software). Note, however, if you use this code (and/or the results of running it) to support any form of publication (e.g.,a book, a journal paper, a conference paper, a patent application, etc.), then we ask you to cite the following papers:

```
@article{freifeld2017transformations,
  title={Transformations Based on Continuous Piecewise-Affine Velocity Fields},
  author={Freifeld, Oren and Hauberg, Soren and Batmanghelich, Kayhan and Fisher, John W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}
@inproceedings{freifeld2015transform,
    title = {Highly-Expressive Spaces of Well-Behaved Transformations: Keeping It Simple},
    author = {Oren Freifeld and S{\o}ren Hauberg and Kayhan Batmanghelich and John W. Fisher III},
    booktitle = {International Conference on Computer Vision (ICCV)},
    address = {Santiago, Chile},
    month = Dec,
    year = {2015}
}
```

## Versions
- 03/28/2016, Version 0.0.5  -- Basic support for dim > 3; monotonic regression (in 1D)
- 03/24/2016, Version 0.0.4  -- Simple Landmark-based inference in 2D.
- 03/21/2016, Version 0.0.3  -- Synthesis in 3D.
- 03/16/2016, Version 0.0.2  -- 1) synthesis in 1D; 2) simple 2D image registration (intensity-based Gaussian likelihood + MCMC inference)
- 03/15/2016, Version 0.0.1  -- First release (synthsis in 2D).

## Coming soon: 
More options for inference, more options for the prior, handling landmarks and registration in all dimensions, more applications, etc. 

## Requirements
- generic python packages: numpy; scipy; matplotlib
- opencv with python's bindings
- CUDA
- pycuda
- My **of** and **pyimg** packages:
```
# Using https:
git clone https://github.com/freifeld/of
git clone https://github.com/freifeld/pyimg
# Using ssh:
git clone git@github.com:freifeld/of.git
git clone git@github.com:freifeld/pyimg.git
```
## OS
The code was tested on Linux and Windows. It should work on Mac, but I didn't get a chance to test it.

## Installation
(todo: add instructions for Windows users)
First, get this repository:
```
# Using https:
git clone https://github.com/freifeld/cpabDiffeo.git
# Using ssh:
git clone git@github.com:freifeld/cpabDiffeo.git
```
Second, assuming you cloned this repository as well the **of** and **pyimg** repositories in your home directory (marked as ~), you
will need to adjust your PYTHONPATH accordingly:
```
# To enable importing of both the "of" and "pyimg" packages which are in ~
export PYTHONPATH=$PYTHONPATH:$~    
# To enable  improting of the "cpab" package which is inside ~/cpabDiffeo
export PYTHONPATH=$PYTHONPATH:$~/cpabDiffeo/  
```

## How to run the code
We provide quick demos that show synthesis in 1d, 2d, or 3d and have several possible configurations that the user can modify. To run the demos, first neviagate into the cpab directory. Then:
```
python cpa1d/TransformWrapper_example.py  # 1d 
python cpa2d/TransformWrapper_example.py  # 2d 
python cpa3d/TransformWrapper_example.py  # 3d 

```
The **example** function in each of these scripts takes several input arguments whose values you can change. 
You can do it either directly from python, as is done in the commented-out examples at the end of each of these scripts,
or (in the 2D case) from the terminal using the following script:
```
python cpa2d/TransformWrapper_example_cmdline.py   # This will just use default parameters
```
Details about how to pass user-defined arguments using the terminal can be found [here](README_cmdline_options.md).

To run an example of **landmark-based inference**:
```
 python cpa2d/apps/landmarks/inference_example_LFW.py
```
To visualize the inference results:
```
python cpa2d/apps/landmarks/visualize_results_LFW.py
```

TODO: add instructions for how to run the example for image registration.

To run a **monotonic-regression** example:
```
python cpa1d/inference/transformation/MonotonicRegression_example.py
```
