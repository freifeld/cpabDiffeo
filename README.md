# cpabDiffeo
Finite-dimensional spaces of simple, fast, and highly-expressive diffeomorphisms, self-coined CPAB transformations, derived from parametric, continuously-defined, velocity fields.

This Python+CUDA implementation is based on our paper, [\[Freifeld et al., ICCV '15\] ](http://people.csail.mit.edu/freifeld/publications.htm), but also contains some extensions and variants of that work that were not included in the ICCV paper due to page limits. 

For example, while the ICCV paper discusses only $R^n$ for n=1,2,3, the implementation here also supports higher values of $n$ (as to be expected, both the dimensionality of the representation and integration computing time increase with $n$ and thus values of $n$ that are too high will be impractical in terms of memory, inference, running time, etc.).
It also contains additional types of tessellations and bases. There are pros and cons for each choice.

**In March 2016 we released a [preprint](http://people.csail.mit.edu/freifeld/papers/freifeld_CPAB_preprint_2016.pdf) (that
extends our ICCV paper) which covers these options.** The supplemental material for this preprint is available [here](http://people.csail.mit.edu/freifeld/papers/freifeld_CPAB_preprint_2016_supmat.pdf).

Finally, you may also want to try a [partial implementation in Julia](https://github.com/angel8yu/cpab-diffeo-julia) written by my student, Angel Yu. Note, however, that Angel's CPU-based implementation has fewer options than the one I will maintain here (e.g, it is only in 1D or 2D, has less options for the prior, doesn't have image/signal registration, etc.)

## Author of this software

Oren Freifeld (email: freifeld@csail.mit.edu)

## License

This software is released under the MIT License (included with the software). Note, however, that using this code (and/or the results of running it) to support any form of publication (e.g.,a book, a journal paper, a conference paper, a patent application, etc.) requires you to cite the following paper:

```
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
- 03/24/2016, Version 0.0.4  -- Simple Landmark-based inference in 2D.
- 03/21/2016, Version 0.0.3  -- Synthesis in 3D.
- 03/16/2016, Version 0.0.2  -- 1) synthesis in 1D; 2) simple 2D image regiration (intensity-based Gaussian likelihood + MCMC inference)
- 03/15/2016, Version 0.0.1  -- First release (synthsis in 2D).

Coming soon: More options (dim>3, more options for inference, more options for the prior, handling landmarks, more applications, etc). 

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
The code was tested on Linux and Windows. I believe it should work on Mac, but didn't get a chance to test it.

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
That's it. You should be good to go.
## How to run the code
For now, these are just quick demos that show synthesis in 1d, 2d, or 3d and have several possible configurations that the user can modify. To run the demos, first neviagate into the cpab directory. Then:
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
