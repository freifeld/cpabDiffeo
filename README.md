# cpabDiffeo
Finite-dimensional spaces of simple, fast, and highly-expressive diffeomorphisms, self-coined CPAB transformations, derived from parametric, continuously-defined, velocity fields.

This Python+CUDA implementation is based on our paper, [\[Freifeld et al., ICCV '15\] ](http://people.csail.mit.edu/freifeld/publications.htm), but also contains some extensions and variants of that work that were not included in the ICCV paper due to page limits. 

For example, while the ICCV paper discusses only $R^n$ for n=1,2,3, the implementation here also supports higher values of $n$ (as to be expected, both the dimensionality of the representation and integration computing time increase with $n$ and thus values of $n$ that are too high will be impractical in terms of memory, inference, running time, etc.).
It also contains additional types of tessellations and bases. There are pros and cons for each choice.

**During Spring 2016 we will release an extended TR that will cover these options.**

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
- 03/21/2016, Version 0.0.3  -- Synthesis in 3D.
- 03/16/2016, Version 0.0.2  -- 1) synthesis in 1D; 2) simple 2D image regiration (intensity-based Gaussian likelihood + MCMC inference)
- 03/15/2016, Version 0.0.1  -- First release (synthsis in 2D).

Coming soon: More options (dim>3, more options for inference, more options for the prior, handling landmarks, more applications, etc). 

## Requirements
- generic python packages: numpy; scipy; matplotlib
- opencv with python's bindings
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
# To enable you import both the of and pyimg packages which are in ~
export PYTHONPATH=$PYTHONPATH:$~    
# To enable you import the cpab package which is inside ~/cpabDiffeo
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
The **example** function in either of these files takes several input arguments whose values you can change. 
You can do it either directly from python, e.g.  (these commented-out examples are taken from the end of **cpa2d/TransformWrapper_example.py**):
```
#    Here are some other options you may want to try.
#    You can also try to combine these options, but note
#    that few of these combinations are invalid -- in which case 
#    an Exception will be thrown.
#    tw = example(tess='II') # OK
#    tw = example(nLevels=2) # OK
#    tw = example(base=[2,3]) # OK
#    tw = example(vol_preserve=True) # OK
#    tw = example(zero_v_across_bdry=[1,1],valid_outside=True) # Will fail (as it should)
#    tw = example(zero_v_across_bdry=[1,1]) # Will also fail, since valid_outside defaults to True
#    tw = example(zero_v_across_bdry=[1,1],valid_outside=False)  # OK
#    tw = example(tess='II',zero_v_across_bdry=[1,1]) # Will fail (as it should) 
                                                      # as there are too many constraints
                                                      # The problem is that base=[1,1]
                                                      # means we have only one cell, 
                                                      # so with the added boundary constraints. 
                                                      # there are no degrees of freedom.
#    tw = example(tess='II',zero_v_across_bdry=[1,1],base=[1,2]) # OK
#    tw = example(tess='II',zero_v_across_bdry=[1,1],base=[2,2]) # OK
#    tw =example(zero_v_across_bdry=[1,1],valid_outside=False,vol_preserve=True) # Will fail; no DoF.
#    tw =example(zero_v_across_bdry=[1,1],valid_outside=False,vol_preserve=True,base=[1,2]) # OK
#     For the effect of scale_spatial on the prior's smoothness, compare the following two lines
#    tw = example(scale_spatial=.01,base=[4,4],nLevels=1) # OK
#    tw = example(scale_spatial=10,base=[4,4],nLevels=1) # OK
#     For the effect of scale_value on the prior's variance, compare the following two lines
#    tw = example(scale_value=100.0,base=[4,4],nLevels=1) # OK
#    tw = example(scale_value=300.0,base=[4,4],nLevels=1) # OK
```
or (in the 2D case) from the terminal using the following script:
```
python cpa2d/TransformWrapper_example_cmdline.py   # This will just use default parameters
```
More details about how to pass different arguments using the terminal option can be found in 
TBD

To run an example for landmark-based inference:
```
 python cpa2d/apps/landmarks/inference_example_LFW.py
```
To visualize the inference results:
```
python cpa2d/apps/landmarks/visualize_results_LFW.py
```





