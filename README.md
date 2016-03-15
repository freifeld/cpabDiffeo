# cpabDiffeo
Finite-dimensional spaces of simple, fast, and highly-expressive diffeomorphisms, self-coined CPAB transformations, derived from parametric, continuously-defined, velocity fields.

I have just uploaded a partial version of the code and will keep updating it during March 2016.

You may also want to try a [partial implementation in Julia](https://github.com/angel8yu/cpab-diffeo-julia) written by my student, Angel Yu. That CPU-based implementation has far less options than the one I will post here (e.g, it is only in 1D or 2D, has less options for the prior, doesn't have image/signal registration, etc.)

This implementation is based on our recent paper, [\[Freifeld et al., ICCV '15\] ](http://people.csail.mit.edu/freifeld/publications.htm), but also contains some extensions and variants of that work that were not included in the ICCV paper due to page limits. 

For example, while the ICCV paper discusses only $R^n$ for n=1,2,3, the implementation here also supports higher values of $n$ (as to be expected, both the dimensionality of the representation and integration computing time increase with $n$ and thus values of $n$ that are too high will be impractical in terms of memory, inference, running time, etc.).
It also contains additional types of tessellations and bases. There are pros and cons for each choice.

**During Spring 2016 we will release an extended TR that will cover these options.**


# Requirements
- opencv with python's bindings
- pycuda
- My "of" and "pyimg" packages:
```
# To get them using https:
git clone https://github.com/freifeld/of
git clone https://github.com/freifeld/pyimg
# To get them using ssh:
git clone git@github.com:freifeld/of.git
git clone git@github.com:freifeld/pyimg.git
```
# OS
The code was tested on Linux and Windows. I believe it should work on Mac, but didn't get a chance to test it.

# Installation
(todo: add instructions for Windows users)
Assuming you cloned this repository as well the "of" and "pyimg" repo into your home directory (marked as ~), you
will need to adjust your PYTHONPATH accordingly:
```
# To enable you import both the of and pyimg packages which are in ~
export PYTHONPATH=$PYTHONPATH:$~    
# To enable you import the cpab package which is inside ~/cpabDiffeo
export PYTHONPATH=$PYTHONPATH:$~/cpabDiffeo/  
```
# How to run the code
For now, this is just quick demo that shows synthesis in 2d and has several possible configurations. I will soon upload more options (other dimensions, inference, etc.). More details coming soon.

To run the demo, first neviagate into the cpab directory. Then:
```
python cpa2d/TransformWrapper_example_usage_cmdline.py 
```







