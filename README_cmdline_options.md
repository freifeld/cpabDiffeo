For help, run:
```
python cpa2d/TransformWrapper_example_cmdline.py -h
```
Below are examples for how to change the input arguments and observe the associated effects. You can also combine more than one option at the time (but see remark below -- TODO). Note that at the first time you run a given configuration, the program will first need to encode the continuity constraints and extract the associated global basis (see the paper for details; we will soon add an option for using the local basis -- whose construction is much faster -- instead ; each choice has pros and cons). If the number of cells is large, this may take some time. For a given configuration, however, this is done only once; the results computed here will be saved and reused the next time you use the same configuration.

- The default tessellation type is set to 'I' (triangles in 2D). That was the only tessellation mentioned in the ICCV '15 paper. To change it to type 'II' (rectangles in 2D), run:
```
python cpa2d/TransformWrapper_example_cmdline.py --tess=II   # or -t II for a shorter notation
```
- To change the number of levels in the multiscale representation to, say, 2, run:
```
python cpa2d/TransformWrapper_example_cmdline.py --nLevels=2 # or -nl 2 
```
- By default, the base (i.e., coarsest) level uses single rectangle (if tess=II), possibly divided into 4 triangles (if tess=I). In effect, the number of rows and columns are both 1. To change it to, say, 2 rows and 3 columns, run:
```
python cpa2d/TransformWrapper_example_cmdline.py --base 2 3 # or -b 2 3 
```
- To sepcify your own image of choice, run
```
python cpa2d/TransformWrapper_example_cmdline.py --img=<image_filename> # or -i <image_filename> 
```
- To enforce volume preservation (area preservation in 2D), run:
```
python cpa2d/TransformWrapper_example_cmdline.py --vol_preserve 1 # or use -vp 1
```
- To enforce zero velocity across the boundary, you need to use the option
```
--zero_v_across_bdry 1 1 # or use -zbdry 1 1 
# The 1st 1 is for the horizontal boundary, the 2nd is for the vertical boundary; 
# mixed boundary types are not supported at the moment. 
```
In this case, **however**, if you use tess=I (which is the default) you will also need to set valid_outside to 0 (by default it is set to 1) while if you use tess=II you will need to use more than a single cell (which is the default) as the base level (since the only affine velocity on a single cell which is zero at the boundary is the one which is zero everywhere), e.g., you can set it to a 2x2 grid. Here is how these calls look like:
```
python cpa2d/TransformWrapper_example_cmdline.py --zero_v_across_bdry 1 1 --valid_outside 0
python cpa2d/TransformWrapper_example_cmdline.py --tess=II --zero_v_across_bdry 1 1 --base 2 2
```
