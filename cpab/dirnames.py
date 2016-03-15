#!/usr/bin/env python
"""
Created on Tue Feb  4 11:04:35 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import os
import inspect

from of.utils import Bunch,FilesDirs
dirname_of_this_file = os.path.dirname(os.path.abspath(
                        inspect.getfile(inspect.currentframe())))
if len(dirname_of_this_file)==0:
    raise ValueError
print 'dirname_of_this_file',dirname_of_this_file
dirname = os.path.join(dirname_of_this_file,'..')
dirname = os.path.abspath(dirname)

FilesDirs.raise_if_dir_does_not_exist(dirname)

dirnames = Bunch()
dirnames.cpa = os.path.join(dirname,'cpa_files')



if __name__ == "__main__":
    for k in sorted(dirnames.keys()):
        print '{}:\n\t{}'.format(k,dirnames[k])
    
