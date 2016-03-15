#!/usr/bin/env python



import os


L = []
top = '.'
for rootdir,dirnames,filenames in os.walk(top):
   # filenames = [f for f in filenames if len(f)]
    L += [os.path.join(rootdir,f) for f in filenames if f.endswith('pyc')]

L = [f for f in L if '(' not in f and ')' not in f]
for f in L:
    print f

cmd = 'rm ' + ' '.join(L)
print cmd

if len(L):
    os.system(cmd)
else:
    print 'no files'
