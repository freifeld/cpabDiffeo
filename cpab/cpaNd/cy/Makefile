PYX_FILENAME = expm

all:
	python setup.py build_ext --inplace
annotate:
	cython -a $(PYX_FILENAME).pyx  
	firefox $(PYX_FILENAME).html
clean:
	rm -rf build
	rm *.c *.so	
	rm $(PYX_FILENAME).html
