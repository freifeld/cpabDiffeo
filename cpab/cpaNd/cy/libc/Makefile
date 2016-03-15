PYX_FILENAME = test_call_from_cy

all:
	python setup.py build_ext --inplace
annotate:
	cython -a $(PYX_FILENAME).pyx  
	firefox $(PYX_FILENAME).html
clean:
	rm -rf build
	rm test_call_from_cy.c test_call_from_cy.so	
	rm $(PYX_FILENAME).html
