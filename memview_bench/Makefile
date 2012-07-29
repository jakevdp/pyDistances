all: distmetrics.so pairwise.so

distmetrics.so : distmetrics.pyx
	python setup.py build_ext --inplace

pairwise.so : pairwise.pyx
	python setup.py build_ext --inplace

clean:
	rm *.c
	rm *.so