all: distmetrics.so brute_neighbors.so ball_tree.so

distmetrics.so : distmetrics.pyx distmetrics.pxd distfuncs.pxi
	touch distmetrics.pyx
	python setup.py build_ext --inplace

distfuncs.pxi : distfuncs.pxi_src
	python _parse_src.py

brute_neighbors.so : brute_neighbors.pyx
	python setup.py build_ext --inplace

ball_tree.so : ball_tree.pyx
	python setup.py build_ext --inplace

clean:
	rm *.c
	rm *.so