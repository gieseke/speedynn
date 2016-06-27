all: clean nearest_neighbors_feature_selection_cpu nearest_neighbors_feature_selection_gpu

nearest_neighbors_feature_selection_cpu:
	cd lib; python setup_cpu.py build; cp build/lib.*/*_cpu*.so .. ; cp *feature_selection_cpu.py ..;

nearest_neighbors_feature_selection_gpu:
	cd lib; python setup_gpu.py build; cp build/lib.*/*_gpu*.so .. ; cp *feature_selection_gpu.py ..;
clean:
	cd lib; rm -rf build; rm -f *.o; rm -f *.pyc; rm -f *.so; rm -f *feature_selection_*.py; rm -f *feature_selection_*_wrap.c
	rm -f nearest_neighbors_feature_selection_*.py
	rm -f *.so
	rm -f *.pyc
