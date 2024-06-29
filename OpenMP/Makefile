C++ = g++
opts = -ggdb -fopenmp -o
flags = `pkg-config opencv4 --cflags --libs`

omp_exec = openmp_kmeans
omp_code = openmp_kmeans.cpp

rng_exec = rng
rng_code = rng.cpp
rng_f = rng.txt

result = paviakmeanscpp.jpg

run: compile
	./$(rng_exec)
	./$(omp_exec)

compile: openmp_kmeans.cpp
	$(C++) $(omp_code) $(opts) $(omp_exec) $(flags)
	$(C++) $(rng_code) -o $(rng_exec)

clean:
	rm -f $(omp_exec)
	rm -f $(rng_exec)
	rm -f $(rng_f)
	rm -f $(result)