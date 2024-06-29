C++ = g++
mpi++ = mpic++
opts = -ggdb -o
flags = `pkg-config opencv4 --cflags --libs`
run = mpirun
run_opts = -n
ps = 2

mpi_exec = mpi_kmeans
mpi_code = mpi_kmeans.cpp

rng_exec = rng
rng_code = rng.cpp
rng_f = rng.txt

result = paviakmeanscpp.jpg

run: compile
	./$(rng_exec)
	$(run) $(run_opts) $(ps) ./$(mpi_exec)

compile: mpi_kmeans.cpp
	$(mpi++) $(mpi_code) $(opts) $(mpi_exec) $(flags)
	$(C++) $(rng_code) $(opts) $(rng_exec)

clean:
	rm -f $(mpi_exec)
	rm -f $(rng_exec)
	rm -f $(rng_f)
	rm -f $(result)