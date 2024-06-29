C++ = g++
opts = -ggdb -o
flags = `pkg-config opencv4 --cflags --libs`
exec = custom
file = custom_kmeans.cpp
result = paviakmeanscpp.jpg

compile: custom_kmeans.cpp
	$(C++) $(opts) $(exec) $(file) $(flags)
clean:
	rm -f $(exec)
	rm -f $(result)