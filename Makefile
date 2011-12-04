all: exact rational.so

CXX = g++
CXXFLAGS = -Wall -Werror -O2 -framework OpenCL -fopenmp

exact.txt: exact score.cl
	time ./exact all > $@

exact: exact.cpp score.h
	$(CXX) $(CXXFLAGS) -o $@ $<

%.E: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -E $^

rational.so: rational.cpp
	python setup.py build
	cp build/lib.*/rational.so .

.PHONY: clean test

test: exact score.cl
	time ./exact test

clean:
	rm -f exact *.o *.E *.so
