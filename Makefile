all: exact

CXX = g++
CXXFLAGS = -Wall -Werror -O2 -framework OpenCL -fopenmp

exact.txt: exact score.cl
	time ./exact all > $@

exact: exact.cpp score.h
	$(CXX) $(CXXFLAGS) -o $@ $<

%.E: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -E $^

.PHONY: clean
clean:
	rm -f exact *.o *.E
