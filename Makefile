all: exact preflop-matchups.txt

CXX = g++-mp-4.5
CXXFLAGS = -Wall -Wunused -std=c++0x -O3 -funroll-loops -march=core2 -fopenmp -framework OpenCL

preflop-matchups.txt:
	wget http://www.pokerstove.com/analysis/preflop-matchups.txt.gz
	gunzip preflop-matchups.txt.gz

exact.txt: exact
	./exact all | tee $@

exact: exact.cpp exact.cl
	$(CXX) $(CXXFLAGS) -o $@ $<

%.E: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -E $^

.PHONY: clean
clean:
	rm -f exact *.o *.E
