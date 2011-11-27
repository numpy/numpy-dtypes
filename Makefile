all: exact preflop-matchups.txt

CXX = g++-mp-4.5
CXXFLAGS = -Wall -std=c++0x -O3 -funroll-loops -march=core2 -fopenmp

preflop-matchups.txt:
	wget http://www.pokerstove.com/analysis/preflop-matchups.txt.gz
	gunzip preflop-matchups.txt.gz

exact.txt: exact
	./exact all | tee $@

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

%.E: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -E $^
