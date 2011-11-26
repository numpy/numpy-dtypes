all: crunch preflop-matchups.txt

CXX = g++-mp-4.5
CXXFLAGS = -Wall -O3 -fopenmp

preflop-matchups.txt:
	wget http://www.pokerstove.com/analysis/preflop-matchups.txt.gz
	gunzip preflop-matchups.txt.gz

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

%.E: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -E $^
