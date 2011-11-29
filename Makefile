all: exact preflop-matchups.txt

CXX = g++
CXXFLAGS = -Wall -Werror -O2 -framework OpenCL -fopenmp

preflop-matchups.txt:
	wget http://www.pokerstove.com/analysis/preflop-matchups.txt.gz
	gunzip preflop-matchups.txt.gz

exact.txt: exact score.cl
	time ./exact all > $@

exact: exact.cpp score.h
	$(CXX) $(CXXFLAGS) -o $@ $<

%.E: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -E $^

.PHONY: clean
clean:
	rm -f exact *.o *.E
