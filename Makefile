all: rational.so

rational.so: rational.cpp
	python setup.py build
	cp build/lib.*/rational.so .

.PHONY: clean test

clean:
	rm -f *.o *.E *.so
