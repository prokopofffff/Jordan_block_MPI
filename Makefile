OPTS = -std=c++11 -O3 -fstack-protector-all -g -W -Wall -Wextra -Wunused -Wcast-align -pedantic -pedantic-errors -Wfloat-equal -Wpointer-arith -Wformat-security -Wmissing-format-attribute -Wformat=2 -Wwrite-strings -Wcast-align -Wno-long-long -Woverloaded-virtual -Wnon-virtual-dtor -Wcast-qual -Wno-property-attribute-mismatch
all: clear a.out
a.out: main.o funcs.o jordan.o matrix.o
	mpicxx $(OPTS) $^ -o a.out
main.o: main.cpp funcs.h jordan.h matrix.h
	mpicxx -c $(OPTS) $<
funcs.o: funcs.cpp funcs.h
	mpicxx -c $(OPTS) $<
jordan.o: jordan.cpp jordan.h
	mpicxx -c $(OPTS) $<
matrix.o: matrix.cpp matrix.h
	mpicxx -c $(OPTS) $<
clear:
	rm -f *.o
clean:
	rm -f *.out *.o *.bak
