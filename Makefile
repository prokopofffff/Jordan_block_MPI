OPTS = -std=c++11 -O3 -fstack-protector-all -g -W -Wall -Wextra -Wunused -Wcast-align -Werror -pedantic -pedantic-errors -Wfloat-equal -Wpointer-arith -Wformat-security -Wmissing-format-attribute -Wformat=2 -Wwrite-strings -Wcast-align -Wno-long-long -Woverloaded-virtual -Wnon-virtual-dtor -Wcast-qual -Wno-property-attribute-mismatch
all: clear a.out
a.out: main.o funcs.o jordan.o matrix.o
	mpic++ $(OPTS) $^ -o a.out
main.o: main.cpp funcs.h jordan.h matrix.h
	mpic++ -c $(OPTS) $<
funcs.o: funcs.cpp funcs.h
	mpic++ -c $(OPTS) $<
jordan.o: jordan.cpp jordan.h
	mpic++ -c $(OPTS) $<
matrix.o: matrix.cpp matrix.h
	mpic++ -c $(OPTS) $<
clear:
	rm -f *.o
clean:
	rm -f *.out *.o *.bak
