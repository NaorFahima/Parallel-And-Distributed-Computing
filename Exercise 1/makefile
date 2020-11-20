static:
	mpicc -c Trg1_static.c
	mpicc -o exec_static Trg1_static.o -lm
	mpiexec -np 4 exec_static

dynamic:
	mpicc -c Trg1_dynamic.c
	mpicc -o exec_dynamic Trg1_dynamic.o -lm
	mpiexec -np 4 exec_dynamic
