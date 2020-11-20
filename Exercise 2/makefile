make:
	mpicc -c main.c
	mpicc -o exec_main main.o -lm -w
	mpiexec -np 9 exec_main num
make2:
	mpicc -c main.c
	mpicc -o exec_main main.o -lm -w
	mpiexec -np 4 exec_main 
