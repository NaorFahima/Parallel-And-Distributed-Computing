build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c func.c -o func.o
	nvcc -I./inc -c cuda_func.cu -o cuda_func.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o func.o cuda_func.o   /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt 
	


clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP 8 helloWorld.e words

run1:
	mpiexec -np 2 ./mpiCudaOpemMP 8 lawyerJoke.e words
 
run2:
	mpiexec -np 2 ./mpiCudaOpemMP 16 gettysburgAddress.e words

run3:
	mpiexec -np 2 ./mpiCudaOpemMP 32 gettysburgAddress.e words
