build:
	mpicxx -fopenmp -c histogram.cpp -o histogram.o
	nvcc -I./inc -c cuda_histogram.cu -o cuda_histogram.o
	mpicxx -fopenmp -o mpiCudaOpemMP  histogram.o  cuda_histogram.o  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP


