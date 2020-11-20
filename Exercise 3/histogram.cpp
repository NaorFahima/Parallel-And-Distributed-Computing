
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "omp.h"
#include "histogram.h"

const int MASTER_RANK = 0;
const int SLAVE_RANK = 1;
const int MAX_WORKERS = 2;


int main(int argc, char* argv[])
{
	char input[ARR_SIZE];
	int numberOProcess,rank,size;
	int* histogram,*histogram2,*numbers,*received_array;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOProcess);

	if (numberOProcess != MAX_WORKERS)
	{
		printf("Program requires %d nodes\n", MAX_WORKERS);
		MPI_Finalize();
		exit(1);
	}

	if (MASTER_RANK == rank)
	{

		received_array = new int[ARR_SIZE]; 
		reset_array(received_array, ARR_SIZE);

		// read numbers from stdin
		numbers = read_numbers(input,&size);

		// send half of array and size to slave
		MPI_Send(&size,1, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD);
		MPI_Send(numbers + (int)ceil((double) size/2), size/2, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD);

		// calculate histogram with openMP and cuda
		histogram = openMP_task(numbers,(int)ceil((double) size/4));
		histogram2 = cuda_task(numbers+(int)ceil((double)size/4),size/4);

		// merge 2 histogram
		merge_task(histogram, histogram2);

		// receiving the histogram that slave calculate
		MPI_Recv(received_array, ARR_SIZE, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// merge 2 histogram
		merge_task(histogram, received_array);

		// print histogram
		print_hist(histogram);

	}
	else
	{
		// receiving size of array from master
		MPI_Recv(&size,1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		size = size/2;
		received_array = new int[size];

		// receiving array from master
		MPI_Recv(received_array, size, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		// calculate histogram with openMP and cuda
		histogram = openMP_task(received_array,size/2);
		histogram2 = cuda_task(received_array+size/2,(int)ceil((double)size/2));

		// merge 2 histogram
		merge_task(histogram, histogram2);

		// send histogram to master
		MPI_Send(histogram, ARR_SIZE, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD);
	}


	MPI_Finalize();
	return 0;

}

int* read_numbers(char* input, int* size)
{
	int* numbers;
	int temp;
	printf("Input size and numbers:\n");
	fgets(input,ARR_SIZE,stdin);
	*size = atoi(strtok(input, " "));
	numbers = (int*) malloc((*size)*sizeof(int));
	for (int i = 0; i < *size; i++){
		temp = atoi(strtok(NULL, " "));
		if(temp>0 && temp<257)
			numbers[i] = temp;
		else {
			*size = *size - 1;
			i = i - 1;
		}
	}
	return numbers;
}



void reset_array(int* arr, int size)
{
	for (int i = 0; i < size; i++)
		arr[i] = 0;
}



int* openMP_task(int* src_arr,int size)
{
	int* dst_arr = new int[ARR_SIZE];
	int* tmp_hist;

	reset_array(dst_arr, ARR_SIZE);

#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
#pragma omp single
		{
			tmp_hist = new int[ARR_SIZE*nthreads];
			reset_array(tmp_hist, ARR_SIZE*nthreads);
		}

#pragma omp for
		for (int i = 0; i < size; i++)
			tmp_hist[tid*ARR_SIZE + src_arr[i]]++;

// merge
#pragma omp for
		for (int i = 0; i < ARR_SIZE; i++)
			for (int j = 0; j < nthreads; j++)
				dst_arr[i] += tmp_hist[j*ARR_SIZE + i]; // each thread merges specific cell in tmp_arr to dst_arr

	}
	free(tmp_hist);
	return dst_arr;
}

int* cuda_task(int* arr, int size)
{
	return calculate_histogram(arr, size);
}

void merge_task(int* dest_array, int* src_array)
{
#pragma omp for
	for (int i = 0; i < ARR_SIZE; i++)
		dest_array[i] += src_array[i];
}

void print_hist(int* histogram)
{
	printf("Histogram:\n");
	for (int i = 1; i < ARR_SIZE; ++i) {
		if(histogram[i] != 0)
			printf("%d: %d\n",i,histogram[i]);
	}
}

