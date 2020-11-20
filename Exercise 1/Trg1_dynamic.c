#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define HEAVY 100000
#define SHORT 1
#define LONG 10

enum ranks{ROOT,N=20,WORK_TAG=0,END_TAG = 1};

void generateArr(int* arr, int size);
void masterProcess(int num_procs);
void slaveProcess();

// This function performs heavy computations,
// its run time depends on x and y values
double heavy(int x, int y) {
	int i, loop = SHORT;
	double sum = 0;
	// Super heavy tasks
	if (x < 3 || y < 3)
		loop = LONG;
	// Heavy calculations
	for (i = 0; i < loop * HEAVY; i++)
		sum += cos(exp(sin((double) i / HEAVY)));
	return sum;
}

int main(int argc, char *argv[]){

	int my_rank,num_procs;

	// Start up MPI
	MPI_Init(&argc, &argv);

	// Find out process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Find out number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if(my_rank == ROOT)
		masterProcess(num_procs);
	else
		slaveProcess();


	MPI_Finalize();
	return 0;
}

void masterProcess(int num_procs){
	MPI_Status status;
	int worker_id,source,tag;
	int jobs_sent=0 ,jobs_to_do = N*N ,arr_size = N*N*2;
	int* arr = (int*)malloc(sizeof(int)*arr_size);
	double start = MPI_Wtime() ,global_count=0,localCount;

	generateArr(arr,N);

	// Send first task to processes
	for(worker_id = 1; worker_id < num_procs; worker_id++)
		MPI_Send(arr+(worker_id-1)*2,2,MPI_INT,worker_id,WORK_TAG,MPI_COMM_WORLD);

	// Send task and Recv calculations from processes
	for(jobs_sent = num_procs-1;jobs_sent<jobs_to_do;jobs_sent++)
	{
		if(jobs_to_do-jobs_sent > num_procs-1)
			tag = WORK_TAG;
		else
			tag = END_TAG;

		MPI_Recv(&localCount,1,MPI_DOUBLE,MPI_ANY_SOURCE,WORK_TAG,MPI_COMM_WORLD,&status);
		source = status.MPI_SOURCE;
		MPI_Send(arr + jobs_sent*2,2,MPI_INT,source,tag,MPI_COMM_WORLD);
		global_count += localCount;
	}

	// Recv last calculations from processes
	for(worker_id = 1;worker_id<num_procs;worker_id++){
		MPI_Recv(&localCount,1,MPI_DOUBLE,MPI_ANY_SOURCE,END_TAG,MPI_COMM_WORLD,&status);
		global_count += localCount;
	}

	printf("answer = %e\n",global_count);
	printf("Time %lf\n",MPI_Wtime()-start);
}

void slaveProcess()
{
	MPI_Status status;
	int tag ,myArr[2];
	double temp;
	do{
		MPI_Recv(myArr,2,MPI_INT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		tag = status.MPI_TAG;
		temp = heavy(myArr[0],myArr[1]);
		MPI_Send(&temp,1,MPI_DOUBLE,ROOT,tag,MPI_COMM_WORLD);
	} while(tag != END_TAG);
}

void generateArr(int* arr, int size)
{
	int count = 0;
	for (int x = 0; x < size; x++){
		for (int y = 0; y < size; y++){
			arr[count++] = x;
			arr[count++] = y;
		}
	}
}


