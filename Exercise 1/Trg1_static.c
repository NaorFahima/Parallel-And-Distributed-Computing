
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define HEAVY 100000
#define SHORT 1
#define LONG 10

enum ranks{ROOT};

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

int main(int argc, char *argv[]) {
	int x, y,my_rank,num_procs,start;
	int N = 20;
	double answer = 0,temp = 0;

	// Start up MPI
	MPI_Init(&argc, &argv);

	// Find out process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Find out number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if(my_rank == ROOT)
		start = MPI_Wtime();

	// Divides the work by process
	for(x = my_rank*N/num_procs;x<(my_rank+1)*N/num_procs;x++)
		for(y = 0;y<N;y++)
			temp += heavy(x, y);

	// Sum all calculations
	 MPI_Reduce(&temp, &answer, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);

	if(my_rank == ROOT)
		printf("answer = %e\nTime: %lf\n ", answer,MPI_Wtime()-start);
	
	MPI_Finalize();
	return 0;
}




