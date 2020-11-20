#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define ROOT 0
#define SIZE 200

typedef struct Vector {
	int x,y,z;
}Vector;

int is_perefect_square(int number);
void print_vectors_matrix(Vector* mat, int vectorsInRow);
Vector* read_from_file_or_input(int argc, char *argv[],int numOfVector,int* error);
void read_numbers(FILE* f,Vector *matrix,int numOfVector,int* error);
void create_vector_type(MPI_Datatype * vector_type);
void shear_sort(Vector *myVector , int size ,int my_rank,MPI_Datatype VectorType);
void odd_even_sort_cols(Vector *myVector,int size,int my_rank,MPI_Datatype VectorType);
void odd_even_sort_rows(Vector *myVector,int size,int my_rank,MPI_Datatype VectorType);
void compare_and_exchange(Vector *v1,Vector *v2);
void swap(Vector *v1,Vector *v2);


int main(int argc, char *argv[]) {

	int my_rank,num_procs,rowSize,error = 0;
	Vector *matrix ,myVector ;
	MPI_Datatype VectorType;

	  // Start up MPI
	  MPI_Init(&argc, &argv);

	  // Find out process rank
	  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	  // Find out number of processes
	  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	  create_vector_type(&VectorType);

	  if(my_rank == ROOT){
		  rowSize = is_perefect_square(num_procs); // check if the number is square
		  if(rowSize != 0)
			  matrix = read_from_file_or_input(argc,argv,num_procs,&error);
		  else{
			  printf("Number of processes is not square.\n");
			  error = 1;
		  }

	  }

	  MPI_Bcast(&error, 1, MPI_INT, ROOT, MPI_COMM_WORLD); // send the type of error to all processes
	  MPI_Barrier(MPI_COMM_WORLD); // wait to all processes

	  if (error == 0) { // check if succeed to read the numbers

		  if(my_rank == ROOT){
			  printf("\nMatrix before sorting:\n");
			  print_vectors_matrix(matrix,rowSize); //prints matrix before sorting
		  }
		  MPI_Bcast(&rowSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD); //broadcast the number of vectors in a row(needed for fuctions)

		  MPI_Scatter(matrix, 1, VectorType, &myVector, 1, VectorType, ROOT, MPI_COMM_WORLD); //send a vector to each processes.

		  shear_sort(&myVector,num_procs,my_rank,VectorType); //sort

		  MPI_Gather(&myVector, 1, VectorType, matrix, 1, VectorType, ROOT, MPI_COMM_WORLD); //get all the numbers from processes

		  if (my_rank == ROOT)
		  {
			  printf("\nMatrix after sorting:\n");
			  print_vectors_matrix(matrix, rowSize); //prints matrix after sorting
			  free(matrix);
		  }
	  }

	  MPI_Finalize();
	  return 0;
}


int is_perefect_square(int number){
	int iVar;
	float fVar;

	fVar=sqrt((double)number);
	iVar = fVar;
	if(iVar == fVar)
		return iVar;
	else
		return 0;
}

void print_vectors_matrix(Vector* mat, int vectorsInRow)
{
	int i, j, n = vectorsInRow;

	//prints as a matrix. when sorted it'll be as a snake
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
			printf("(%d,%d,%d) \t", mat[i * n + j].x, mat[i * n + j].y, mat[i * n + j].z);
		printf("\n");
	}
}

Vector* read_from_file_or_input(int argc, char *argv[],int numOfVector,int* error){
	FILE *f;
	char str[SIZE];
	Vector *matrix;
	matrix = (Vector*)calloc(numOfVector, sizeof(Vector)); //initialize array for vector

	if (!matrix)
		MPI_Abort(MPI_COMM_WORLD, 2);

	f = fopen(argv[1], "r");
	if (f == NULL) { // Read from input
		printf("File could not open or not input file name.\n");
		printf("Input %d numbers in the format: \"number number number $\" \n",numOfVector);
		fgets(str,sizeof(str),stdin); // read user input
		f = fopen("userInput.txt", "w+");
		fprintf(f,"%s",str);
		fseek(f,0,SEEK_SET);
		read_numbers(f,matrix,numOfVector,error);
		remove("userInput.txt");
	}
	else // Read from file
		read_numbers(f,matrix,numOfVector,error);

	return matrix;
}

void read_numbers(FILE* f,Vector *matrix,int numOfVector,int* error){
	int count = 0;
	char dollar;

	while(!feof(f)){ // read all vectors
		if(count < numOfVector){
			fscanf(f,"%d %d %d %c",&matrix[count].x,&matrix[count].y,&matrix[count].z,&dollar);
			while(dollar != '$')
				fscanf(f,"%c",&dollar);
		}
		else
			fscanf(f,"%d %d %d %c",&matrix[0].x,&matrix[0].y,&matrix[0].z,&dollar);

		count++;
	}
	count--;
	fclose(f);

	if (numOfVector != count) // every process must have a vector
	{
		printf("\nInvalid number of processes: %d instead of %d\n", numOfVector, count);
		*error = 1;
	}

}

void create_vector_type(MPI_Datatype * vector_type)
{
	int block_lengths[3] = {1,1,1};
	MPI_Aint disp[3];
	MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT};

	disp[0] = offsetof(Vector,x);
	disp[1] = offsetof(Vector,y);
	disp[2] = offsetof(Vector,z);

	MPI_Type_create_struct(3,block_lengths,disp,types,vector_type);
	MPI_Type_commit(vector_type);
}


void shear_sort(Vector *myVector , int size ,int my_rank,MPI_Datatype VectorType){

	int iterations = (int)ceil(log2(size) + 1); //number of iterations in shear sort

	for(int i = 0 ; i<iterations ;i++){
		odd_even_sort_rows(myVector,size,my_rank,VectorType);
		odd_even_sort_cols(myVector,size,my_rank,VectorType);
	}
	odd_even_sort_rows(myVector,size,my_rank,VectorType);

}

void odd_even_sort_cols(Vector *myVector,int size,int my_rank,MPI_Datatype VectorType){
	int sqrt_val = (int)sqrt(size);
	Vector temp;
	for(int i = 0 ; i<sqrt_val;i++){
		if(i%2 == my_rank%2 && my_rank<size-sqrt_val){
			MPI_Recv(&temp,1,VectorType,my_rank+sqrt_val,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			compare_and_exchange(myVector,&temp);
			MPI_Send(&temp,1,VectorType,my_rank+sqrt_val,0,MPI_COMM_WORLD);

		} else if(my_rank >=sqrt_val && i%2 != my_rank%2){
			MPI_Send(myVector,1,VectorType,my_rank-sqrt_val,0,MPI_COMM_WORLD);
			MPI_Recv(myVector,1,VectorType,my_rank-sqrt_val,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void odd_even_sort_rows(Vector *myVector,int size,int my_rank,MPI_Datatype VectorType){
	int sqrt_val = (int)sqrt(size);
	Vector temp;
	for(int i = 0 ; i<sqrt_val;i++){
		if(i%2 == my_rank%2 && my_rank%sqrt_val != sqrt_val-1 ){
			MPI_Recv(&temp,1,VectorType,my_rank+1,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

			if((my_rank/sqrt_val)%2 == 0) // sort like a snake
				compare_and_exchange(myVector,&temp);
			else
				compare_and_exchange(&temp,myVector);

			MPI_Send(&temp,1,VectorType,my_rank+1,0,MPI_COMM_WORLD);

		} else if(my_rank%sqrt_val != 0 && i%2 != my_rank%2){
			MPI_Send(myVector,1,VectorType,my_rank-1,0,MPI_COMM_WORLD);
			MPI_Recv(myVector,1,VectorType,my_rank-1,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void compare_and_exchange(Vector *v1,Vector *v2){
	if(v1->x > v2->x)
		swap(v1,v2);
	else if((v1->x == v2->x) && (v1->y > v2->y))
		swap(v1,v2);
	else if ((v1->x == v2->x) && (v1->y == v2->y) && (v1->z > v2->z))
		swap(v1,v2);

}

void swap(Vector *v1,Vector *v2){
	Vector temp;
	temp = *v1;
	*v1 = *v2;
	*v2 = temp;
}



