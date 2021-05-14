#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include "func.h"


int main(int argc, char* argv[])
{
	char** word_list;
	char* cipher_text;
	int words_size,cipher_size,tempSize;
	int numberOProcess,rank,size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOProcess);

	// check the number of process
	if (numberOProcess != MAX_WORKERS) {
		printf("Program requires %d nodes\n", MAX_WORKERS);
		MPI_Finalize();
		return 0;
	}

	// check the number of arguments
	if( argc > 4 || argc < 3) {
		if(MASTER_RANK == rank)
			printf("Need 2 or 3 arguments to supplie.\nLEN_BITS FILE_CIPHER [FILE_WORDS]\n");
		MPI_Finalize();
		return 0;
	}

	//MASTER
	if (MASTER_RANK == rank) {

		// if not input file of common words use default
		if(argc == 3)
			argv[3] = (char*) "unix_words";

		// read cipher text from file
		cipher_text = readCipherText(argv[2],&cipher_size);

		// read words from file
		word_list = readFile(argv[3],&words_size);

		// send words size to slave
		MPI_Send(&words_size,1, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD);


		// send size cipher text to slave
		MPI_Send(&cipher_size,1, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD);


		// send words to slave
		for(int i = 0 ; i < words_size ; i++){
			tempSize = strlen(word_list[i]);
			MPI_Send(&tempSize,1, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD);
			MPI_Send(word_list[i], tempSize, MPI_CHAR, SLAVE_RANK, 0, MPI_COMM_WORLD);
		}

		// send cipher text to slave
		MPI_Send(cipher_text,cipher_size, MPI_CHAR, SLAVE_RANK, 0, MPI_COMM_WORLD);

	//SLAVE
	} else {

		// receiving  words size from master
		MPI_Recv(&words_size,1, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// receiving size of cipher text from master
		MPI_Recv(&cipher_size,1, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		word_list = (char**) malloc(sizeof(char*)*words_size);
		cipher_text = (char*) malloc(sizeof(char)*cipher_size);

		// check if allocation succeed
		if(word_list == NULL || cipher_text == NULL){
			printf("Malloc failed in process %d\n",rank);
			exit(1);
		}

		// receiving words from master
		for(int i = 0 ; i < words_size ; i++){
			MPI_Recv(&tempSize, 1, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			word_list[i] = (char*) malloc(sizeof(char)*tempSize);
			word_list[i][tempSize] = '\0';
			MPI_Recv(word_list[i], tempSize, MPI_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}

		// receiving cipher text from master
		MPI_Recv(cipher_text,cipher_size, MPI_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// found the key
		BruteForce(atoi(argv[1]),cipher_text,word_list,words_size,cipher_size);

		// free memory
		for(int i = 0 ; i<words_size;i++)
			free(word_list[i]);
		free(word_list);
		free(cipher_text);
	}

	MPI_Finalize();
	return 0;
}


