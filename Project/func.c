#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/sysinfo.h>
#include "func.h"
#include "mpi.h"
#include "omp.h"

char keyBytes[MAX_KEY_SIZE];


int checkStr(char* sentence , char* word,char* tempStr){
	memcpy(tempStr, sentence, strlen(sentence)+1);
	tempStr = strtok(tempStr," ");
	while(tempStr != NULL){
		if(!strcmp(tempStr,word))
			return 1;
		tempStr = strtok(NULL," ");
	}
	return 0;
}


int addBit(char* ch){
	unsigned int mask = 1;
	while (*ch & mask)
	{
		*ch &= ~mask;
		mask <<= 1;
	}
	*ch |= mask;

	if(*ch < 0x2F) // 0x2F == '/'
		*ch = 0X0000 | 0x30; // 0x30 = '0'  
	if(*ch > 0x47) // 0x46 == 'G'
		*ch = *ch & 0X0046; // 0x46 = 'F'
	if(0x39<*ch && *ch<0x41) // 0x39 = '9' , 0x41 = 'A'
		*ch = 0X0000 | 0x41;

	return (int) *ch;
}

char** readFile(char* nameFile,int* size){
	FILE *fp;
	char** list;
	char* tempStr;

	fp = fopen(nameFile,"r");
	if(fp == NULL){
		printf("Failed to open the file.\n");
		MPI_Abort(MPI_COMM_WORLD,-1);
	}
	fscanf(fp, "%d", size);
	list = (char**) malloc(sizeof(char*)*(*size));
	if(list == NULL){
		printf("Malloc failed in function readFile\n");
		MPI_Abort(MPI_COMM_WORLD,-1);
	}
	for(int i = 0 ; i <*size ; i++){ // get all the word from file
		tempStr = (char*) malloc(sizeof(char)*15);
		if(tempStr == NULL){
			printf("Malloc failed in function readFile\n");
			MPI_Abort(MPI_COMM_WORLD,-1);
		}
		fscanf(fp, "%s",tempStr);
		list[i] = tempStr;
	}
	fclose(fp);
	return list;
}

char* readCipherText(char* nameFile,int* size){
	FILE *fp;
	char* list;

	fp = fopen(nameFile,"r");
	if(fp == NULL){
		printf("Failed to open the file.\n");
		MPI_Abort(MPI_COMM_WORLD,-1);
	}
	//get the size of the file
	fseek(fp, 0L, SEEK_END);
	*size = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	list = (char*) malloc(sizeof(char)*(*size));
	if(list == NULL){
		printf("Malloc failed in function readCipherText\n");
		MPI_Abort(MPI_COMM_WORLD,-1);
	}
	fread(list, sizeof(char), *size, fp);
	fclose(fp);
	return list;
}



int hex2int(char h) {
    h = toupper(h); // if h is a lowercase letter then convert it to uppercase
    if (h >= '0' && h <= '9')
        return h - '0';
    else if (h >= 'A' && h <= 'F')
        return h - 'A' + 10;
    return 0;
}

int processKey(char *key) {
    int n = strlen(key);
    if (n%2 || n/2 > MAX_KEY_SIZE) {
        fprintf(stderr, "key must have even number of bytes. Number of bytes \
should not exceed %d\n", MAX_KEY_SIZE);
        exit(1);
    }

    for(int i = 0; i < n; i += 2) {
         keyBytes[i/2] = (hex2int(key[i]) << 4) | hex2int(key[i+1]);
    }
    return n/2;
}
    

void BruteForce(int keylen,char* cipherText,char** plaintextList,int sizeList,int cipher_size){
	int index = 0 ,keyFound = 0,counter = 0,i = 0,lenght = 16; // lenght = 16 is the number of legal symbols that the key can have
	int sizeKey = ceil(keylen/BITS)*2,nCPU = get_nprocs_conf();
	long long iterations = 1;
	char pKey[sizeKey];
	char decipherText[cipher_size];
	char* tempStr = (char*) malloc(sizeof(char)*cipher_size);
	char* key = (char*)calloc(sizeKey,sizeof(char));

	if(key == NULL || tempStr == NULL){ // chack if memeoy allocation success
		printf("Malloc failed in function BruteForce\n");
		MPI_Abort(MPI_COMM_WORLD,-1);
	}

	for(int i = 0 ; i <sizeKey ; i++) {// calculates the number of iterations
		iterations *= lenght;
		key[i] = '0';
	}
	iterations = iterations - 3;

	int nbytes; // number of bytes in key
    nbytes = processKey(key);
	char pKeyBytes[nbytes];


#pragma omp parallel private(pKey,pKeyBytes,decipherText,counter) num_threads(nCPU)
	{
		while(i<iterations && keyFound == 0){
			strcpy(pKeyBytes,keyBytes); // copy the key to thread private variable 
			strcpy(pKey,key); // copy the key to thread private variable 

// change the key
#pragma omp critical
			{
				while(addBit(&key[index]) == 0X0047 && index + 1 < sizeKey){
					key[index] = '0';
					index += 1;

				}
				index = 0;
			}
			processKey(key);
			myXor(cipherText,pKeyBytes,decipherText,cipher_size,nbytes); // decipher the text with the key by xor [cuda]
		
			
#pragma omp parallel for
			for(long long j = 0 ; j<sizeList; j++){ // loop that check if the text contain word from the file
				counter  += checkStr(decipherText,plaintextList[j],tempStr);
				if(counter == 3 && keyFound == 0){
					printf("Key found is: %s\n",pKey);
					printf("Decipher text:\n%s\n",decipherText);
					keyFound = 1;
					j = sizeList; // break for loop
				}
			}
			counter = 0;
			i++;
		}
	}
	if(!keyFound)
		printf("not found\n");

	free(tempStr);
	free(key);
}


