#pragma once

#define ARR_SIZE 257

int* openMP_task(int* src_arr,int size);
int* cuda_task(int* arr, int size);
void merge_task(int* dest_array, int* src_array);
void reset_array(int* arr, int size);
int* read_numbers(char* input, int* size);
void print_hist(int* histogram);
int* calculate_histogram(int *image, unsigned int size);
