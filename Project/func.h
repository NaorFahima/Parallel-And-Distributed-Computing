#define BITS 8
#define SIZE 256
#define  MAX_KEY_SIZE  8


const int MASTER_RANK = 0;
const int SLAVE_RANK = 1;
const int MAX_WORKERS = 2;


// check if sentence contains word
int checkStr(char* sentence , char* word,char* tempStr);

// add 1 to the char if its not valid symbol return '!'
int addBit(char* ch);

// read from file and return array of string
char** readFile(char* nameFile,int* size);

// read from file and return string
char* readCipherText(char* nameFile,int* size);

// Brute Force search
void BruteForce(int keylen,char* cipherText,char** plaintextList,int sizeList,int cipher_size);

// xor operation [cuda]
void myXor(char* data ,char* key,char* xorstring,int size_data,int size_key);

// This function stores the bytes of the key in the array 'keyBytes'. This will simplify the access to the bytes later on.
int processKey(char *key);

// convert  hexadecimal digit to number between 0 and 15
int hex2int(char h);