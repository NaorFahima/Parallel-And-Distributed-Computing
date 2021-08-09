#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <string.h>


/**
 *    This program reads its input (the plaintext) from the standard input.
 *    It encrypts the plaintext and writes the ciphertext to the standard
 *    output.
 * 
 *    Alternatively, the ciphertext may be supplied as input and the plaintext
 *    will be written to the output.
 * 
 *    The program accepts a single command line argument: the key.  The key
 *    should  be specified as a sequence of hexadecimal digits. The number of digits
 *    should be even.  This means the key will have an integer number of bytes
 *    (since 2 hexadecimal digits specify the bits in one byte).
 *   
 *    example:     $ ./encode ABCD12 < plaintext > ciphertext
 * 
 *    Here the program is called 'encode' and the key is ABCD12. ($ is the shell prompt)
 *    This means the key has  3 bytes:  The first byte is 0xAB, the second is 0xCD and the
 *    third is 0x12.  
 * 
 *      
 *    Encryption method:  the plaintext is xored with the key.
 *    The first byte of the plaintext is xored with the first byte of the key, the second
 *    byte of the plaintext is xored with the second byte of the key and so on.
 *    When the bytes of the key are used up, we use the first byte of the key again 
 *    and then the second byte and so on.
 * 
 *    Note that decryption is done by xoring the ciphertext with the same key.
 * 
 *    This means that if we execute for example
 * 
 *    $ ./encode ABCD12 < plaintext |  ./encode ABCD12 > result
 * 
 *    then 'result'  will  be the same as 'plaintext'
**/

// max number of bytes in key
#define  MAX_KEY_SIZE  8
 
char keyBytes[MAX_KEY_SIZE];

int processKey(char *key);

int main(int argc, char **argv) {

    if (argc != 2) { 
         fprintf(stderr, "usage: %s <key>\n\
key should be specified with hexa digits\n", argv[0]);
         exit(1);
    }

    int nbytes; // number of bytes in key
    nbytes = processKey(argv[1]);

    int c;
    
    //  encode the input
    for (int i = 0; (c = getchar()) != EOF; ) {
        putchar (c ^ keyBytes[i]); // ^  is the xor operator
        i++;
        if (i >= nbytes)
            i = 0;
    }
    return 0; 
} // main

 
// convert  hexadecimal digit to number between 0 and 15
// examples:  hexint('3') returns 3,  hexint('A') returns 10, hex2int('B') returns 11
// hexa letters may be lowercase or uppercase (e.g. 'a' or 'A')
int hex2int(char h) {
    h = toupper(h); // if h is a lowercase letter then convert it to uppercase

    if (h >= '0' && h <= '9')
        return h - '0';
    else if (h >= 'A' && h <= 'F')
        return h - 'A' + 10;
    else { 
        fprintf(stderr, "key should contain hexa digits\n");
        exit(1);
    }
    return 0;
}

/**
  This function stores the bytes of the key in the array 'keyBytes'. This will simplify the 
  access to the bytes later on.

   example:  Assume the key has 4 bytes:   <byte 0> <byte 1> <byte 2> <byte 3>

   Then the contents of 'KeyBytes'  will be:
   char keyBytes[] = { <byte 0>, <byte 1>, <byte 2>, <byte 3> };  
   
   byte0  will be applied  to the first character of the plaintext,
   <byte 1> will be applied to the second character of the plaintext and so on.

   The number of bytes in the key is returned.
   (the number of hexa digits in the key must be even so that the key has an integer number of bytes)

**/
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
    
