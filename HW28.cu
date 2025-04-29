// Name:
// CPU random walk. 
// nvcc HW28.cu -o temp

/*
 What to do:
 This is some code that runs a random walk for 10000 steps.
 Use cudaRand and run 10 of these runs at once with diferent seeds on the GPU.
 Print out all 10 final positions.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);

/*
 RAND_MAX = 2147483647
 rand() returns a value in [0, 2147483647].
 Because RAND_MAX is odd and we are also using 0 this is an even number.
 Hence there is no middle interger so RAND_MAX/2 will divide the number in half if it is a float.
 You might could do this faster with a clever idea using ints but I'm going to use a float.
 Also I'm not sure how long the string of random numbers is. I'm sure it is longer than 10,000.
 Before you use this a a huge string check this out.
*/
int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

int main(int argc, char** argv)
{
	srand(time(NULL));
	
	int position = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		position += getRandomDirection();
	}
	
	printf(" Final position = %d \n", position);
	return 0;
}

