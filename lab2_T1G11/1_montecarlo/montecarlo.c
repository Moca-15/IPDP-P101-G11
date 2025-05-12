#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct { uint64_t state; uint64_t inc; } pcg32_random_t;
double pcg32_random( pcg32_random_t *rng){

	uint64_t oldstate = rng->state;
	rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
	uint32_t xorshifted = ((oldstate >> 18u) ^  oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

	return (double)ran_int / (double)UINT32_MAX;
}

int main(int argc, char *argv[]){

    int numtasks, rank;

    //default vals
    int d = 3;
    long NUM_SAMPLES = 1000000;
    long SEED = time(NULL);

	if(argc > 1) d = atoi(argv[1]);
	if(argc > 2) NUM_SAMPLES = atol(argv[2]);
	if(argc > 3) SEED = atol(argv[3]);

	MPI_Init(&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	//init rng
    pcg32_random_t rng;
    rng.state = SEED + rank;
    rng.inc = (rank << 16) | 0x3039;


	if(rank == 0) {
    	printf("Monte Carlo sphere/cube ratio estimation\n");
    	printf("N: %d samples, d: %d, seed %d, size: %d\n", NUM_SAMPLES, d, SEED, numtasks);
	}

	printf("Rank %d generated %d\n", rank, pcg32_random(&rng));

    MPI_Finalize();

return 0;

}
