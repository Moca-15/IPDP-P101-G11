#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef struct { uint64_t state; uint64_t inc; } pcg32_random_t;
double pcg32_random( pcg32_random_t *rng){

	uint64_t oldstate = rng->state;
	rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

	return (double)ran_int / (double)UINT32_MAX;
}

//generate random doubles in [-1,1] space (shift from [0,1])
double shifted_random(pcg32_random_t *rng){
	double val = pcg32_random(rng);
	val = (val - 0.5) * 2;
	return val;
}

void generate_d_vector(double *vect, int d, pcg32_random_t *rng){
	for(int i=0; i<d; i++) vect[i] = shifted_random(rng);
}

//calculate module for d dimension vector
double vector_mod(double *vect, int d){
	double accum = 0;
	for(int i=0; i<d; i++) accum += vect[i]*vect[i];
	return sqrt(accum);
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
    rng.state = SEED + rank;  //0x853c49e6748fea9b + rank; //SEED + rank;
    rng.inc = (rank<<16) | 0x3039; //0xda3e39cb94b95bdb;//(rank << 16) | 0x3039;

	//amount of samples to take care of in this node
	long local_n_samples = NUM_SAMPLES / numtasks + ((rank >= NUM_SAMPLES%numtasks)? 0 : 1);

	double time = MPI_Wtime();

	pcg32_random(&rng);

	int count = 0;
	double *vector = malloc(sizeof(double) * d);
	for(int i=0; i<local_n_samples; i++){
		generate_d_vector(vector, d, &rng);
		if(vector_mod(vector, d) <= 1.0) count++;
	}
	//printf("Rank: %d; points inside sphere: %d out of %d\n",rank, count, local_n_samples);
	time = MPI_Wtime() - time;

	int total_sum;
	double maxtime;

	MPI_Reduce(&count, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if(rank == 0){
		double estimated_ratio = (double)total_sum / (double)NUM_SAMPLES;
		double real_ratio = pow(M_PI, (double)d/(double)2) / (pow(2,d) * tgamma((double)d/(double)2 + 1));
		double err = fabs(real_ratio-estimated_ratio);

		printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("N: %d samples, d: %d, seed %d, size: %d\n", NUM_SAMPLES, d, SEED, numtasks);
		printf("Ratio = %.3e (%.3e) Err: %.3e\n", estimated_ratio, real_ratio, err);
		printf("Elapsed time: %5.3lf seconds\n", maxtime);
	}

	 MPI_Finalize();

return 0;

}
