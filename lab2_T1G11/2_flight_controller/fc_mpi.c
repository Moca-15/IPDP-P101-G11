#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "auxiliar.h"


//#define DEBUG 0

void debug(int rank, char  *format, ...) {
	#ifdef DEBUG

	va_list args; //variable argument list

	va_start(args, format);

	printf("%2d | ", rank);
	vprintf(format, args);

	va_end(args);

	#endif
}


/*
//// MPI TAG INDEX ////
0 : for sending and receiving plane info during file reading






*/



/// TODO CHECK :)
/// Reading the planes from a file for MPI
void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{

	int num_planes;
	int int_arr[3];
	double db_arr[2];

	MPI_File fh;
	MPI_Status status;
	MPI_Offset data_start = 0; // quan acaba el header i començen els planes

	// MPI escriurà el file handler a l'adreça de fh
	int err = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	if(err != MPI_SUCCESS) {
		fprintf(stderr, "error opening file :(\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// 0 llegirà data del header (les 1eres línies)
	if(rank == 0) {
		int max_line_size = 100;
		char line[max_line_size];

		char c;
		int line_pos;

		for(int i = 0; i<4; i++) {
			line_pos = 0;
			while(1) {
				MPI_File_read_at(fh, data_start, &c, 1, MPI_CHAR, &status);
				data_start++;
				if(c == '\n') break;
				line[line_pos++] = c;
				if(line_pos >= max_line_size-1) break;
			}
			line[line_pos] = '\0'; 
			debug(rank, "read line %s\n", line);

			// línies 2 i 3 que contenen info >:v
			if(i == 1) {sscanf(line, "# Map: %lf, %lf : %d %d", x_max, y_max, N, M);}
			if(i == 2) {sscanf(line, "# Number of Planes: %d", &num_planes);}
		}

		// arrays per als broadcasts (tots els ranks necessiten aquesta info
		int_arr[0] = *N; int_arr[1] = *M; int_arr[2] = num_planes;
		db_arr[0] = *x_max; db_arr[1] = *y_max;

	}

	MPI_Bcast(int_arr, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(db_arr, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&data_start, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

	debug(rank, "received: N = %d : M = %d : num_planes = %d \n\t x_max = %f : y_max = %f \n\t data_start = %d\n", int_arr[0], int_arr[1], int_arr[2], db_arr[0], db_arr[1], data_start);

	// assignar valors als altres ranks
	*N = int_arr[0];
	*M = int_arr[1];
	num_planes = int_arr[2];

	*x_max = db_arr[0];
	*y_max = db_arr[1];

	//omplir tile_displacements
	int total_tiles = *N * *M;

	for(int i=0; i<size+1; i++){
		tile_displacements[i] = i * (total_tiles/size) + ((i>=total_tiles%size)? 0 : 1);
		debug(rank, "Tile_displacements[%d] : %d\n", i, tile_displacements[i]);
	}

	MPI_Offset file_size;
	MPI_File_get_size(fh, &file_size);

	MPI_Offset plane_data_size = file_size - data_start; // mida de les dades dels avions (fitxer sense el header)
	int line_size = (num_planes != 0 ? plane_data_size / num_planes : 0); // mida d'una línia (totes seràn iguals). serà enter pq per cada plane hi ha 1 línia
	int local_planes = num_planes / size + ((rank >= num_planes%size)? 0 : 1); // els que ha d'executar aquest rank
	MPI_Offset displacement = data_start;
	for(int i = 0; i < rank; i++) { // calcular on ha de començar a llegir sabent on acaben els ranks anteriors
		// loacl planes del rank i (tots els anteriors) per line size
		displacement += (num_planes / size + ((i >= num_planes%size) ? 0 : 1)) * line_size;
	}
	// MPI_Offset displacement = data_start + rank * local_planes * line_size;

	debug(rank, "\tfile size = %d\n\tdata size = %d\n\tline size = %d\n\tmy planes = %d\n\tdisplacement = %lld\n", file_size, plane_data_size, line_size, local_planes, displacement);


	// llegir planes
	char line[line_size];
	char c;
	int line_pos;

	int idx;
	double x, y, vx, vy;
	int index_i, index_j, index_map, prank;
	double to_send[6];
	int flag;

	for(int i = 0; i<local_planes; i++) {
		line_pos = 0;
		for(int j = 0; j<line_size; j++) {
			MPI_File_read_at(fh, displacement, &c, 1, MPI_CHAR, &status);
			displacement++;
			line[line_pos++] = c;
		}
		line[line_pos-1] = '\0';
		debug(rank, "read line %s\n", line);

		// parsejar-la
		if(sscanf(line, "%d %lf %lf %lf %lf", &idx, &x, &y, &vx, &vy) == 5) {
			//printf("%2d | plane read with idx %d \n", rank, idx);
			index_i = get_index_i(x, *x_max, *N);
			index_j = get_index_j(y, *y_max, *M);
			index_map = get_index(index_i, index_j, *N, *M);
			prank = get_rank_from_index(index_map, tile_displacements, size);
 			debug(rank, "idx_i = %d\tidx_j = %d\tidx_map = %d\tprank = %d\n", index_i, index_j, index_map, prank);
			// com que cada plane ha de tenir a la llista només els que estàn a les seves tiles:
			if(prank != rank) {
				to_send[0] = (double)idx; to_send[1] = (double)index_map; to_send[2] = x; to_send[3] = y; to_send[4] = vx; to_send[5] = vy;
				MPI_Send(to_send, 6, MPI_DOUBLE, prank, 0, MPI_COMM_WORLD); 
				//printf("%2d | sent over plane with idx %d and rank %d\n", rank, idx, prank);
			} else {
				//printf("%2d | 1 inserted plane with idx %d and rank %d\n", rank, idx, prank);
				insert_plane(planes, idx, index_map, prank, x, y, vx, vy);
			}
			// per anar rebent missatges sobre la marxa
			//printf("%2d | flag before iprobe= %d\n", rank, flag);
			MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
			//printf("%2d | flag after iprobe = %d\n", rank, flag);
			if(flag) {
				MPI_Recv(to_send, 6, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
									// idx 				index_map	  current	x			y			vx			vy
				insert_plane(planes, (int)to_send[0], (int)to_send[1], rank, to_send[2], to_send[3], to_send[4], to_send[5]);
				//printf("%2d | 2 inserted plane with idx %d and rank %d\n", rank, idx, prank);
			}
		}
	}
	// per si ha quedat algun missatge per rebre quan s'han llegit tots els planes
	MPI_Barrier(MPI_COMM_WORLD);
	flag = 1;
	while(flag) {
		MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
		//printf("%2d | flag while = %d\n", rank, flag);
		if(flag) {
			MPI_Recv(to_send, 6, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
			insert_plane(planes, (int)to_send[0], (int)to_send[1], rank, to_send[2], to_send[3], to_send[4], to_send[5]);
			//printf("%2d | 3 inserted plane with idx %d and rank %d\n", rank, idx, prank);
		}
	}


	MPI_File_close(&fh);

	debug(rank, "planes read : %d\n", local_planes);

	int total_read;

	MPI_Reduce(&local_planes, &total_read, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0) printf("Total planes read : %d\n", total_read);



}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
	// quants planes hem d'enviar a cada altre rank
	int *num_sends = (int*)calloc(size, sizeof(int));
	int *num_receives = (int*)calloc(size, sizeof(int));

	double to_send[6], to_receive[6];
	int index_i, index_j, prank;
	int send_to, receive_from;

	printf("%2d | enter comm_planes_send\n", rank);
	// comptar els que surten i cap a on
	PlaneNode *current = list->head;
	while(current != NULL) {
		index_i = get_index_i(current->x, x_max, N);
		index_j = get_index_j(current->y, y_max, M);
		current->index_map = get_index(index_i, index_j, N, M);
		current->rank = get_rank_from_index(current->index_map, tile_displacements, size);
		if(current->rank != rank) num_sends[current->rank]++;
		current = current->next;
	}


	// esquema d'enviar creuat per evitar deadlocks
	for(int i = 0; i < size; i++) {
		send_to = (rank+1)%size;
		receive_from = ((rank-1 < 0) ? size-1 : rank-1);
		MPI_Sendrecv(&num_sends[send_to], 1, MPI_INT, send_to, 1, &num_receives[receive_from], 1, MPI_INT, receive_from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%2d | sent[%d]: %d, received[%d]: %d\n", rank, send_to, num_sends[send_to], receive_from, num_receives[receive_from]);
	}
	// obtenim array amb el num de planes que aquest rank ha de rebre de cada altre.
	printf("%2d | sendrecv num_sends/num_recvs\n", rank);



	// array de requests per al waitall
	int total_sends = 0;
	for(int i = 0; i < size; i++) total_sends += num_sends[i];
	MPI_Request req[total_sends];
	int req_idx = 0;

	printf("%2d | array reqs\n", rank);

	current = list->head;
	while(current != NULL) {
		if(current->rank != rank) {
			to_send[0] = (double)current->index_plane;
			to_send[1] = (double)current->index_map;
			to_send[2] =  current->x; to_send[3] =  current->y;
			to_send[4] = current->vx; to_send[5] = current->vy;
			MPI_Isend(&to_send, 6, MPI_DOUBLE, current->rank, 1, MPI_COMM_WORLD, &req[req_idx++]);

			remove_plane(list, current);
		}
		current = current->next;
	}
	
	for(int i = 0; i < size; i++) printf("%2d | num_receives[%d] = %d\n",rank, i, num_receives[i]);

	for(int i = 0; i < size; i++) {
		printf("%2d | num receives [%d] = %d\n", rank, i, num_receives[i]);
		for(int j = 0; j < num_receives[i]; j++) {
			MPI_Recv(&to_receive, 6, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			insert_plane(list, (int)to_send[0], (int)to_send[1], rank, to_send[2], to_send[3], to_send[4], to_send[5]);
			printf("%2d | received plane %d\n", rank, (int)to_send[1]);
		}

	}
	printf("%2d | finish exec\n", rank);
}

/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
}

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

/// TODO
/// Communicate planes using all to all calls with custom data types
void communicate_planes_struct_mpi(PlaneList* list,
                               int N, int M,
                               double x_max, double y_max,
                               int rank, int size,
                               int* tile_displacements)
{
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    /// TODO
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int tile_displacements[size+1];
    int mode = 0;
    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        if (max_steps <= 0) {
            fprintf(stderr, "max_steps needs to be a positive integer\n");
            return 1;
        }
        mode = atoi(argv[3]);
        if (mode > 2 || mode < 0) {
            fprintf(stderr, "mode needs to be a value between 0 and 2\n");
            return 1;
        }
        check = atoi(argv[4]);
        if (check >= 2 || check < 0) {
            fprintf(stderr, "check needs to be a 0 or 1\n");
            return 1;
        }
    }
    else {
        fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        return 1;
    }

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    print_planes_par_debug(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total=0;

    double start_time = MPI_Wtime();
    int step = 0;
    for (step = 1; step <= max_steps; step++) {
        double start = MPI_Wtime();
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }
        filter_planes(&owning_planes, x_max, y_max);
        time_sim += MPI_Wtime() - start;

        start = MPI_Wtime();
        if (mode == 0)
            communicate_planes_send(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 1)
            communicate_planes_alltoall(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else
            communicate_planes_struct_mpi(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        time_comm += MPI_Wtime() - start;
    }
    time_total = MPI_Wtime() - start_time;

    /// TODO Check computational times


    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);
 
    MPI_Finalize();
    return 0;
}
