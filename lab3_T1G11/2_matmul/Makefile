CC=nvcc
CFLAGS=-lcublas -O2 -Xcompiler -Wall -std=c++11
OBJ=matmul

all:
	$(CC) $(CFLAGS) $(OBJ).cu -o $(OBJ) 

clean:
	rm $(OBJ)
