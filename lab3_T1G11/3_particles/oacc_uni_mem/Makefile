CC = nvc
CFLAGS = -acc=gpu -Minfo=all -lm
OBJ = partis_oacc_uni_mem

all:
	$(CC) $(CFLAGS) -gpu=cc90 $(OBJ).c -mp  -o $(OBJ)

clean:
	rm $(OBJ)
