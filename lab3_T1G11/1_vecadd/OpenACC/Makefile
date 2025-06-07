CC = nvc
CFLAGS = -acc=gpu -Minfo=all
OBJ = vecadd_oacc

all:
	$(CC) $(CFLAGS) $(OBJ).c -mp -o $(OBJ)

clean:
	rm $(OBJ)
