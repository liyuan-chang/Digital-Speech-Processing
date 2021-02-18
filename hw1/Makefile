.PHONY: all clean run
# CC and CFLAGS are varilables
CC = g++
# -c option ask g++ to compile the source files, but do not link.
# -g option is for debugging version
# -O3 option is for optimized version
DBGFLAGS = -g -Wall -std=c++11 -D_DEBUG_ON_
OPTFLAGS = -O3 -std=c++11
TARGET = train test
TRAIN_ITER = 100

all : $(TARGET)

# optimized version
train : train.o
	$(CC) $(OPTFLAGS) $< -o $@

train.o : src/train.cpp
	$(CC) -c $(OPTFLAGS) $< -o $@

test : test.o
	$(CC) $(OPTFLAGS) $< -o $@

test.o : src/test.cpp
	$(CC) -c $(OPTFLAGS) $< -o $@

# clean all the .o and executable files
clean:
	rm -rf *.o $(TARGET) 

