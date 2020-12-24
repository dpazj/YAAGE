
CXX = g++

CFLAGS = -Wall -Werror
CPPFLAGS = $(CFLAGS)

LDFLAGS = 


OBJ=main.o \
	src/matrix.o \
	src/node.o \



%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

test: $(OBJ)
	$(CXX) $(OBJ) -o main $(LDFLAGS) 

run:
	./main

clean:
	rm -f $(OBJ) main








