all: 
	g++ -Wall -fPIC -O2 -c ./libgdiam/gdiam.cpp -o gdiam.o 
	g++ -shared -fPIC gdiam.o -o libgdiam.so
	rm gdiam.o
	cc -fPIC -shared -o liblebedevlaikov.so lebedev/lebedev.c

clean: 
	rm liblebedevlaikov.so
	rm libgdiam.so
