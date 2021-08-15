all : 
	cc -fPIC -shared -o liblebedevlaikov.so lebedev/lebedev.c

clean: 
	rm liblebedevlaikov.so
