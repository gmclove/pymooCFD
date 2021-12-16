#Makefile

# parameters
include $(YALES2_HOME)/src/Makefile.in

PROG2MAKE=2D_cylinder

# defaults
defaults: $(PROG2MAKE)

# Libraries
LINKS = -L$(YALES2_LIB) -lyales2main $(ADD_LIBS)

# Program rule
$(PROG2MAKE): %: %.o
	$(F90) $(FLD) -I. -I$(YALES2_LIB) -o $@ $@.o $(LINKS)

# Generic rules
%.o: %.f90
	$(F90) $(FFLAGS) -I. -I$(YALES2_LIB) -c $*.f90

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c

# clean
veryclean: clean
	rm -f dump/*

clean:
	rm -f $(PROG2MAKE) *.o *.mod *.log .gdb_history
