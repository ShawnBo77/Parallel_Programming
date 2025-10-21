DATASIZE = MEDIUMN
COMMON = common
CC = gcc
LDLIBS = -lm
LDFLAGS = -Wl,--allow-multiple-definition -fopenmp
CFLAGS = -g -O3 -mcmodel=medium -fopenmp -I$(COMMON) -D$(DATASIZE)

default: cg grade
	mv grade cg_grader

# You need the '-o' option to ensure that the output goes into the right file
# when the source file is in a different directory.
# See the bottom of the following GNU make manual page for more information:
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html
OUTPUT_OPTION = -o $@
OBJS = cg_impl.o \
       $(COMMON)/randdp.o \
       $(COMMON)/c_timers.o \
       $(COMMON)/wtime.o

cg: cg.o $(OBJS)
grade: grade.o $(OBJS) ref_cg.a def_cg.a

cg.o: cg.c globals.h
cg_impl.o: cg_impl.c globals.h

clean:
	-$(RM) *.o *~
	$(RM) $(COMMON)/*.o
	$(RM) cg_grader cg
