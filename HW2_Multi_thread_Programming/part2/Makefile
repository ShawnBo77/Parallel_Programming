APP_NAME = mandelbrot
OBJDIR = objs
COMMONDIR = ./common

CXX = g++ -m64
CXXFLAGS = -I$(COMMONDIR) -I$(OBJDIR) -O3 -std=c++17 -Wall
LDLIBS = -lm -lpthread

PPM_CXX = $(COMMONDIR)/ppm.cpp
PPM_OBJ = $(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))


default: $(APP_NAME)

dirs:
	mkdir -p $(OBJDIR)/
.PHONY: dirs

clean:
	$(RM) -r $(OBJDIR) *.ppm *~ $(APP_NAME)
.PHONY: clean

OBJS = $(OBJDIR)/main.o $(OBJDIR)/mandelbrot_serial.o $(OBJDIR)/mandelbrot_thread.o $(PPM_OBJ)

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/cycle_timer.h
