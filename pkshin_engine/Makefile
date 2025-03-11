ROOT_DIR := .
OBJ_DIR := obj
SRC_DIR := src
TARGET := lib/libpkshin_engine.so

CXX := g++
CXXFLAGS := -mcpu=cortex-a76 -O2 -shared -fPIC
LDFLAGS := -Wl,--rpath=$(ROOT_DIR)/lib \
-ltensorflowlite \
-lmaccel \
-lhailort

INCS := -I $(ROOT_DIR)/include
LIBS := -L $(ROOT_DIR)/lib

SRCS := engine.cpp
HDRS := $(shell find $(SRC_DIR) -name '*.hpp')
OBJS := $(SRCS:%.cpp=%.o)
OBJECTS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
DEPS = $(OBJECTS:.o=.d)

.PHONY: all clean

all: $(TARGET) 
	@echo The build completed successfully

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDRS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -c $< -o $@ $(LDFLAGS)

$(TARGET): $(OBJECTS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) $(OBJECTS) -o $(TARGET) $(LDFLAGS) 
 
clean:
	rm -f $(TARGET)
	rm -f $(OBJECTS)
	rm -f $(DEPS)