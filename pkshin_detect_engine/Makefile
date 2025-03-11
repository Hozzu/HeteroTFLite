ROOT_DIR := .
OBJ_DIR := obj
SRC_DIR := src
TARGET := pkshin_detect

CXX := g++
CXXFLAGS := -mcpu=cortex-a76 -O2
LDFLAGS := -Wl,--rpath=$(ROOT_DIR)/lib \
-lpkshin_engine \
-ltensorflowlite -ltensorflowlite_gpu_delegate -ltensorflowlite_hexagon_delegate \
-lmaccel \
-lhailort \
-lais_client -lfastcvopt -ljson-c -ljpeg -fopenmp \
-lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_imgcodecs

INCS := -I $(ROOT_DIR)/include
LIBS := -L $(ROOT_DIR)/lib

SRCS := main.cpp run_image.cpp hailo_post/yolo_hailortpp.cpp
HDRS := $(shell find $(SRC_DIR) -name '*.hpp')
OBJS := $(SRCS:%.cpp=%.o)
OBJECTS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
DEPS = $(OBJECTS:.o=.d)

.PHONY: all clean

all: $(TARGET) 
	@echo The build completed successfully

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDRS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -c $< -o $@ -MD $(LDFLAGS)

$(TARGET): $(OBJECTS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) $(OBJECTS) -o $(TARGET) $(LDFLAGS) 
 
clean:
	rm -f $(TARGET)
	rm -f $(OBJECTS)
	rm -f $(DEPS)

-include $(DEPS)
