COMMON_REPO = ./
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

#sw_emu , hw_emu
TARGETS := hw_emu
TARGET := $(TARGETS)
DEVICES := xilinx_aliyun-f3_dynamic_5_0
#DEVICES := xilinx_vcu1525_dynamic_5_1
DEVICE := $(DEVICES)
XCLBIN := ./xclbin
DSA := $(call device2sandsa, $(DEVICE))

CXX := $(XILINX_SDX)/bin/xcpp
XOCC := $(XILINX_SDX)/bin/xocc

CXXFLAGS := $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14
LDFLAGS := $(opencl_LDFLAGS)

HOST_SRCS = src/host.cpp
ddr_banks := 4

# Host compiler global settings
CXXFLAGS = -I $(XILINX_SDX)/runtime/include/1_2/ -I/-$(XILINX_SDX)/Vivado_HLS/include/ -O0 -g -Wall -fmessage-length=0 -std=c++14
CXXFLAGS += -I $(ABS_COMMON_REPO)/libs/xcl2

# [thunderGP]
#CXXFLAGS += -I/$(XILINX_SDX)/Vivado_HLS/include/ -O3 -g -fmessage-length=0 -std=c++14 -Wno-deprecated-declarations
LDFLAGS = -lOpenCL -lpthread -lrt -lstdc++ -L$(XILINX_SDX)/runtime/lib/x86_64

# Kernel compiler global settings
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps -O3 --xp param:compiler.acceleratorBinaryContent=dcp
#CLFLAGS +=  --max_memory_ports gs_top
CLFLAGS += -I src/fpga

# Kernel linker flags  ============need change===================
# LDCLFLAGS = --xp prop:solution.kernel_compiler_margin=10% --kernel_frequency=250
# LDCLFLAGS    += --nk  gs_top:1
# LDCLFLAGS    += --sp  gs_top_1.edgeSrcArray:bank0
# LDCLFLAGS    += --sp  gs_top_1.vertexValue:bank0
# LDCLFLAGS    += --sp  gs_top_1.edgeDstArray:bank0
# LDCLFLAGS    += --sp  gs_top_1.tmpvertexValue:bank0

# LDCLFLAGS += --profile_kernel data:all:all:all
#LDCLFLAGS    += --slr gs_top_1:SLR0

xcl2_SRCS:=${COMMON_REPO}/libs/xcl2/xcl2.cpp
xcl2_HDRS:=${COMMON_REPO}/libs/xcl2/xcl2.hpp
xcl2_CXXFLAGS:=-I ${COMMON_REPO}/libs/xcl2

CXXFLAGS += ${xcl2_CXXFLAGS}
HOST_SRCS += ${xcl2_SRCS}
HOST_HDRS += ${xcl2_HDRS}

EXECUTABLE = host

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/gs_top.$(TARGET).$(DSA).xclbin
BINARY_CONTAINER_OBJS += $(XCLBIN)/gs_top.$(TARGET).$(DSA).xo

#DEVICE_FPGA = $(xclbin)/gs_top.$(TARGET).$(DSA).xclbin
BINARY_CONTAINERS = $(XCLBIN)/gs_top.$(TARGET).xclbin
CP = cp -rf
DATA = ./data

.PHONY: all clean cleanall docs emconfig host
all: $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig run

run: $(EXECUTABLE) $(BINARY_CONTAINERS)
	XCL_EMULATION_MODE=${TARGET} ./$(EXECUTABLE) ${BINARY_CONTAINERS}

.PHONY: exe
exe: emconfig $(EXECUTABLE) $(BINARY_CONTAINERS) run

# Building kernel
host:$(EXECUTABLE)

$(XCLBIN)/gs_top.$(TARGET).xo: ./src/fpga/gs_top.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k gs_top -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/gs_top.$(TARGET).xclbin: $(XCLBIN)/gs_top.$(TARGET).xo
	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) -o'$@' $(+)

# Building Host
$(EXECUTABLE): $(HOST_SRCS) $(HOST_HDRS)
	mkdir -p $(XCLBIN)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) ./_x
	-$(RMDIR) ./output.bmp 
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""

docs: README.md

README.md: description.json
	$(ABS_COMMON_REPO)/utility/readme_gen/readme_gen.py description.json
