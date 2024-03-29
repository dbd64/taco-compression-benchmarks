BENCHES := ""
BENCHFLAGS := #"--benchmark-group-by=func"

# Pytest Specific Flags
#IGNORE := numpy/image.py
IGNORE += taco 
IGNORE_FLAGS := $(addprefix --ignore=,$(IGNORE)) 

benches_name := $(patsubst %.py,%,$(BENCHES))
benches_name := $(subst /,_,$(benches_name))
benches_name := $(subst *,_,$(benches_name))

# Taco Specific Flags
ifeq ($(TACO_OUT),)
TACO_OUT := results/taco/$(benches_name)benches_$(shell date +%Y_%m_%d_%H%M%S).csv
endif

# Set LANKA=ON if compiling on the MIT Lanka cluster.
ifeq ($(LANKA),)
LANKA := "OFF"
endif

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifeq ($(BUILD_DIR),)
BUILD_DIR := "$(ROOT_DIR)/taco/build"
endif

ifeq ($(TACO_SRC_DIR),)
TACO_SRC_DIR := "$(ROOT_DIR)/taco/"
endif

ifeq ("$(shell nproc)","")
NPROC_VAL := $(shell sysctl -n hw.logicalcpu)
else
NPROC_VAL := $(shell nproc)
endif

ifeq ("$(LANKA)","ON")
CMD := LD_LIBRARY_PATH=$(BUILD_DIR)/lib/:$(LD_LIBRARY_PATH) numactl -C 0 -m 0 $(BUILD_DIR)/taco-bench $(BENCHFLAGS)
MAKE_CMD := $(MAKE) taco-bench -j$(NPROC_VAL)
else
CMD := LD_LIBRARY_PATH=$(BUILD_DIR)/lib/:$(LD_LIBRARY_PATH) $(BUILD_DIR)/taco-bench $(BENCHFLAGS)
MAKE_CMD := $(MAKE) taco-bench -j$(NPROC_VAL)
endif

export TACO_TENSOR_PATH = data/

taco-bench: taco/build/taco-bench
	$(CMD)

taco-bench-nodep:
	$(CMD)

# Separate target to run the TACO benchmarks with numpy-taco cross validation logic.
validate-taco-bench: taco/build/taco-bench validation-path
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_repetitions=1
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_repetitions=1
endif

.PHONY: validation-path
validation-path:
ifeq ($(VALIDATION_OUTPUT_PATH),)
	$(error VALIDATION_OUTPUT_PATH is undefined)
endif

taco/build/taco-bench: results check-and-reinit-submodules taco/benchmark/googletest
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake -DLANKA=$(LANKA) $(TACO_SRC_DIR) && $(MAKE_CMD)

taco/benchmark/googletest: check-and-reinit-submodules
	if [ ! -d "taco/benchmark/googletest" ] ; then git clone https://github.com/google/googletest taco/benchmark/googletest; fi

opencv: check-and-reinit-submodules
	mkdir -p opencv/build && \
	mkdir -p opencv/install && \
	cd opencv/build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../install .. && \
	make -j$(NPROC_VAL) && \
	make install

opencv_bench/build/opencv_bench: opencv
	mkdir -p opencv_bench/build && \
	cd opencv_bench/build && \
	CMAKE_PREFIX_PATH=../../opencv/install/ cmake .. && \
	make -j$(NPROC_VAL)

.PHONY: results
results:
	mkdir -p results/taco

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
