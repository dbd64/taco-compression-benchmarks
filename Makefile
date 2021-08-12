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

ifeq ("$(LANKA)","ON")
CMD := LD_LIBRARY_PATH=taco/build/lib/:$(LD_LIBRARY_PATH) numactl -C 0 -m 0 taco/build/taco-bench $(BENCHFLAGS)
MAKE_CMD := $(MAKE) taco-bench -j48
else
CMD := LD_LIBRARY_PATH=taco/build/lib/:$(LD_LIBRARY_PATH) taco/build/taco-bench $(BENCHFLAGS)
MAKE_CMD := $(MAKE) taco-bench -j16
endif

export TACO_TENSOR_PATH = data/

taco-bench: taco/build/taco-bench
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
endif

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
	mkdir -p taco/build/ && cd taco/build/ && cmake -DLANKA=$(LANKA) ../ && $(MAKE_CMD)

taco/benchmark/googletest: check-and-reinit-submodules
	if [ ! -d "taco/benchmark/googletest" ] ; then git clone https://github.com/google/googletest taco/benchmark/googletest; fi

.PHONY: results
results:
	mkdir -p results/taco

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
