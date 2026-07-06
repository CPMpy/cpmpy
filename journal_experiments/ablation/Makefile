# Ablation-study pipeline.
#
#   make run-ablation     solve each model with each solver x ablation
#   make plot             draw runtime ECDFs and transform-size boxplots
#   make all              run-ablation then plot
#   make clean            remove generated results and figures (not models)
#   make clean-errors     remove result JSONs that recorded an error (for re-run)
#
# Input models are read from MODELS_DIR (default: models/). Each *.pickle
# file is run against every solver x ablation combination.
#
#   make run-ablation MODELS_DIR=../../journal_experiments/xcsp3_models
#   make run-ablation MODELS_DIR=models RESULTS_DIR=my-runs
#
# Pass STOP_AFTER_TRANSFORM=1 to run the transform pipeline only (no solve)
# and write n_constraints / n_integer / n_boolean to JSON instead:
#
#   make run-ablation STOP_AFTER_TRANSFORM=1
#   make plot STOP_AFTER_TRANSFORM=1
#
# Plain file rules + parallel make, e.g. `make -j8 run-ablation`.
# Add `-k` to keep going past a failing run.

PYTHON       ?= python3

ILP_SOLVERS   ?= gurobi pysat
CP_SOLVERS    ?= 

# ILP solvers: baseline, no-ilpfriendly, no-detect-categorical, no-positive-decompositions
# CP solvers: baseline, no-cp-friendly, no-positive-decompositions
ILP_ABLATIONS ?= baseline no-ilpfriendly no-detect-categorical no-positive-decompositions
CP_ABLATIONS  ?= baseline no-cp-friendly no-positive-decompositions

TIME_LIMIT   ?= 3600
TIME_LIMIT_GRACE_PCT ?= 10   # extra wall-clock before SIGKILL (after TIME_LIMIT SIGTERM)
TIME_LIMIT_KILL_AFTER = $(shell k=$$(( $(TIME_LIMIT) * $(TIME_LIMIT_GRACE_PCT) / 100 )); echo $$(( k > 0 ? k : 1 )))
# GNU coreutils timeout: SIGTERM at TIME_LIMIT, then SIGKILL after grace period.
TIMEOUT_CMD  ?= timeout
MEMORY_LIMIT_GB ?= 8 # 8GB

# 1 = run transform only (--stop-after-transform); 0 = solve as usual
STOP_AFTER_TRANSFORM ?= 0

MODELS_DIR   ?= models
RESULTS_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),run-ablation-stats,run-ablation)
FIGURES_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),figures-stats,figures)

SOLVER_KWARGS_gurobi  ?= Threads=1
SOLVER_KWARGS_ortools ?= num_workers=1
SOLVER_KWARGS_exact   ?= lp=0.1
SOLVER_KWARGS_pysat   ?=

_ABLATION_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
_RUN_MODEL := $(_ABLATION_DIR)run_model.py
$(foreach s,$(ILP_SOLVERS) $(CP_SOLVERS),$(eval \
  KWARGS_SEGMENT_$(s) := $(shell $(PYTHON) $(_RUN_MODEL) --kwargs-segment $(SOLVER_KWARGS_$(s)))))

result_json_suffix = __$(1)$(if $(KWARGS_SEGMENT_$(1)),__$(KWARGS_SEGMENT_$(1)),)__$(2).json

MODEL_PICKLES := $(wildcard $(MODELS_DIR)/*.pickle)

# run-ablation/<model>__<solver>__[<kwargs>]__<ablation>.json
RESULTS := $(foreach s,$(ILP_SOLVERS),$(foreach a,$(ILP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%$(call result_json_suffix,$(s),$(a)),$(MODEL_PICKLES)))) \
           $(foreach s,$(CP_SOLVERS),$(foreach a,$(CP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%$(call result_json_suffix,$(s),$(a)),$(MODEL_PICKLES))))

PLOT_STAMP := $(FIGURES_DIR)/.plot.stamp

ablate_arg = $(if $(filter baseline,$(1)),,--ablate=$(1))
stop_after_transform_arg = $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),--stop-after-transform,)
run_model_args = --model-path=$$< --solver-name=$(1) $(call ablate_arg,$(2)) $(stop_after_transform_arg) --out=$$@ --time-limit=$(TIME_LIMIT) --memory-limit=$(MEMORY_LIMIT_GB) $(SOLVER_KWARGS_$(1))
run_with_timeout = $(TIMEOUT_CMD) --signal=TERM --kill-after=$(TIME_LIMIT_KILL_AFTER) $(TIME_LIMIT)

.PHONY: all run-ablation plot clean clean-errors
.DELETE_ON_ERROR:

all:
	$(MAKE) run-ablation STOP_AFTER_TRANSFORM=$(STOP_AFTER_TRANSFORM) \
		MODELS_DIR=$(MODELS_DIR) RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) plot RESULTS_DIR=$(RESULTS_DIR) FIGURES_DIR=$(FIGURES_DIR)

run-ablation: $(RESULTS)
plot: $(PLOT_STAMP)

define ABLATION_RULE
$(RESULTS_DIR)/%$(call result_json_suffix,$(1),$(2)): $(MODELS_DIR)/%.pickle
	@mkdir -p $(RESULTS_DIR)
	$(run_with_timeout) $(PYTHON) run_model.py $(call run_model_args,$(1),$(2)) \
		|| $(PYTHON) write_dummy_file.py $(call run_model_args,$(1),$(2))
endef

$(foreach s,$(ILP_SOLVERS),$(foreach a,$(ILP_ABLATIONS),$(eval $(call ABLATION_RULE,$(s),$(a)))))
$(foreach s,$(CP_SOLVERS),$(foreach a,$(CP_ABLATIONS),$(eval $(call ABLATION_RULE,$(s),$(a)))))

$(PLOT_STAMP): $(RESULTS) plot_ablation.py
	$(PYTHON) plot_ablation.py $(RESULTS_DIR) $(FIGURES_DIR)
	@touch $@

clean:
	rm -rf run-ablation run-ablation-stats figures figures-stats

clean-errors:
	$(PYTHON) clean_errors.py $(RESULTS_DIR)
