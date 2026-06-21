# Ablation-study pipeline.
#
#   make run-ablation     solve each model with each solver x ablation
#   make plot             draw runtime ECDFs and transform-size boxplots
#   make all              run-ablation then plot
#   make clean            remove generated results and figures (not models)
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

ILP_SOLVERS   ?= gurobi pysat highs
CP_SOLVERS    ?= choco ortools

# ILP solvers: baseline, no-ilpfriendly, no-detect-categorical
# CP solvers: baseline, no-cp-friendly
ILP_ABLATIONS ?= baseline no-ilpfriendly no-detect-categorical
CP_ABLATIONS  ?= baseline no-cp-friendly

TIME_LIMIT   ?= 30

# 1 = run transform only (--stop-after-transform); 0 = solve as usual
STOP_AFTER_TRANSFORM ?= 0

CPMPY_DIR    := ../../cpmpy
MODELS_DIR   ?= models
RESULTS_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),run-ablation-stats,run-ablation)
FIGURES_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),figures-stats,figures)

CPMPY_DEPS   := $(shell find $(CPMPY_DIR) -type f 2>/dev/null)

MODEL_PICKLES := $(wildcard $(MODELS_DIR)/*.pickle)

# run-ablation/<model>__<solver>__<ablation>.json
RESULTS := $(foreach s,$(ILP_SOLVERS),$(foreach a,$(ILP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(MODEL_PICKLES)))) \
           $(foreach s,$(CP_SOLVERS),$(foreach a,$(CP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(MODEL_PICKLES))))

PLOT_STAMP := $(FIGURES_DIR)/.plot.stamp

ablate_arg = $(if $(filter baseline,$(1)),,--ablate=$(1))
stop_after_transform_arg = $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),--stop-after-transform,)

.PHONY: all run-ablation plot clean
.DELETE_ON_ERROR:

all:
	$(MAKE) run-ablation STOP_AFTER_TRANSFORM=$(STOP_AFTER_TRANSFORM) \
		MODELS_DIR=$(MODELS_DIR) RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) plot RESULTS_DIR=$(RESULTS_DIR) FIGURES_DIR=$(FIGURES_DIR)

run-ablation: $(RESULTS)
plot: $(PLOT_STAMP)

define ABLATION_RULE
$(RESULTS_DIR)/%__$(1)__$(2).json: $(MODELS_DIR)/%.pickle run_model.py $(CPMPY_DEPS)
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) run_model.py $$< $(1) $(call ablate_arg,$(2)) $(stop_after_transform_arg) --out=$$@ time_limit=$(TIME_LIMIT)
endef

$(foreach s,$(ILP_SOLVERS),$(foreach a,$(ILP_ABLATIONS),$(eval $(call ABLATION_RULE,$(s),$(a)))))
$(foreach s,$(CP_SOLVERS),$(foreach a,$(CP_ABLATIONS),$(eval $(call ABLATION_RULE,$(s),$(a)))))

$(PLOT_STAMP): $(RESULTS) plot_ablation.py
	$(PYTHON) plot_ablation.py $(RESULTS_DIR) $(FIGURES_DIR)
	@touch $@

clean:
	rm -rf run-ablation run-ablation-stats figures figures-stats
