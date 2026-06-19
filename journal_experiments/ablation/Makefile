# Ablation-study pipeline.
#
#   make extract-models   dump every csplib example into a pickled model
#   make run-ablation     solve each model with each solver x ablation
#   make plot             draw runtime ECDFs and transform-size ECDFs
#   make all              do all of the above
#   make clean            remove every generated artifact
#
# Pass STOP_AFTER_TRANSFORM=1 to run the transform pipeline only (no solve)
# and write n_constraints / n_integer / n_boolean to JSON instead:
#
#   make run-ablation STOP_AFTER_TRANSFORM=1
#   make plot RESULTS_DIR=run-ablation-stats FIGURES_DIR=figures-stats
#
# Plain file rules + parallel make, e.g. `make -j8 run-ablation`.
# Add `-k` to keep going past a failing run. Stages are decoupled: extraction
# is best-effort (some examples can't be captured); run-ablation and plot use
# whichever pickles were produced. Run in order, or `make all`.

PYTHON       ?= python3

ILP_SOLVERS   ?= gurobi pysat highs
CP_SOLVERS    ?= choco ortools

# ILP solvers: baseline, no-ilpfriendly, no-detect-categorical
# CP solvers: baseline, ilpfriendly
ILP_ABLATIONS ?= baseline no-ilpfriendly no-detect-categorical
CP_ABLATIONS  ?= baseline no-cp-friendly

TIME_LIMIT   ?= 30

# 1 = run transform only (--stop-after-transform); 0 = solve as usual
STOP_AFTER_TRANSFORM ?= 0

EXAMPLES_DIR := ../../examples/csplib
CPMPY_DIR    := ../../cpmpy
MODELS_DIR   ?= models
RESULTS_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),run-ablation-stats,run-ablation)
FIGURES_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),figures-stats,figures)

CPMPY_DEPS   := $(shell find $(CPMPY_DIR) -type f 2>/dev/null)

MODEL_SCRIPTS    := $(shell grep -lE '\.solve' $(EXAMPLES_DIR)/*.py)
MODEL_PICKLES    := $(patsubst $(EXAMPLES_DIR)/%.py,$(MODELS_DIR)/%.pickle,$(MODEL_SCRIPTS))
EXISTING_PICKLES := $(wildcard $(MODELS_DIR)/*.pickle)

# run-ablation/<model>__<solver>__<ablation>.json
RESULTS := $(foreach s,$(ILP_SOLVERS),$(foreach a,$(ILP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(EXISTING_PICKLES)))) \
           $(foreach s,$(CP_SOLVERS),$(foreach a,$(CP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(EXISTING_PICKLES))))

PLOT_STAMP := $(FIGURES_DIR)/.plot.stamp

ablate_arg = $(if $(filter baseline,$(1)),,--ablate=$(1))
stop_after_transform_arg = $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),--stop-after-transform,)

.PHONY: all extract-models run-ablation plot clean
.DELETE_ON_ERROR:

# Sub-makes re-read which pickles exist after extraction.
all:
	-$(MAKE) extract-models
	$(MAKE) run-ablation STOP_AFTER_TRANSFORM=$(STOP_AFTER_TRANSFORM) \
		RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) plot RESULTS_DIR=$(RESULTS_DIR) FIGURES_DIR=$(FIGURES_DIR)

# extract-models: $(MODEL_PICKLES)
run-ablation: $(RESULTS)
plot: $(PLOT_STAMP)

# $(MODELS_DIR)/%.pickle: $(EXAMPLES_DIR)/%.py dump_model.py
# 	$(PYTHON) dump_model.py --out=$(MODELS_DIR) $<

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
	rm -rf $(MODELS_DIR) $(RESULTS_DIR) $(FIGURES_DIR)
