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
MEMORY_LIMIT_GB ?= 8 # 8GB

# 1 = run transform only (--stop-after-transform); 0 = solve as usual
STOP_AFTER_TRANSFORM ?= 0

MODELS_DIR   ?= models
RESULTS_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),run-ablation-stats,run-ablation)
FIGURES_DIR  ?= $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),figures-stats,figures)

MODEL_PICKLES := $(wildcard $(MODELS_DIR)/*.pickle)

# run-ablation/<model>__<solver>__<ablation>.json
RESULTS := $(foreach s,$(ILP_SOLVERS),$(foreach a,$(ILP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(MODEL_PICKLES)))) \
           $(foreach s,$(CP_SOLVERS),$(foreach a,$(CP_ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(MODEL_PICKLES))))

PLOT_STAMP := $(FIGURES_DIR)/.plot.stamp

ablate_arg = $(if $(filter baseline,$(1)),,--ablate=$(1))
stop_after_transform_arg = $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),--stop-after-transform,)
ablate_json = $(if $(filter baseline,$(1)),null,"$(1)")
stop_after_transform_json = $(if $(filter 1,$(STOP_AFTER_TRANSFORM)),true,false)

.PHONY: all run-ablation plot clean clean-errors
.DELETE_ON_ERROR:

all:
	$(MAKE) run-ablation STOP_AFTER_TRANSFORM=$(STOP_AFTER_TRANSFORM) \
		MODELS_DIR=$(MODELS_DIR) RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) plot RESULTS_DIR=$(RESULTS_DIR) FIGURES_DIR=$(FIGURES_DIR)

run-ablation: $(RESULTS)
plot: $(PLOT_STAMP)

define ABLATION_RULE
$(RESULTS_DIR)/%__$(1)__$(2).json: $(MODELS_DIR)/%.pickle
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) run_model.py --model-path=$$< --solver-name=$(1) $(call ablate_arg,$(2)) $(stop_after_transform_arg) --out=$$@ --time-limit=$(TIME_LIMIT) --memory-limit=$(MEMORY_LIMIT_GB) \
		|| $(PYTHON) -c 'import json,sys; json.dump({"model": sys.argv[2]+".pickle","solver": sys.argv[3],"solver_kwargs": {},"time_limit": int(sys.argv[5]),"ablate": json.loads(sys.argv[4]),"stop_after_transform": json.loads(sys.argv[6]),"cpmpy_commit": None,"cpmpy_commit_message": None,"transformation_time": None,"runtime": None,"status": "error","objective_value": None,"error": "run_model.py exited nonzero (killed: OOM/segfault)"}, open(sys.argv[1],"w"), indent=2)' $$@ $$* $(1) '$(call ablate_json,$(2))' $(TIME_LIMIT) $(stop_after_transform_json)
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
