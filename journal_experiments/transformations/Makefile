# Transformation-pipeline benchmark.
#
#   make run          run the PySAT transform stack on every model
#   make plot         ECDF plots from JSON results
#   make all          run then plot
#   make clean        remove generated results and figures (not models)
#   make clean-errors remove JSONs that recorded an error (for re-run)
#
# Input models are read from MODELS_DIR (default: models/). Each *.pickle
# produces one JSON under RESULTS_DIR.
#
#   make run MODELS_DIR=../../journal_experiments/xcsp3_models
#   make run MODELS_DIR=models RESULTS_DIR=my-runs MEMORY_LIMIT_GB=16
#
# Plain file rules + parallel make, e.g. `make -j8 run`.
# Add `-k` to keep going past a failing run.

PYTHON          ?= python3

MEMORY_LIMIT_GB ?= 8
MEMORY_LIMIT_ARG  = $(if $(MEMORY_LIMIT_GB),--memory-limit=$(MEMORY_LIMIT_GB),)

MODELS_DIR      ?= models
RESULTS_DIR     ?= transformation_results
FIGURES_DIR     ?= figures

MODEL_PICKLES   := $(wildcard $(MODELS_DIR)/*.pickle)
RESULTS         := $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%.json,$(MODEL_PICKLES))
PLOT_STAMP      := $(FIGURES_DIR)/.plot.stamp

.PHONY: all run plot clean clean-errors
.DELETE_ON_ERROR:

all:
	$(MAKE) run MODELS_DIR=$(MODELS_DIR) RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) plot RESULTS_DIR=$(RESULTS_DIR) FIGURES_DIR=$(FIGURES_DIR)

run: $(RESULTS)
plot: $(PLOT_STAMP)

$(RESULTS_DIR)/%.json: $(MODELS_DIR)/%.pickle
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) experiment_transformation.py --model-path=$< --out=$@ $(MEMORY_LIMIT_ARG) \
		|| $(PYTHON) -c 'import json,sys; json.dump({"model": sys.argv[2]+".pickle","model_path": None,"cpmpy_commit": None,"cpmpy_commit_message": None,"stages": None,"total_runtime": None,"error": "experiment_transformation.py exited nonzero (killed: OOM/segfault)"}, open(sys.argv[1],"w"), indent=2)' $@ $*

$(PLOT_STAMP): $(RESULTS) plot_transformation.py
	$(PYTHON) plot_transformation.py $(RESULTS_DIR) $(FIGURES_DIR)
	@touch $@

clean:
	rm -rf transformation_results figures

clean-errors:
	@$(PYTHON) -c 'import glob,json,os,sys; \
d=sys.argv[1]; \
[os.remove(p) for p in glob.glob(os.path.join(d,"*.json")) if json.load(open(p)).get("error")] \
' $(RESULTS_DIR)
