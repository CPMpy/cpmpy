# Ablation-study pipeline.
#
#   make extract-models   dump every csplib example into a pickled model
#   make run-ablation      solve each model with each solver x ablation
#   make plot              draw the runtime ECDFs from the run results
#   make all               do all of the above
#   make clean             remove every generated artifact
#
# The targets are plain file rules, so `make` only redoes the work whose inputs
# changed, and everything is independent so `make -j<threads>` runs in parallel,
# e.g.  `make -j8 run-ablation`.  Add `-k` to keep going past a failing run.
#
# The stages are decoupled on purpose: `extract-models` is best-effort over the
# example scripts (some can't be captured, e.g. those that solve via a raw
# solver-interface object), while `run-ablation` and `plot` work on whatever
# pickles were actually produced. Run them in order (or just `make all`).

PYTHON       ?= /Users/ignaceb/miniforge3/envs/cpmpy/bin/python

# Solvers to benchmark (only solvers with an ablation pipeline in run_model.py:
# gurobi, pysat, rc2, scip).
SOLVERS      ?= gurobi scip

# Ablation variants. "baseline" is the un-ablated run; the rest map to
# run_model.py's --ablate values.
ABLATIONS    ?= baseline no-ilpfriendly no-detect-categorical

# Per-solve wall-clock limit (seconds), forwarded to run_model.py.
TIME_LIMIT   ?= 30

EXAMPLES_DIR := ../examples/csplib
MODELS_DIR   := models
RESULTS_DIR  := run-ablation
FIGURES_DIR  := figures

# extract-models: one pickle per example script that solves a model. The grep
# only excludes scripts that clearly never solve anything; some matched scripts
# still can't be captured and are simply skipped at run time (see dump_model.py).
MODEL_SCRIPTS  := $(shell grep -lE '\.solve' $(EXAMPLES_DIR)/*.py)
MODEL_PICKLES  := $(patsubst $(EXAMPLES_DIR)/%.py,$(MODELS_DIR)/%.pickle,$(MODEL_SCRIPTS))

# run-ablation / plot work on the pickles that actually exist, so models that
# failed to extract don't break the downstream stages.
EXISTING_PICKLES := $(wildcard $(MODELS_DIR)/*.pickle)

# Every (solver, ablation, model) run produces one deterministic JSON file in
# a single flat directory: run-ablation/<model>__<solver>__<ablation>.json
RESULTS := $(foreach s,$(SOLVERS),$(foreach a,$(ABLATIONS),\
             $(patsubst $(MODELS_DIR)/%.pickle,$(RESULTS_DIR)/%__$(s)__$(a).json,$(EXISTING_PICKLES))))

PLOT_STAMP := $(FIGURES_DIR)/.plot.stamp

.PHONY: all extract-models run-ablation plot clean
.DELETE_ON_ERROR:

# Run the stages in order; each sub-make re-reads which pickles now exist.
# `-` lets extraction skip the handful of scripts it can't capture.
all:
	-$(MAKE) extract-models
	$(MAKE) run-ablation
	$(MAKE) plot

extract-models: $(MODEL_PICKLES)

run-ablation: $(RESULTS)

plot: $(PLOT_STAMP)

# --- model extraction: examples/csplib/<name>.py -> models/<name>.pickle ------
$(MODELS_DIR)/%.pickle: $(EXAMPLES_DIR)/%.py dump_model.py
	$(PYTHON) dump_model.py --out=$(MODELS_DIR) $<

# --- ablation runs: one pattern rule per (solver, ablation) -------------------
# $(1) = solver, $(2) = ablation label, $(3) = run_model.py flag (none for baseline)
define ABLATION_RULE
$(RESULTS_DIR)/%__$(1)__$(2).json: $(MODELS_DIR)/%.pickle run_model.py
	@mkdir -p $(RESULTS_DIR)
	$$(PYTHON) run_model.py $$< $(1) $(3) --out=$$@ time_limit=$$(TIME_LIMIT)
endef

$(foreach s,$(SOLVERS),$(foreach a,$(ABLATIONS),\
  $(eval $(call ABLATION_RULE,$(s),$(a),$(if $(filter baseline,$(a)),,--ablate=$(a))))))

# --- plotting: redraw whenever any run result (or the script) changes ---------
$(PLOT_STAMP): $(RESULTS) plot_ablation.py
	$(PYTHON) plot_ablation.py $(RESULTS_DIR) $(FIGURES_DIR)
	@mkdir -p $(FIGURES_DIR)
	@touch $@

clean:
	rm -rf $(MODELS_DIR) $(RESULTS_DIR) $(FIGURES_DIR)
