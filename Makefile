# Keep It Warm: common commands. Run `make help` for the list.
PY ?= python

.PHONY: help world test fake analyze clean-fake

help:
	@echo "make world     rebuild the Incident Desk world + tasks + sessions from seed 42"
	@echo "make test      run the test suite (no GPU)"
	@echo "make fake      dry-run every policy with the fake model + simulated cache (no GPU)"
	@echo "make analyze   tables + charts from runs/main into results/"
	@echo "GPU runs: see notebooks/ (Kaggle 2x T4) and scripts/run_matrix.py"

world:
	$(PY) -m env.make_tasks --seed 42
	$(PY) -m env.make_sessions

test:
	$(PY) -m pytest -q

fake:
	@for p in full window mask batch summary timestamp; do \
		rm -rf runs/fake_$$p; \
		$(PY) scripts/run_episodes.py --fake --policy $$p --out runs/fake_$$p > /dev/null; \
	done
	$(PY) scripts/compare_runs.py runs/fake_full runs/fake_window runs/fake_mask runs/fake_batch runs/fake_summary runs/fake_timestamp

analyze:
	$(PY) analysis/analyze.py

clean-fake:
	rm -rf runs/fake_*
