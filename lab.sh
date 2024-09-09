#!/usr/bin/env bash

# so that tqdm doesn't flood the screen with the warning
jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10
