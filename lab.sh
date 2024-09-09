#!/usr/bin/env bash

# removes annoying tqdm spam
jupyter lab --ServerApp.iopub_msg_rate_limit=1.0e10
