---
name: document-perf-test
description: Your goal is to take python script command for a perf test and make an entry to the ~/experiments/perf_tests.jsonl file about the results.
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, Task
allowed-prompts:
  - run python scripts
  - run perf tests
  - check build configuration
  - get hardware info with tt-smi
  - get git commit info
  - get current date
  - install python packages
---
You need to gather information about the given experiment and its performance results.
Steps:
1. run the given python script with its arguments and dump results to a log file e.g. `source venv/activate && python3 path/to/script.py --args &> script_log.log` (add dumping to log and env activation if it didn't exist. Important: don't add source venv/bin/activate, but exactly source venv/activate)
3. Parse the necessary information from the logfile and the script command.
2. append new entry to the ~/experiments/perf_tests.jsonl file. Create ~/experiments/perf_tests.jsonl file if it doesn't exist, same for the experiments folder.

Prerequisite: Make sure that tt-xla repo is built in release mode! Check this using `grep CMAKE_BUILD_TYPE build/CMakeCache.txt`. If it is not built in release mode, use `/clean-build-xla release` command to build it in release mode. Measurements are not representative in debug mode. 

Necessary fields to extract:
1. Script name - you have this as input already (in example path/to/script.py)
2. Script args - these determine what specific configuration was run. (--args)
3. Error - whether the script produced an error. A way to check this is to grep error in the log. If it did, copy paste the relevant part of the error to this field. Feel free to add a lot of context about the error, for easier reprduction. If it didn't, feel this field empty. If you got an error, other fields can be empty.
4. Hardware architecture - which TT chip are you using (e.g. n150, n300, p100a, ...). You get this by using tt-smi -s | grep "board_type". If tt-smi is not installed, use pip install tt-smi
5. e2e inference time - This can be extracted from the log. You should grep keyword "time" and see the results. Pick the thing that is e2e single forward pass inference time, I trust you can conclude that.
6. other times - these are all the other inference times you gather. Put them in this field.
7. Date - use linux date command
8. tt-xla commit sha - this is for reproducibility, use git rev-parse HEAD in tt-xla repo root. 

These are all the fields you need to extract. Please organize these fields in the jsonl entry format and append the entry to the ~/experiments/perf_tests.jsonl. 

This is the template output format 
{"script_name": "<script_name>.py", "script_args": ["--arg1", "<value>", "--arg2", "<value>"], "error": <error_string>, "hardware_architecture": "<board_type>", "e2e_inference_time_ms": <float_value>, "other_times": <string dump, free form>, "date": "e.g. 31. Jan 2026.", "tt_xla_commit_sha": "<git_sha>"}

Please run the experiment script, gather all the information I told you and log this entry to the perf tests. 