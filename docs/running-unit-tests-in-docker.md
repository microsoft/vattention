# Running Unit Tests in Docker

This document explains how to run the `sarathi-lean` unit tests inside the project Docker container.

It assumes the multiuser Docker setup described in [docker-multiuser.md](/home/anodyine/repos/vattention/docs/docker-multiuser.md) is already in place and that your workspace is mounted into the container at `/workspace`.

## Current Test Location

The current unit tests live under:

- `/workspace/sarathi-lean/tests`

The first test file added for MLA-related config validation is:

- `/workspace/sarathi-lean/tests/test_config_cache_architecture.py`

## Recommended Container

Use your existing project container, for example:

- `vattn-anodyine`

If your container is stopped, start it first:

```bash
docker start vattn-anodyine
```

## Run All `sarathi-lean` Unit Tests

From the host machine, run:

```bash
docker exec -w /workspace vattn-anodyine python -m unittest discover -s sarathi-lean/tests
```

This runs all unit tests currently present in `sarathi-lean/tests`.

## Run a Single Test File

To run only the cache-architecture tests:

```bash
docker exec -w /workspace vattn-anodyine python -m unittest sarathi-lean/tests/test_config_cache_architecture.py
```

## Expected Result

For the current config-helper test suite, a successful run should look like:

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK
```

## Why Run in Docker

The `sarathi` codebase depends on the runtime environment provided by the project container, including:

- the correct Python environment
- the installed `torch` version
- the expected package layout for the repo

Running tests inside the container is the preferred validation path because it verifies behavior in the same environment used for development and execution.

## Notes

- These tests are currently written with Python `unittest`.
- The test harness loads `sarathi/config.py` directly to avoid unrelated package-import side effects.
- As more MLA work is added, new unit tests should be added under `sarathi-lean/tests` and run with the same `docker exec ... python -m unittest discover ...` command.
