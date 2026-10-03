# mlx-container-agent

A simple, fast, macOS-only agent. It runs a local LLM with [MLX](https://github.com/ml-explore/mlx) and runs the LLM's shell commands in a disposable Linux VM with [apple/container](https://github.com/apple/container). It works with any `mlx-lm` model that supports tool calling.

## How it works

- At startup, one container is started (`alpine:latest` by default).
- The model has a single tool, `run`, which executes `sh -c <command>` in the container through `container exec` and returns the exit code and output.
- The container lasts for the whole session and is force-deleted on exit.

## Requirements

- An Apple silicon Mac running macOS 26 or later
- [apple/container](https://github.com/apple/container/releases)
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
git clone https://github.com/pwngd/mlx-container-agent.git
```

```bash
cd mlx-container-agent
```

```bash
uv sync
```

## Run

```bash
uv run main.py
```

```bash
uv run main.py "install python and write a fizzbuzz script"
```

The first run downloads the model weights and a Linux kernel for the container VM.

In the chat, `/help` lists the commands: `/info`, `/sh <cmd>`, `/clear` and `/exit`. Ctrl-C interrupts a reply.

## Config

Settings live in `config.toml`: the model, sampling, the container image and resources, and the agent's limits. Command-line flags override it:

```bash
uv run main.py --cpus 4 --memory 4G --config my.toml
```
