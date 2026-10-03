"""Qwen sandbox agent.

Spec:
  - Runs Qwen3.8-27B (4-bit MLX) locally via mlx-lm.
  - Spins up a throwaway Linux container with apple/container for the model to
    work inside. The model gets one tool, `run`, which executes a shell command
    in that container and returns the output.
  - Checks that the `container` CLI is installed and its system service is
    running (starts it if not) before doing anything else.
  - One KV cache is kept across steps and messages, so each step only processes
    the new part of the conversation (prompt caching).
  - Interactive chat: the model and container stay alive across messages, so
    the agent remembers the conversation and everything it did in the container.
    A status line shows what the agent is doing; each reply ends with a summary.
    Slash commands (/help) let you inspect and use the container yourself.
  - The container is always force-deleted on exit: /exit, Ctrl-D, Ctrl-C at the
    prompt, SIGTERM, or an error. Ctrl-C during a reply only interrupts it.

Usage:
  uv run main.py                                   # start chatting
  uv run main.py "install python and run fizzbuzz"  # with a first message
  uv run main.py --memory 4G --cpus 4              # bigger container
  uv run main.py --config my.toml                  # use another config file

Settings live in config.toml next to this file (every key is optional;
see that file for the defaults). Command-line flags override it.
"""

import argparse
import json
import readline  # noqa: F401  (line editing and history for input())
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

CONFIG_PATH = Path(__file__).parent / "config.toml"

# Defaults for every setting; config.toml overrides any of them.
CONFIG = {
    "model": {
        "name": "mlx-community/Qwen3.8-27B-4bit",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 4096,  # per model response
        "thinking": False,  # reason before answering
        "reasoning_effort": "low",  # "low", "medium" or "xhigh"
        "keep_thinking": True,  # keep past reasoning in the history (lets the cache reuse it)
        "seed": None,  # None = random
        "system_prompt": "You are an autonomous agent with root access to a disposable "
        "Linux container. Use the `run` tool to execute shell commands there (on Alpine, "
        "install packages with `apk add`). Work step by step, check results, and fix "
        "errors yourself. When the task is complete, reply with a short summary and no "
        "tool call.",
    },
    "performance": {
        "prompt_cache": True,  # reuse already-processed history between steps
        "prefill_step_size": 2048,  # prompt tokens processed per batch
        "kv_bits": None,  # 4 or 8 quantizes the KV cache; None = full precision
        "kv_group_size": 64,
        "quantized_kv_start": 0,  # tokens before KV quantization kicks in
        "draft_model": None,  # speculative decoding; needs a trimmable-cache model
        "num_draft_tokens": 3,
    },
    "container": {
        "image": "alpine:latest",
        "cpus": None,  # None = container's default
        "memory": None,
    },
    "agent": {
        "max_steps": 30,  # tool calls per reply
        "command_timeout": 300,  # seconds per command
        "max_output": 4000,  # chars of command output fed back to the model
    },
    "display": {
        "output_lines": 15,  # lines of command output shown in the terminal
    },
}


def load_config(path: Path, required: bool) -> None:
    """Merge a TOML file into CONFIG. Unknown keys are an error to catch typos."""
    if not path.exists():
        if required:
            sys.exit(f"error: config file not found: {path}")
        return
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"error: {path}: {e}")
    for section, values in data.items():
        if section not in CONFIG or not isinstance(values, dict):
            sys.exit(f"error: {path}: unknown section [{section}]")
        for key, value in values.items():
            if key not in CONFIG[section]:
                sys.exit(f"error: {path}: unknown key {key!r} in [{section}]")
            CONFIG[section][key] = value
    effort = CONFIG["model"]["reasoning_effort"]
    if effort not in ("low", "medium", "xhigh"):
        sys.exit(f"error: {path}: reasoning_effort must be low, medium or xhigh, not {effort!r}")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Run a shell command in the sandbox container. "
            "Returns the exit code and combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run with sh -c."}
                },
                "required": ["command"],
            },
        },
    }
]

HELP = """\
  /info        container image, CPUs, memory and disk
  /sh <cmd>    run a command in the container yourself (the agent won't see it)
  /clear       start a new conversation (container is kept)
  /exit        quit and delete the container (also Ctrl-D)
  Ctrl-C       interrupt the agent's current reply"""


# --- terminal ----------------------------------------------------------------

_tty = sys.stdout.isatty()


def style(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty else text


def dim(t): return style("2", t)
def bold(t): return style("1", t)
def cyan(t): return style("36", t)
def green(t): return style("32", t)
def yellow(t): return style("33", t)
def red(t): return style("31", t)


_spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spin_i = 0


def status(text: str) -> None:
    """Draw a one-line spinner status, overwritten in place."""
    global _spin_i
    if not _tty:
        return
    _spin_i += 1
    width = shutil.get_terminal_size().columns - 1
    line = f"{_spinner[_spin_i % len(_spinner)]} {text}"[:width]
    print(f"\r\033[K{dim(line)}", end="", flush=True)


def clear_status() -> None:
    if _tty:
        print("\r\033[K", end="", flush=True)


def print_output(result: str) -> None:
    lines = result.splitlines()
    show = CONFIG["display"]["output_lines"]
    for line in lines[:show]:
        print(dim(f"  │ {line}"))
    if len(lines) > show:
        print(dim(f"  │ … {len(lines) - show} more lines"))


# --- container ---------------------------------------------------------------


def ensure_container_cli() -> None:
    if shutil.which("container") is None:
        sys.exit("error: `container` CLI not found. Install it from github.com/apple/container")
    if subprocess.run(["container", "system", "status"], capture_output=True).returncode != 0:
        print("Starting container system service...")
        # First run needs a Linux kernel; install apple/container's recommended default.
        subprocess.run(["container", "system", "start", "--enable-kernel-install"], check=True)


def start_container(name: str, image: str, cpus: str | None, memory: str | None) -> None:
    status(f"starting container ({image})…")
    cmd = ["container", "run", "-d", "--rm", "--name", name]
    if cpus:
        cmd += ["--cpus", str(cpus)]
    if memory:
        cmd += ["--memory", str(memory)]
    r = subprocess.run(cmd + [image, "tail", "-f", "/dev/null"], capture_output=True, text=True)
    clear_status()
    if r.returncode != 0:
        sys.exit(red(f"error: failed to start container:\n{r.stderr.strip()}"))


def exec_in_container(name: str, command: str) -> tuple[int, str]:
    timeout = CONFIG["agent"]["command_timeout"]
    # `timeout` runs inside the container so the command itself is killed;
    # killing only the local `container exec` client could leave it running.
    # `exec 2>&1` merges stderr into stdout in order (the exec client keeps them apart).
    cmd = ["container", "exec", name, "timeout", "-s", "KILL", str(timeout), "sh", "-c", "exec 2>&1\n" + command]
    start = time.monotonic()
    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",  # binary output must not crash the session
            timeout=timeout + 10,
        )
    except subprocess.TimeoutExpired:
        return -1, f"command timed out after {timeout}s"
    out = r.stdout.strip()
    if time.monotonic() - start >= timeout:
        out = f"{out}\ncommand timed out after {timeout}s".strip()
    return r.returncode, out


def container_info(name: str) -> str:
    r = subprocess.run(["container", "inspect", name], capture_output=True, text=True)
    if r.returncode != 0:
        return red(r.stderr.strip() or "container not found")
    info = json.loads(r.stdout)[0]
    config, res = info["configuration"], info["configuration"]["resources"]
    ip = info["status"]["networks"][0]["ipv4Address"].split("/")[0] if info["status"]["networks"] else "-"
    _, usage = exec_in_container(
        name,
        "free -m | awk '/Mem/{print $3\" MiB used / \"$2\" MiB\"}'; "
        "df -h / | awk 'NR==2{print $3\" used / \"$2}'",
    )
    mem, disk = (usage.splitlines() + ["?", "?"])[:2]
    rows = [
        ("container", f"{name} ({info['status']['state']}, {ip})"),
        ("image", config["image"]["reference"]),
        ("cpus", str(res["cpus"])),
        ("memory", f"{mem}  (allocated {res['memoryInBytes'] / 2**30:g} GiB)"),
        ("disk", disk),
    ]
    return "\n".join(f"  {dim(k.ljust(10))} {v}" for k, v in rows)


def delete_container(name: str) -> None:
    # Ignore Ctrl-C while cleaning up so a second press can't leave it behind.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    clear_status()
    print(dim(f"\nDeleting container {name}…"))
    subprocess.run(["container", "delete", "--force", name], capture_output=True)


# --- agent -------------------------------------------------------------------


def split_output(text: str, thinking: bool, tool_start: str) -> tuple[str, str, str]:
    """Split raw model output into (phase, reasoning, answer)."""
    reasoning = ""
    if thinking or text.lstrip().startswith("<think>"):
        if "</think>" not in text:
            return "thinking", text.replace("<think>", ""), ""
        reasoning, text = text.split("</think>", 1)
    if tool_start in text:
        return "tool", reasoning, text.split(tool_start, 1)[0].lstrip()
    return "answer", reasoning, text.lstrip()


class PromptCache:
    """One KV cache kept across model calls, keyed by the tokens it has processed.

    Each step's prompt is the previous prompt plus the model's reply plus new
    messages, so usually only the new part needs processing. Hybrid models
    (like Qwen3.8) have caches that can't be rewound: if the history diverges
    there, the cache is rebuilt from scratch.
    """

    def __init__(self, model, draft_model, enabled: bool):
        self.models = [m for m in (model, draft_model) if m is not None]
        self.enabled = enabled
        self.cache, self.tokens, self.miss = None, [], None

    def prepare(self, prompt: list[int]) -> tuple[list, list[int]]:
        """Return (cache, tokens still to process) for this prompt.

        If the history diverged from the cache and it had to be rebuilt, sets
        self.miss to (position, cached tokens from there) for debugging.
        """
        n, self.miss = 0, None
        if self.enabled and self.cache is not None:
            limit = min(len(self.tokens), len(prompt) - 1)  # always process at least one token
            while n < limit and self.tokens[n] == prompt[n]:
                n += 1
            if n < len(self.tokens):
                if n > 0 and can_trim_prompt_cache(self.cache):
                    trim_prompt_cache(self.cache, len(self.tokens) - n)
                else:
                    self.miss = (n, self.tokens[n : n + 12])
                    n = 0
        if n == 0:
            self.cache = [c for m in self.models for c in make_prompt_cache(m)]
        self.tokens = prompt[:]
        return self.cache, prompt[n:]

    def reset(self) -> None:
        self.cache, self.tokens = None, []


class LLM:
    def __init__(self):
        m, p = CONFIG["model"], CONFIG["performance"]
        if m["seed"] is not None:
            mx.random.seed(m["seed"])
        self.model, self.tokenizer = load(m["name"])
        if not self.tokenizer.has_tool_calling:
            sys.exit(f"error: {m['name']} has no tool-calling support in mlx-lm")
        self.draft_model = None
        if p["draft_model"]:
            if not can_trim_prompt_cache(make_prompt_cache(self.model)):
                sys.exit(
                    f"error: {m['name']} can't use a draft model (its cache can't be "
                    "rewound); remove performance.draft_model from the config"
                )
            print(dim(f"Loading draft model {p['draft_model']}…"))
            self.draft_model, _ = load(p["draft_model"])
        self.sampler = make_sampler(temp=m["temperature"], top_p=m["top_p"], top_k=m["top_k"])
        self.cache = PromptCache(self.model, self.draft_model, p["prompt_cache"])
        self.gen_kwargs = {
            "max_tokens": m["max_tokens"],
            "sampler": self.sampler,
            "draft_model": self.draft_model,
            "prefill_step_size": p["prefill_step_size"],
            "kv_bits": p["kv_bits"],
            "kv_group_size": p["kv_group_size"],
            "quantized_kv_start": p["quantized_kv_start"],
        }
        if self.draft_model is not None:
            self.gen_kwargs["num_draft_tokens"] = p["num_draft_tokens"]
        self.template_kwargs = {
            "enable_thinking": m["thinking"],
            "reasoning_effort": m["reasoning_effort"],
            "preserve_thinking": m["keep_thinking"],
        }


def generate(llm: LLM, messages: list, stats: dict) -> tuple[str, bool, str]:
    """Stream one model response: spinner while thinking, answer text as it arrives.

    Returns (raw text, whether it started in thinking mode, finish reason) and
    adds generated tokens, generation time and context size to stats.
    """
    tokenizer = llm.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, tokenize=False, **llm.template_kwargs
    )
    thinking = prompt.rstrip().endswith("<think>")
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    cache, todo = llm.cache.prepare(prompt_tokens)
    if llm.cache.miss:
        n, old = llm.cache.miss
        new = prompt_tokens[n : n + len(old)]
        print(dim(f"  (prompt cache miss at token {n}: cached {tokenizer.decode(old)!r}, "
                  f"history now {tokenizer.decode(new)!r})"))
    cached = len(prompt_tokens) - len(todo)
    tool_start = tokenizer.tool_call_start
    text, shown, start = "", 0, time.monotonic()
    status(f"reading {len(todo)} tokens" + (f" ({cached} cached)…" if cached else "…"))
    try:
        for n, chunk in enumerate(
            stream_generate(llm.model, tokenizer, todo, prompt_cache=cache, **llm.gen_kwargs), 1
        ):
            llm.cache.tokens.append(chunk.token)
            text += chunk.text
            phase, reasoning, answer = split_output(text, thinking, tool_start)
            if phase == "thinking":
                last = " ".join(reasoning.split())[-80:]
                status(f"thinking · {n} tok · {time.monotonic() - start:.0f}s · {last}")
                continue
            # Hold back a few chars in case they're the start of a tool-call tag.
            end = len(answer) if phase == "tool" else max(shown, len(answer) - len(tool_start))
            if end > shown:
                if shown == 0:
                    clear_status()
                    print("● ", end="")
                print(answer[shown:end], end="", flush=True)
                shown = end
            if phase == "tool":
                status("writing command…")
    except BaseException:
        # An interrupted prefill or generation leaves the cache half-updated.
        llm.cache.reset()
        raise
    if text and chunk.generation_tps:
        stats["tokens"] += chunk.generation_tokens
        stats["gen_time"] += chunk.generation_tokens / chunk.generation_tps
    _, _, answer = split_output(text, thinking, tool_start)
    if len(answer) > shown:
        if shown == 0:
            clear_status()
            print("● ", end="")
        print(answer[shown:], end="")
        shown = len(answer)
    clear_status()
    if shown:
        print()
    stats["context"] = len(llm.cache.tokens)
    return text, thinking, chunk.finish_reason


def parse_tool_calls(tokenizer, text: str) -> tuple[list[dict], list[str]]:
    """Return (parsed tool calls, errors for malformed ones) found in text."""
    start, end = tokenizer.tool_call_start, tokenizer.tool_call_end
    calls, errors = [], []
    for block in text.split(start)[1:]:
        body, closed, _ = block.partition(end)
        try:
            if not closed:
                raise ValueError(f"missing {end}")
            parsed = tokenizer.tool_parser(body, TOOLS)
        except Exception as e:  # parsers raise various errors on bad input
            errors.append(f"could not parse tool call ({e}); use the exact tool-call format")
            continue
        calls.extend(parsed if isinstance(parsed, list) else [parsed])
    return calls, errors


def agent_turn(llm: LLM, name: str, messages: list, max_steps: int, stats: dict) -> int:
    """Let the model work until it replies without a tool call. Returns commands run."""
    tokenizer = llm.tokenizer
    max_output = CONFIG["agent"]["max_output"]
    commands = 0
    for _ in range(max_steps):
        text, thinking, finish = generate(llm, messages, stats)
        phase, reasoning, content = split_output(text, thinking, tokenizer.tool_call_start)
        calls, errors = [], []
        if phase != "thinking":  # tool calls only count after the reasoning
            calls, errors = parse_tool_calls(tokenizer, text.rpartition("</think>")[2])
        # Store reasoning separately and drop raw tool-call markup; the chat
        # template re-renders both, matching what the model wrote (so the
        # prompt cache stays valid).
        step = [
            {
                "role": "assistant",
                "content": content.strip(),
                "reasoning_content": reasoning.replace("<think>", "").strip(),
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            }
        ]
        for call in calls:
            command = call.get("arguments", {}).get("command")
            if call.get("name") != "run" or not command:
                result = "error: only the `run` tool with a `command` argument is available"
                print(red(f"  ✗ {result}"))
            else:
                print(cyan(f"  $ {command}"))
                status("running…")
                code, out = exec_in_container(name, command)
                clear_status()
                print_output(out)
                if code != 0:
                    print(red(f"  │ exit {code}"))
                commands += 1
                if len(out) > max_output:
                    # Keep both ends: errors are often at the start, results at the end.
                    half = max_output // 2
                    cut = len(out) - 2 * half
                    out = f"{out[:half]}\n...({cut} chars truncated)...\n{out[-half:]}"
                result = f"exit code: {code}\n{out}"
            step.append({"role": "tool", "content": result})
        for error in errors:
            print(red(f"  ✗ {error}"))
            step.append({"role": "tool", "content": f"error: {error}"})
        if finish == "length":
            print(yellow(f"! reply cut off at max_tokens ({CONFIG['model']['max_tokens']})"))
            step.append(
                {"role": "user", "content": "Your reply was cut off at the token limit. "
                 "Continue, more concisely."}
            )
        # Only commit a step once it fully finishes, so Ctrl-C mid-step keeps history valid.
        messages.extend(step)
        if not calls and not errors and finish != "length":
            return commands
    print(yellow(f"! stopped after {max_steps} steps"))
    return commands


def handle_command(line: str, llm: LLM, name: str, messages: list) -> bool:
    """Run a slash command. Returns False when the chat should end."""
    cmd, _, arg = line.partition(" ")
    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/info":
        print(container_info(name))
    elif cmd == "/sh":
        if not arg.strip():
            print(yellow("usage: /sh <command>"))
        else:
            code, out = exec_in_container(name, arg)
            print_output(out)
            if code != 0:
                print(red(f"  │ exit {code}"))
    elif cmd == "/clear":
        del messages[1:]
        llm.cache.reset()
        print(dim("conversation cleared (container kept)"))
    else:
        print(yellow(f"unknown command {cmd}, try /help"))
    return True


def chat(name: str, image: str, first_message: str, max_steps: int) -> None:
    m = CONFIG["model"]
    print(dim(f"Loading {m['name']}…"))
    llm = LLM()

    print(f"\n{bold('qwen sandbox')} {dim(f'· {m["name"].split("/")[-1]} · {image} · {name}')}")
    print(dim("Type a message. /help for commands, Ctrl-C interrupts, /exit quits."))
    messages = [{"role": "system", "content": m["system_prompt"]}]
    message = first_message
    while True:
        if not message:
            try:
                message = input(f"\n{bold(green('you ›'))} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not message:
                continue
            if message.startswith("/"):
                if not handle_command(message, llm, name, messages):
                    return
                message = ""
                continue
        else:
            print(f"\n{bold(green('you ›'))} {message}")
        messages.append({"role": "user", "content": message})
        message = ""
        print()
        start = time.monotonic()
        stats = {"tokens": 0, "gen_time": 0.0, "context": 0}
        try:
            commands = agent_turn(llm, name, messages, max_steps, stats)
        except KeyboardInterrupt:
            clear_status()
            print(yellow("\n✗ interrupted"))
            continue
        summary = f"done in {time.monotonic() - start:.0f}s"
        if commands:
            summary += f" · {commands} command{'s' * (commands != 1)}"
        if stats["gen_time"]:
            summary += f" · {stats['tokens'] / stats['gen_time']:.1f} tok/s"
        if stats["context"]:
            summary += f" · {stats['context']} tok context"
        print(green(f"✓ {summary}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a Qwen agent in a throwaway container.")
    parser.add_argument("message", nargs="*", help="optional first message for the agent")
    parser.add_argument("--config", type=Path, help=f"config file (default: {CONFIG_PATH.name})")
    parser.add_argument("--image", help="container image")
    parser.add_argument("--cpus", help="CPUs for the container")
    parser.add_argument("--memory", help="memory for the container, e.g. 4G")
    parser.add_argument("--max-steps", type=int, help="max tool calls per reply")
    args = parser.parse_args()

    load_config(args.config or CONFIG_PATH, required=args.config is not None)
    container = CONFIG["container"]
    image = args.image or container["image"]
    cpus = args.cpus or container["cpus"]
    memory = args.memory or container["memory"]
    max_steps = args.max_steps or CONFIG["agent"]["max_steps"]

    ensure_container_cli()
    # Turn SIGTERM into a normal exit so the finally block still runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    name = f"qwen-sandbox-{uuid.uuid4().hex[:8]}"
    try:
        start_container(name, image, cpus, memory)
        chat(name, image, " ".join(args.message), max_steps)
    except KeyboardInterrupt:
        pass
    finally:
        delete_container(name)


if __name__ == "__main__":
    main()
