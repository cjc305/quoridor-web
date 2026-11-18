"""Cloud self-play for 步步為營 (Quoridor): generates training data on a free GPU.

The notebook (quoridor-selfplay.ipynb) downloads this file from GitHub Pages on every start
and runs it, so fixes and new networks reach every notebook without editing it. It:
  1. finds where it runs (Google Colab, Kaggle, or a local machine),
  2. makes sure Java 17+ is there, and uses the GPU when there is one (else the CPU),
  3. downloads the bundle listed in manifest.json (trainer, inference server, network),
  4. starts infer_server.py on the GPU and runs trainer self-play in batches,
  5. copies each finished batch (.npz) to the output folder:
       Colab : Google Drive, MyDrive/quoridor-train/selfplay/<network>/<session>/
       Kaggle: /kaggle/working/selfplay/<network>/<session>/ (the notebook's output)
  6. between batches, re-reads manifest.json and switches to a newer bundle if one was deployed.
It stops at the time limit (QUORIDOR_HOURS, set by the notebook) and never touches the
mistake book. Environment overrides: QUORIDOR_TRAIN_BASE (bundle URL), QUORIDOR_OUT (folder).
"""
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("QUORIDOR_TRAIN_BASE", "https://cjc305.github.io/quoridor-web/train/").rstrip("/") + "/"
PORT = 7878


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def where():
    # Kaggle first: its image also ships the google.colab package (2026-11-18, the first Kaggle
    # run took itself for Colab and failed mounting Drive).
    if os.path.exists("/kaggle/working") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.exists("/var/colab/hostname") or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def fetch(url):
    sep = "&" if "?" in url else "?"
    with urllib.request.urlopen(f"{url}{sep}t={int(time.time())}", timeout=60) as r:
        return r.read()


def manifest():
    return json.loads(fetch(BASE + "manifest.json").decode("utf-8"))


def java_cmd():
    """A Java 17+ executable, installed with apt on Linux notebooks when missing."""
    def major(java):
        try:
            out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
            v = out.split('"')[1]
            return int(v.split(".")[0] if not v.startswith("1.") else v.split(".")[1])
        except Exception:
            return 0
    candidates = ["java", "/usr/lib/jvm/java-17-openjdk-amd64/bin/java"]
    for java in candidates:
        if major(java) >= 17:
            return java
    if shutil.which("apt-get"):
        say("installing Java 17 (about a minute)...")
        subprocess.run("apt-get -qq update && apt-get -qq install -y openjdk-17-jre-headless > /dev/null",
                       shell=True, check=True)
        for java in candidates:
            if major(java) >= 17:
                return java
    raise SystemExit("Java 17 or newer is needed and could not be installed")


def check_gpu():
    """Reports the GPU; without one the network runs on the CPU, slower but still useful
    (Kaggle's scheduled notebooks cannot use a GPU)."""
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is missing; use a Colab or Kaggle notebook")
    if torch.cuda.is_available():
        say(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        say("no GPU: the network runs on the CPU (slower). For speed, Colab: Runtime > Change "
            "runtime type > T4 GPU; Kaggle: Settings > Accelerator > GPU.")


def download(m, cache):
    folder = os.path.join(cache, m["version"])
    for name in m["files"]:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = fetch(BASE + name)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    return folder


def port_open():
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
        return True
    except OSError:
        return False


def start_server(folder, m, logs):
    log = open(os.path.join(logs, "infer_server.log"), "a")
    p = subprocess.Popen([sys.executable, "-u", "infer_server.py", m["net"], "--port", str(PORT)],
                         cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if port_open():
            return p
        if p.poll() is not None:
            raise SystemExit("infer_server.py stopped; see " + log.name)
        time.sleep(1)
    p.kill()
    raise SystemExit("infer_server.py did not start")


def stop(p):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
    for _ in range(20):
        if not port_open():
            return
        time.sleep(0.5)


def output_root(place):
    if os.environ.get("QUORIDOR_OUT"):
        return os.environ["QUORIDOR_OUT"]
    if place == "colab":
        from google.colab import drive
        drive.mount("/content/drive")
        return "/content/drive/MyDrive/quoridor-train/selfplay"
    if place == "kaggle":
        return "/kaggle/working/selfplay"
    return os.path.abspath("qtrain-out")


def main():
    place = where()
    hours = float(os.environ.get("QUORIDOR_HOURS", "11" if place != "local" else "0.25"))
    deadline = time.time() + hours * 3600
    say(f"running on {place} for up to {hours:g} h; bundle from {BASE}")
    check_gpu()
    java = java_cmd()
    root = output_root(place)
    cache = "/tmp/quoridor-train" if place != "local" else os.path.abspath("qtrain-cache")
    session = f"{place}-{time.strftime('%Y%m%d-%H%M')}-{random.randrange(16**4):04x}"
    threads = max(1, os.cpu_count() or 1)

    m = manifest()
    folder = download(m, cache)
    out = os.path.join(root, m["network"], session)
    os.makedirs(out, exist_ok=True)
    say(f"network {m['network']} (bundle {m['version']}), {threads} CPU threads; output {out}")
    server = start_server(folder, m, out)
    batch, games, last = 0, 0, None
    try:
        while True:
            left = deadline - time.time()
            if last is not None and left < last * 1.2 or left <= 60:
                say("time limit reached")
                break
            chunk = os.path.join(cache, "chunk")
            shutil.rmtree(chunk, ignore_errors=True)
            t0 = time.time()
            cp = os.pathsep.join(os.path.join(folder, f) for f in m["files"] if f.endswith(".jar"))
            cmd = [java, "-Xmx4g", "-cp", cp, m["main"], "selfplay", f"server=127.0.0.1:{PORT}", f"out={chunk}",
                   f"games={m['chunk_games']}", f"threads={threads}", f"sims={m['sims']}",
                   f"parallel={m['parallel']}", "drain=0"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            with open(os.path.join(out, "trainer.log"), "a") as f:
                f.write(r.stdout + r.stderr)
            if r.returncode != 0:
                raise SystemExit("self-play failed; see trainer.log:\n" + (r.stdout + r.stderr)[-2000:])
            batch += 1
            files = [f for f in os.listdir(chunk) if f.endswith(".npz")] if os.path.isdir(chunk) else []
            for f in files:
                shutil.copy(os.path.join(chunk, f), os.path.join(out, f"b{batch:03d}-{f}"))
            games += m["chunk_games"]
            last = time.time() - t0
            line = f"batch {batch}: {m['chunk_games']} games in {last / 60:.1f} min, {games} so far ({m['network']})"
            say(line)
            with open(os.path.join(out, "progress.log"), "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
            # A newer bundle (network, trainer or this script) was deployed: switch to it.
            try:
                newer = manifest()
            except Exception as e:
                say(f"could not re-read manifest ({e}); keeping {m['version']}")
                continue
            if newer["version"] != m["version"]:
                say(f"new bundle {newer['version']} ({newer['network']}); switching")
                stop(server)
                m = newer
                folder = download(m, cache)
                out = os.path.join(root, m["network"], session)
                os.makedirs(out, exist_ok=True)
                server = start_server(folder, m, out)
    except KeyboardInterrupt:
        say("stopped by hand")
    finally:
        stop(server)
    say(f"done: {games} games in {batch} batches, saved under {root}")


main()
