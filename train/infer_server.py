"""Local GPU inference server for the Kotlin self-play tool (trainer RemoteEvaluator).

Protocol over TCP on 127.0.0.1, little-endian, one request in flight per connection:
  request : int32 count, then count * PLANES * n * n float32 features
  response: count * 3 * n * n float32 policy logits, then count float32 values
A single batcher thread merges requests from all connections into one GPU batch.

usage: python infer_server.py MODEL.(onnx|pt) [--port 7878] [--size 9]
  (a .pt checkpoint from train.py, or --channels/--blocks for a random net)
"""
import argparse
import queue
import socket
import struct
import threading
import time

import numpy as np
import torch

from model import INPUT_PLANES, POLICY_PLANES, QuoridorNet

MAX_BATCH = 2048
# The GPU when there is one; the CPU otherwise (Kaggle's scheduled notebooks have no GPU).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_net(path, channels, blocks):
    if path and path.endswith(".pt"):
        ck = torch.load(path, map_location="cpu")
        net = QuoridorNet(ck["channels"], ck["blocks"])
        net.load_state_dict(ck["model"])
    else:
        torch.manual_seed(0)
        net = QuoridorNet(channels, blocks)
    return net.to(DEVICE).eval()


def recv_exact(conn, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = conn.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("client closed")
        got += k
    return buf


class Batcher(threading.Thread):
    def __init__(self, net, size):
        super().__init__(daemon=True)
        self.net, self.size = net, size
        self.requests = queue.Queue()
        self.positions = 0
        self.batches = 0

    def run(self):
        while True:
            items = [self.requests.get()]
            total = items[0][0].shape[0]
            while total < MAX_BATCH:
                try:
                    item = self.requests.get_nowait()
                except queue.Empty:
                    break
                items.append(item)
                total += item[0].shape[0]
            x = torch.from_numpy(np.concatenate([it[0] for it in items])).to(DEVICE, non_blocking=True)
            if DEVICE == "cuda":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    p, v = self.net(x)
            else:
                with torch.no_grad():
                    p, v = self.net(x)
            p = p.float().cpu().numpy()
            v = v.float().cpu().numpy()
            start = 0
            for features, slot in items:
                n = features.shape[0]
                slot["result"] = (p[start:start + n], v[start:start + n])
                slot["done"].set()
                start += n
            self.positions += total
            self.batches += 1


def serve_client(conn, batcher, size):
    input_size = INPUT_PLANES * size * size
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                (count,) = struct.unpack("<i", recv_exact(conn, 4))
                raw = recv_exact(conn, count * input_size * 4)
                features = np.frombuffer(raw, dtype="<f4").reshape(count, INPUT_PLANES, size, size)
                slot = {"done": threading.Event()}
                batcher.requests.put((features, slot))
                slot["done"].wait()
                p, v = slot["result"]
                conn.sendall(np.ascontiguousarray(p, dtype="<f4").tobytes() + np.ascontiguousarray(v, dtype="<f4").tobytes())
    except ConnectionError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--port", type=int, default=7878)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    a = ap.parse_args()

    net = load_net(a.model, a.channels, a.blocks)
    batcher = Batcher(net, a.size)
    batcher.start()
    srv = socket.create_server(("127.0.0.1", a.port))
    print(f"serving {net.channels}x{net.blocks} on {DEVICE}, 127.0.0.1:{a.port}", flush=True)

    def stats():
        last = 0
        while True:
            time.sleep(10)
            done = batcher.positions
            avg = done / max(batcher.batches, 1)
            print(f"{(done - last) / 10:,.0f} pos/s, avg batch {avg:.0f}", flush=True)
            last = done
    threading.Thread(target=stats, daemon=True).start()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve_client, args=(conn, batcher, a.size), daemon=True).start()


if __name__ == "__main__":
    main()
