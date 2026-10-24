// WebGPU forward pass of the trained network (same maths as engine/.../nn/QuoridorNet.kt).
// Reads the same .qnet bytes, keeps the weights on the GPU and evaluates a batch of encoded
// positions per call: one command buffer for all layers, one read-back.
// Exposes window.QuoridorGpu.create(bytes, maxBatch) -> Promise<net | null>, where
// net.evaluate(input Float32Array, count) -> Promise<{policy: Float32Array, values: Float32Array}>.
(function () {
  "use strict";

  const PLANES = 10;          // FeatureEncoder.PLANES
  const HEAD = 32;            // QuoridorNet.HEAD_CHANNELS
  const HIDDEN = 64;          // QuoridorNet.VALUE_HIDDEN
  const POLICY_PLANES = 3;    // PolicyIndex.PLANES
  const WORKGROUP = 64;

  // One invocation per (batch item, output channel, cell). Weights [out][in][k*k], bias [out].
  const CONV_WGSL = `
struct Params { inputs: u32, outputs: u32, n: u32, kernel: u32, relu: u32, residual: u32, batch: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let area = p.n * p.n;
  let total = p.batch * p.outputs * area;
  let idx = gid.x;
  if (idx >= total) { return; }
  let cell = idx % area;
  let o = (idx / area) % p.outputs;
  let b = idx / (area * p.outputs);
  let y = i32(cell / p.n);
  let x = i32(cell % p.n);
  let n = i32(p.n);
  let kk = p.kernel * p.kernel;
  let half = i32(p.kernel / 2u);
  var sum = bias[o];
  let srcBase = b * p.inputs * area;
  let wBase = o * p.inputs * kk;
  for (var i = 0u; i < p.inputs; i++) {
    let plane = srcBase + i * area;
    let wPlane = wBase + i * kk;
    for (var ky = 0u; ky < p.kernel; ky++) {
      let sy = y + i32(ky) - half;
      if (sy < 0 || sy >= n) { continue; }
      for (var kx = 0u; kx < p.kernel; kx++) {
        let sx = x + i32(kx) - half;
        if (sx < 0 || sx >= n) { continue; }
        sum += weight[wPlane + ky * p.kernel + kx] * src[plane + u32(sy) * p.n + u32(sx)];
      }
    }
  }
  if (p.residual == 1u) { sum += res[idx]; }
  if (p.relu == 1u && sum < 0.0) { sum = 0.0; }
  dst[idx] = sum;
}`;

  // Value head tail per batch item: mean and max pooling of the HEAD channels, two dense layers, tanh.
  const VALUE_WGSL = `
struct Params { area: u32, batch: u32, p2: u32, p3: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> head: array<f32>;
@group(0) @binding(2) var<storage, read> w1: array<f32>;
@group(0) @binding(3) var<storage, read> b1: array<f32>;
@group(0) @binding(4) var<storage, read> w2: array<f32>;
@group(0) @binding(5) var<storage, read> b2: array<f32>;
@group(0) @binding(6) var<storage, read_write> values: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= p.batch) { return; }
  var pooled: array<f32, ${2 * HEAD}>;
  for (var c = 0u; c < ${HEAD}u; c++) {
    var sum = 0.0;
    var top = -3.4e38;
    let base = (b * ${HEAD}u + c) * p.area;
    for (var k = 0u; k < p.area; k++) {
      let v = head[base + k];
      sum += v;
      top = max(top, v);
    }
    pooled[c] = sum / f32(p.area);
    pooled[${HEAD}u + c] = top;
  }
  var v = b2[0];
  for (var o = 0u; o < ${HIDDEN}u; o++) {
    var s = b1[o];
    for (var i = 0u; i < ${2 * HEAD}u; i++) { s += w1[o * ${2 * HEAD}u + i] * pooled[i]; }
    if (s < 0.0) { s = 0.0; }
    v += w2[o] * s;
  }
  values[b] = tanh(v);
}`;

  function readQnet(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== "QNET") throw new Error("Not a .qnet file");
    let pos = 4;
    const int = () => { const v = view.getInt32(pos, true); pos += 4; return v; };
    const version = int(), encoder = int(), channels = int(), blocks = int(), size = int();
    if (version !== 1 || encoder !== 1) throw new Error("Unsupported .qnet " + version + "/" + encoder);
    const floats = (count) => {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i++) { out[i] = view.getFloat32(pos, true); pos += 4; }
      return out;
    };
    const conv = (inputs, outputs, k) => ({ w: floats(outputs * inputs * k * k), b: floats(outputs), inputs, outputs, k });
    const stem = conv(PLANES, channels, 3);
    const tower = [];
    for (let i = 0; i < blocks; i++) tower.push([conv(channels, channels, 3), conv(channels, channels, 3)]);
    const policy1 = conv(channels, HEAD, 1);
    const policy2 = conv(HEAD, POLICY_PLANES, 1);
    const value1 = conv(channels, HEAD, 1);
    const fc1 = conv(2 * HEAD, HIDDEN, 1);
    const fc2 = conv(HIDDEN, 1, 1);
    if (pos !== bytes.byteLength) throw new Error("Unexpected trailing bytes in .qnet");
    return { channels, blocks, size, stem, tower, policy1, policy2, value1, fc1, fc2 };
  }

  async function create(bytes, maxBatch) {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const net = readQnet(bytes);
    const n = net.size, area = n * n, C = net.channels;
    const S = GPUBufferUsage.STORAGE, U = GPUBufferUsage.UNIFORM, D = GPUBufferUsage.COPY_DST, R = GPUBufferUsage.COPY_SRC;

    const upload = (data) => {
      const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage: S | D });
      device.queue.writeBuffer(buf, 0, data);
      return buf;
    };
    const storage = (floats, extra = 0) => device.createBuffer({ size: Math.max(16, floats * 4), usage: S | D | extra });

    const input = storage(maxBatch * PLANES * area);
    const act = [storage(maxBatch * C * area), storage(maxBatch * C * area), storage(maxBatch * C * area)];
    const head = storage(maxBatch * HEAD * area);
    const policy = storage(maxBatch * POLICY_PLANES * area, R);
    const values = storage(maxBatch, R);
    const readback = device.createBuffer({ size: maxBatch * (POLICY_PLANES * area + 1) * 4, usage: GPUBufferUsage.MAP_READ | D });
    const dummy = storage(4);

    const convModule = device.createShaderModule({ code: CONV_WGSL });
    const convPipeline = device.createComputePipeline({ layout: "auto", compute: { module: convModule, entryPoint: "main" } });
    const valueModule = device.createShaderModule({ code: VALUE_WGSL });
    const valuePipeline = device.createComputePipeline({ layout: "auto", compute: { module: valueModule, entryPoint: "main" } });

    // Each layer: uniform params (batch filled per call), weights, bind group.
    const layers = [];
    const addConv = (layer, src, dst, relu, residual) => {
      const params = device.createBuffer({ size: 32, usage: U | D });
      const group = device.createBindGroup({
        layout: convPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: upload(layer.w) } },
          { binding: 3, resource: { buffer: upload(layer.b) } },
          { binding: 4, resource: { buffer: residual || dummy } },
          { binding: 5, resource: { buffer: dst } },
        ],
      });
      layers.push({ params, group, layer, relu, residual: residual ? 1 : 0 });
    };
    addConv(net.stem, input, act[0], true, null);
    let current = 0, spare = 1;
    for (const [c1, c2] of net.tower) {
      addConv(c1, act[current], act[2], true, null);
      addConv(c2, act[2], act[spare], true, act[current]);
      const t = current; current = spare; spare = t;
    }
    addConv(net.policy1, act[current], head, true, null);
    addConv(net.policy2, head, policy, false, null);
    // value1 reuses the head buffer after policy2 has consumed it (same queue, in order).
    addConv(net.value1, act[current], head, true, null);
    const valueParams = device.createBuffer({ size: 16, usage: U | D });
    const valueGroup = device.createBindGroup({
      layout: valuePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: valueParams } },
        { binding: 1, resource: { buffer: head } },
        { binding: 2, resource: { buffer: upload(net.fc1.w) } },
        { binding: 3, resource: { buffer: upload(net.fc1.b) } },
        { binding: 4, resource: { buffer: upload(net.fc2.w) } },
        { binding: 5, resource: { buffer: upload(net.fc2.b) } },
        { binding: 6, resource: { buffer: values } },
      ],
    });

    let lastBatch = -1;
    const setBatch = (batch) => {
      if (batch === lastBatch) return;
      for (const l of layers) {
        device.queue.writeBuffer(l.params, 0, new Uint32Array([l.layer.inputs, l.layer.outputs, n, l.layer.k, l.relu ? 1 : 0, l.residual, batch, 0]));
      }
      device.queue.writeBuffer(valueParams, 0, new Uint32Array([area, batch, 0, 0]));
      lastBatch = batch;
    };

    let busy = Promise.resolve();
    async function evaluate(inputData, count) {
      if (count < 1 || count > maxBatch) throw new Error("bad batch " + count);
      const run = busy.then(async () => {
        setBatch(count);
        device.queue.writeBuffer(input, 0, inputData, 0, count * PLANES * area);
        const enc = device.createCommandEncoder();
        // policy2 before value1: both use the head buffer, so each gets its own pass, in order.
        for (const l of layers) {
          const pass = enc.beginComputePass();
          pass.setPipeline(convPipeline);
          pass.setBindGroup(0, l.group);
          pass.dispatchWorkgroups(Math.ceil(count * l.layer.outputs * area / WORKGROUP));
          pass.end();
        }
        const vpass = enc.beginComputePass();
        vpass.setPipeline(valuePipeline);
        vpass.setBindGroup(0, valueGroup);
        vpass.dispatchWorkgroups(count);
        vpass.end();
        const policyBytes = count * POLICY_PLANES * area * 4;
        enc.copyBufferToBuffer(policy, 0, readback, 0, policyBytes);
        enc.copyBufferToBuffer(values, 0, readback, policyBytes, count * 4);
        device.queue.submit([enc.finish()]);
        await readback.mapAsync(GPUMapMode.READ, 0, policyBytes + count * 4);
        const all = new Float32Array(readback.getMappedRange(0, policyBytes + count * 4).slice(0));
        readback.unmap();
        return { policy: all.subarray(0, count * POLICY_PLANES * area), values: all.subarray(count * POLICY_PLANES * area) };
      });
      busy = run.catch(() => {});
      return run;
    }

    return { evaluate, maxBatch, adapter: (adapter.info && adapter.info.vendor) || "" };
  }

  window.QuoridorGpu = { create };
})();
