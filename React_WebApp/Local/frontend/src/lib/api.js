const MODEL_PORTS = {
  resnet50:     5001,
  efficientnet: 5002,
  convnext:     5003,
  swin:         5004,
  raddino:      5005,
  radjepa:      5006,
};

function baseUrl(modelId) {
  const port = MODEL_PORTS[modelId];
  if (!port) throw new Error(`Unknown model: ${modelId}`);
  return `http://localhost:${port}`;
}

export async function fetchAllHealth() {
  const results = await Promise.allSettled(
    Object.entries(MODEL_PORTS).map(async ([id, port]) => {
      const res = await fetch(`http://localhost:${port}/health`, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      return [id, { online: true, device: data.device ?? null }];
    })
  );
  return Object.fromEntries(
    results.map((r, i) => {
      const id = Object.keys(MODEL_PORTS)[i];
      return r.status === "fulfilled"
        ? r.value
        : [id, { online: false, device: null }];
    })
  );
}

export async function predict(modelId, file) {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${baseUrl(modelId)}/predict`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function explain(modelId, method, file) {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${baseUrl(modelId)}/explain/${method}`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
