/**
 * Chest X-Ray API client.
 *
 * Each model runs its own Flask server on a dedicated port.
 * Keep PORTS in sync with the declare -A PORTS block in backend/start_all.sh.
 *
 * Set VITE_API_BASE in .env to override the hostname (default: http://localhost).
 */

const BASE_HOST = import.meta.env.VITE_API_BASE || "http://localhost";

/** Port each model's API server listens on. */
const PORTS = {
  resnet50:    5001,
  efficientnet:5002,
  convnext:    5003,
  swin:        5004,
  raddino:     5005,
  radjepa:     5006,
};

/** Return the base URL for a given model id. */
function baseUrl(modelId) {
  const port = PORTS[modelId];
  if (!port) throw new Error(`Unknown model id: ${modelId}`);
  return `${BASE_HOST}:${port}`;
}

async function postImage(url, file, extra = {}) {
  const form = new FormData();
  form.append("image", file);
  for (const [k, v] of Object.entries(extra)) form.append(k, v);
  const res = await fetch(url, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/**
 * GET /health on the server for the given model.
 * Returns { status, model, device } or throws if unreachable.
 */
export async function fetchHealth(modelId) {
  const res = await fetch(`${baseUrl(modelId)}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Poll /health for every model and return a map of
 * { modelId -> { online: bool, device: string } }
 */
export async function fetchAllHealth() {
  const results = {};
  await Promise.all(
    Object.keys(PORTS).map(async (id) => {
      try {
        const data = await fetchHealth(id);
        results[id] = { online: true, device: data.device ?? "?" };
      } catch {
        results[id] = { online: false, device: null };
      }
    })
  );
  return results;
}

/** Run inference with the model currently selected in the UI. */
export async function predict(modelId, file) {
  return postImage(`${baseUrl(modelId)}/predict`, file);
}

/** Run a single explainability method. */
export async function explain(modelId, method, file, opts = {}) {
  return postImage(`${baseUrl(modelId)}/explain/${method}`, file, opts);
}

/** Run all explainability methods in one shot (slow). */
export async function analyzeAll(modelId, file) {
  return postImage(`${baseUrl(modelId)}/explain/all`, file);
}
