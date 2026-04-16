/**
 * models.js — single source of truth for model configuration.
 *
 * To add a new model:
 *   1. Add an entry to MODELS below.
 *   2. Drop its checkpoint in backend/checkpoints/ — start_all.bat picks it up automatically.
 *      (Both api.js and Analysis.jsx read from here.)
 */

export const MODELS = [
  { id: "resnet50",     port: 5001, label: "ResNet50",        tag: "CNN", desc: "Classic torchvision baseline."             },
  { id: "efficientnet", port: 5002, label: "EfficientNet B0",  tag: "CNN", desc: "Lightweight & fast. Best for quick scans." },
  { id: "convnext",     port: 5003, label: "ConvNeXt V2",      tag: "CNN", desc: "Strong CNN baseline with modern design."   },
  { id: "swin",         port: 5004, label: "Swin Transformer", tag: "CNN", desc: "Hierarchical vision transformer."          },
  { id: "raddino",      port: 5005, label: "RadDINO",           tag: "ViT", desc: "Medical ViT pretrained on chest X-rays."  },
  { id: "radjepa",      port: 5006, label: "RadJEPA",           tag: "ViT", desc: "Self-supervised ViT with joint embedding." },
];

/** Lookup a model by id. Throws if not found. */
export function getModel(id) {
  const m = MODELS.find(m => m.id === id);
  if (!m) throw new Error(`Unknown model id: "${id}"`);
  return m;
}

/** { resnet50: 5001, efficientnet: 5002, ... } — for code that just needs the port map. */
export const PORTS = Object.fromEntries(MODELS.map(m => [m.id, m.port]));

/** Models that use a Vision Transformer backbone (affects which explainability methods are shown). */
export const VIT_MODEL_IDS = MODELS.filter(m => m.tag === "ViT").map(m => m.id);

/** Models that don't support SHAP. */
export const NO_SHAP_MODEL_IDS = ["convnext", "swin", "raddino", "radjepa"];
