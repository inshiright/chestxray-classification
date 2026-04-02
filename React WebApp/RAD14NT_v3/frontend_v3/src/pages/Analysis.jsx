import React, { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import {
  Scan, RotateCcw, Stethoscope, ChevronDown, ChevronUp,
  Loader2, AlertTriangle, CheckCircle2, Cpu, WifiOff,
} from "lucide-react";
import ImageUploader from "../components/analysis/ImageUploader";
import ResultsPanel from "../components/analysis/ResultsPanel";
import HeatmapViewer from "../components/analysis/HeatmapViewer";
import SampleGallery from "../components/analysis/SampleGallery";
import { predict, explain, fetchAllHealth } from "@/lib/api";

// ── Model registry ────────────────────────────────────────────────────────────
const MODELS = [
  { id: "resnet50",     label: "ResNet50",        tag: "CNN", desc: "Classic torchvision baseline."            },
  { id: "efficientnet", label: "EfficientNet B0",  tag: "CNN", desc: "Lightweight & fast. Best for quick scans."},
  { id: "convnext",     label: "ConvNeXt V2",      tag: "CNN", desc: "Strong CNN baseline with modern design."  },
  { id: "swin",         label: "Swin Transformer", tag: "CNN", desc: "Hierarchical vision transformer."         },
  { id: "raddino",      label: "RadDINO",           tag: "ViT", desc: "Medical ViT pretrained on chest X-rays." },
  { id: "radjepa",      label: "RadJEPA",           tag: "ViT", desc: "Self-supervised ViT with joint embedding."},
];

const DISEASES = [
  "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
  "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
  "Emphysema","Fibrosis","Pleural Thickening","Hernia",
];

const EXPLAIN_METHODS = [
  { id: "gradcam",           label: "Grad-CAM",         cnnOnly: true },
  { id: "attention_rollout", label: "Attention Rollout", vitOnly: true },
  { id: "lime",              label: "LIME",              slow: true    },
  { id: "shap",              label: "SHAP",              slow: true, noShap: true },
  { id: "blackout",          label: "Blackout / Ins-Del"               },
];

const STATUS_LABELS = {
  predicting:        "Running inference…",
  gradcam:           "Computing Grad-CAM…",
  attention_rollout: "Computing Attention Rollout…",
  lime:              "Running LIME (this may take ~30 s)…",
  shap:              "Running SHAP (this may take ~30 s)…",
  blackout:          "Running Blackout test…",
};

// ── Model Selector ────────────────────────────────────────────────────────────
function ModelSelector({ selectedModel, onSelect, health }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const active = MODELS.find(m => m.id === selectedModel) ?? MODELS[0];
  const activeHealth = health[selectedModel];
  const isOnline = activeHealth?.online;

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const onlineCount = Object.values(health).filter(h => h?.online).length;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className={[
          "flex items-center gap-2 h-9 pl-3 pr-2.5 rounded-xl border text-sm font-inter",
          "transition-all duration-200 select-none",
          isOnline === false
            ? "border-destructive/40 bg-destructive/5 text-destructive"
            : "border-border bg-secondary/60 hover:bg-secondary text-foreground",
        ].join(" ")}
      >
        <Cpu className="w-3.5 h-3.5 shrink-0 text-primary" />
        <span className="font-medium">{active.label}</span>
        <span className={[
          "text-[10px] font-semibold px-1.5 py-0.5 rounded-md font-space",
          active.tag === "ViT" ? "bg-accent/15 text-accent" : "bg-primary/15 text-primary",
        ].join(" ")}>{active.tag}</span>

        {/* live/offline dot */}
        <span className={[
          "w-1.5 h-1.5 rounded-full shrink-0",
          isOnline ? "bg-emerald-500" : "bg-muted-foreground/40",
        ].join(" ")} />

        <ChevronDown className={`w-3.5 h-3.5 shrink-0 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.97 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-1.5 w-72 rounded-2xl border border-border bg-card shadow-xl shadow-foreground/5 z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="px-3 py-2.5 border-b border-border/60">
              <div className="flex items-center justify-between">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground font-space">
                  Select Model
                </p>
                <p className="text-[10px] text-muted-foreground/60 font-inter">
                  {onlineCount} / {MODELS.length} online
                </p>
              </div>
              {onlineCount === 0 && (
                <p className="text-[11px] text-muted-foreground/70 font-inter mt-1 flex items-center gap-1.5">
                  <WifiOff className="w-3 h-3 shrink-0" />
                  No servers running — see start_all.sh
                </p>
              )}
            </div>

            {/* Model list */}
            <div className="p-1.5 space-y-0.5">
              {MODELS.map(model => {
                const isActive = model.id === selectedModel;
                const h = health[model.id];
                const online = h?.online;

                return (
                  <button
                    key={model.id}
                    onClick={() => { onSelect(model.id); setOpen(false); }}
                    className={[
                      "w-full flex items-start gap-3 px-3 py-2.5 rounded-xl text-left",
                      "transition-colors duration-150",
                      isActive ? "bg-primary/8" : "hover:bg-muted/60",
                      !online ? "opacity-50" : "",
                    ].join(" ")}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-sm font-medium font-space text-foreground">{model.label}</span>
                        <span className={[
                          "text-[10px] font-semibold px-1.5 py-0.5 rounded-md",
                          model.tag === "ViT" ? "bg-accent/15 text-accent" : "bg-primary/15 text-primary",
                        ].join(" ")}>{model.tag}</span>
                        {online && (
                          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md bg-emerald-500/12 text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                            <span className="w-1 h-1 rounded-full bg-emerald-500 inline-block" />
                            {h.device}
                          </span>
                        )}
                        {online === false && (
                          <span className="text-[10px] text-muted-foreground/50 font-inter">offline</span>
                        )}
                      </div>
                      <p className="text-[11px] text-muted-foreground font-inter mt-0.5 truncate">
                        {model.desc}
                      </p>
                    </div>
                    {isActive && (
                      <div className="w-4 h-4 rounded-full bg-primary/15 flex items-center justify-center mt-0.5 shrink-0">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Footer */}
            <div className="px-3 py-2.5 border-t border-border/60 bg-muted/20">
              <p className="text-[10px] text-muted-foreground/70 font-inter leading-relaxed">
                Run{" "}
                <code className="font-mono bg-muted/80 px-1 py-0.5 rounded text-foreground/80 text-[10px]">
                  ./start_all.sh
                </code>{" "}
                in{" "}
                <code className="font-mono bg-muted/80 px-1 py-0.5 rounded text-foreground/80 text-[10px]">
                  backend/
                </code>{" "}
                to start all model servers at once.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Analysis() {
  const [uploadedImage, setUploadedImage]   = useState(null);
  const [isAnalyzing, setIsAnalyzing]       = useState(false);
  const [status, setStatus]                 = useState(null);
  const [results, setResults]               = useState(null);
  const [explainResults, setExplainResults] = useState({});
  const [explainLoading, setExplainLoading] = useState({});
  const [explainErrors, setExplainErrors]   = useState({});
  const [error, setError]                   = useState(null);
  const [showExplain, setShowExplain]       = useState(false);
  const [selectedModel, setSelectedModel]   = useState("efficientnet");
  // health: { [modelId]: { online: bool, device: string|null } }
  const [health, setHealth]                 = useState({});

  // Poll all servers every 10 s
  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchAllHealth().then(h => { if (!cancelled) setHealth(h); });
    };
    poll();
    const id = setInterval(poll, 10_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const isVitModel = ["raddino", "radjepa"].includes(selectedModel);

  const handleImageSelect = useCallback((imageData) => {
    setUploadedImage(imageData);
    setResults(null);
    setExplainResults({});
    setExplainErrors({});
    setError(null);
    setShowExplain(false);
  }, []);

  const handleClear = useCallback(() => {
    setUploadedImage(null);
    setResults(null);
    setExplainResults({});
    setExplainErrors({});
    setError(null);
    setIsAnalyzing(false);
    setStatus(null);
    setShowExplain(false);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!uploadedImage?.file) return;
    setIsAnalyzing(true);
    setResults(null);
    setError(null);
    setStatus("predicting");
    try {
      const data = await predict(selectedModel, uploadedImage.file);
      setResults(data);
    } catch (err) {
      setError(`Prediction failed: ${err.message}. Is the ${selectedModel} server running?`);
    } finally {
      setIsAnalyzing(false);
      setStatus(null);
    }
  }, [uploadedImage, selectedModel]);

  const handleExplain = useCallback(async (methodId) => {
    if (!uploadedImage?.file) return;
    setExplainLoading(prev => ({ ...prev, [methodId]: true }));
    setExplainErrors(prev => ({ ...prev, [methodId]: null }));
    setStatus(methodId);
    try {
      const data = await explain(selectedModel, methodId, uploadedImage.file);
      setExplainResults(prev => ({ ...prev, [methodId]: data }));
    } catch (err) {
      setExplainErrors(prev => ({ ...prev, [methodId]: err.message }));
    } finally {
      setExplainLoading(prev => ({ ...prev, [methodId]: false }));
      setStatus(null);
    }
  }, [uploadedImage, selectedModel]);

  const noShapModels = ["convnext", "swin", "raddino", "radjepa"];
  const shapUnsupported = noShapModels.includes(selectedModel);

  const availableMethods = EXPLAIN_METHODS.filter(m => {
    if (m.vitOnly && !isVitModel) return false;
    if (m.cnnOnly && isVitModel) return false;
    if (m.noShap && shapUnsupported) return false;
    return true;
  });

  const explainSubtitle = isVitModel
    ? "Attention Rollout · LIME · Blackout"
    : selectedModel === "swin"
      ? "Grad-CAM · LIME · Blackout"
      : "Grad-CAM · LIME · SHAP · Blackout";

  const selectedOnline = health[selectedModel]?.online;

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-bold font-space tracking-tight text-foreground">RAD14NT</h1>
              <p className="text-[11px] text-muted-foreground font-inter -mt-0.5">Shining Light on 14-Disease Chest X-Ray Analysis</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ModelSelector
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
              health={health}
            />
            {uploadedImage && (
              <Button variant="ghost" size="sm" onClick={handleClear} className="text-muted-foreground font-inter">
                <RotateCcw className="w-4 h-4 mr-2" /> New Scan
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!uploadedImage && !results && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-10 mt-4">
            <h2 className="text-3xl sm:text-4xl font-bold font-space text-foreground tracking-tight">
              Analyse Chest X-Rays
            </h2>
            <p className="text-muted-foreground font-inter mt-3 max-w-lg mx-auto text-base">
              Upload a chest X-ray image to detect up to 14 different lung conditions using deep learning analysis.
            </p>
          </motion.div>
        )}

        {/* Offline warning */}
        <AnimatePresence>
          {selectedOnline === false && uploadedImage && (
            <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="flex items-center gap-3 mb-6 px-4 py-3 rounded-xl bg-amber-500/8 border border-amber-400/30 text-amber-700 dark:text-amber-400 font-inter text-sm">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              The <span className="font-semibold mx-1">{MODELS.find(m => m.id === selectedModel)?.label}</span> server is offline. Run <code className="mx-1 font-mono bg-amber-400/10 px-1.5 py-0.5 rounded text-xs">./start_all.sh</code> in the backend folder.
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {status && (
            <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="flex items-center gap-3 mb-6 px-4 py-3 rounded-xl bg-primary/5 border border-primary/20 text-primary font-inter text-sm">
              <Loader2 className="w-4 h-4 animate-spin shrink-0" />
              {STATUS_LABELS[status]}
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <div className="mb-6 px-4 py-3 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive font-inter text-sm">
            {error}
          </div>
        )}

        <div className={`grid gap-8 ${results ? "lg:grid-cols-2" : "max-w-xl mx-auto"}`}>
          <div className="space-y-6">
            <ImageUploader
              onImageSelect={handleImageSelect}
              uploadedImage={uploadedImage}
              onClear={handleClear}
              isAnalyzing={isAnalyzing}
            />

            {!uploadedImage && <SampleGallery onSelect={handleImageSelect} selectedModel={selectedModel} />}

            {uploadedImage && !results && !isAnalyzing && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <Button
                  onClick={handleAnalyze}
                  size="lg"
                  disabled={selectedOnline === false}
                  className="w-full h-14 rounded-2xl font-space text-base font-semibold bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Scan className="w-5 h-5 mr-3" /> Run Analysis
                </Button>
              </motion.div>
            )}

            {results && (
              <HeatmapViewer
                results={results}
                originalImage={uploadedImage?.preview}
                explainResults={explainResults}
                isVit={isVitModel}
              />
            )}
          </div>

          {results && (
            <div className="space-y-6">
              <ResultsPanel results={results} />

              <div className="rounded-2xl border border-border bg-card overflow-hidden">
                <button onClick={() => setShowExplain(o => !o)}
                  className="w-full flex items-center justify-between px-5 py-4 hover:bg-muted/30 transition-colors">
                  <div className="text-left">
                    <p className="font-semibold font-space text-sm text-foreground">Explainability Tools</p>
                    <p className="text-xs text-muted-foreground font-inter">{explainSubtitle}</p>
                  </div>
                  {showExplain
                    ? <ChevronUp className="w-4 h-4 text-muted-foreground" />
                    : <ChevronDown className="w-4 h-4 text-muted-foreground" />
                  }
                </button>

                {showExplain && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="border-t border-border px-5 py-5 space-y-6">
                    {availableMethods.map(method => (
                      <ExplainCard
                        key={method.id}
                        method={method}
                        loading={!!explainLoading[method.id]}
                        result={explainResults[method.id]}
                        error={explainErrors[method.id]}
                        onRun={() => handleExplain(method.id)}
                      />
                    ))}
                  </motion.div>
                )}
              </div>
            </div>
          )}
        </div>

        {!uploadedImage && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-12 text-center">
            <p className="text-xs text-muted-foreground/50 font-inter mb-3 uppercase tracking-widest">Detectable Conditions</p>
            <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
              {DISEASES.map((disease, i) => (
                <motion.span key={disease}
                  initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + i * 0.04 }}
                  className="px-3 py-1.5 rounded-full bg-secondary/70 text-secondary-foreground text-xs font-inter">
                  {disease}
                </motion.span>
              ))}
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}

// ── Explain Card ──────────────────────────────────────────────────────────────
function ExplainCard({ method, loading, result, error, onRun }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-semibold font-space text-sm text-foreground">{method.label}</p>
          <p className="text-[11px] text-muted-foreground/60 font-inter">
            {method.slow && "May take ~30 s"}
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={onRun} disabled={loading}
          className="font-inter text-xs h-8 px-3 rounded-xl">
          {loading && <Loader2 className="w-3 h-3 animate-spin mr-1" />}
          {loading ? "Running…" : result ? "Re-run" : "Run"}
        </Button>
      </div>

      {error && (
        <div className="text-xs text-destructive font-inter bg-destructive/5 border border-destructive/20 rounded-xl px-3 py-2">
          {error}
        </div>
      )}

      {result?.image && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          className="rounded-xl overflow-hidden border border-border">
          <img src={result.image} alt={method.label} className="w-full h-auto" />
          {method.id === "blackout" && result.deletionAUC != null && (
            <div className="px-4 py-3 border-t border-border bg-card flex flex-wrap gap-4 text-xs font-inter">
              <span><span className="text-muted-foreground">Target: </span>
                <span className="font-semibold text-foreground">{result.targetDisease}</span></span>
              <span><span className="text-muted-foreground">Deletion AUC: </span>
                <span className="font-semibold text-destructive">{result.deletionAUC.toFixed(3)}</span>
                <span className="text-muted-foreground/60"> ↓ better</span></span>
              <span><span className="text-muted-foreground">Insertion AUC: </span>
                <span className="font-semibold text-accent">{result.insertionAUC.toFixed(3)}</span>
                <span className="text-muted-foreground/60"> ↑ better</span></span>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
