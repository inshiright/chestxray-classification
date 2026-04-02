import React, { useState } from "react";
import { motion } from "framer-motion";
import { FlaskConical, ChevronDown, ChevronUp } from "lucide-react";

const PORTS = {
  resnet50:    5001,
  efficientnet:5002,
  convnext:    5003,
  swin:        5004,
  raddino:     5005,
  radjepa:     5006,
};

const SAMPLES = [
  { label: "Atelectasis",        file: "00000030_001.png" },
  { label: "Cardiomegaly",       file: "00000045_000.png" },
  { label: "Pneumonia",          file: "00005032_000.png" },
  { label: "Mass",               file: "00005246_000.png" },
  { label: "Nodule",             file: "00005417_000.png" },
  { label: "Infiltration",       file: "00030311_000.png" },
  { label: "Effusion",           file: "00030436_000.png" },
  { label: "Pneumothorax",       file: "00005855_000.png" },
  { label: "Consolidation",      file: "00010849_000.png" },
  { label: "Edema",              file: "00015293_000.png" },
  { label: "Emphysema",          file: "00017738_000.png" },
  { label: "Fibrosis",           file: "00016998_000.png" },
  { label: "Pleural Thickening", file: "00016576_000.png" },
  { label: "Hernia",             file: "00013716_002.png" },
];

export default function SampleGallery({ onSelect, selectedModel }) {
  const [open, setOpen] = useState(false);

  // Use the first online port we can find, falling back through the port list
  const port = PORTS[selectedModel] ?? Object.values(PORTS)[0];
  const baseUrl = `http://localhost:${port}`;

  const handleSelect = async (sample) => {
    const url = `${baseUrl}/samples/${sample.file}`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: blob.type || "image/png" });
      const preview = URL.createObjectURL(blob);
      onSelect({ file, preview });
    } catch (err) {
      console.error("Failed to load sample:", err);
    }
  };

  return (
    <div className="rounded-2xl border border-border bg-card overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <FlaskConical className="w-4 h-4 text-primary" />
          </div>
          <div className="text-left">
            <p className="font-semibold font-space text-sm text-foreground">Try a Sample X-Ray</p>
            <p className="text-xs text-muted-foreground font-inter">
              From the NIH Chest X-ray Dataset · Click to expand
            </p>
          </div>
        </div>
        {open
          ? <ChevronUp className="w-4 h-4 text-muted-foreground" />
          : <ChevronDown className="w-4 h-4 text-muted-foreground" />
        }
      </button>

      {open && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="border-t border-border px-5 py-4"
        >
          <div className="grid grid-cols-4 sm:grid-cols-7 gap-3">
            {SAMPLES.map((sample) => (
              <button
                key={sample.file}
                onClick={() => handleSelect(sample)}
                className="group flex flex-col items-center gap-2 rounded-xl p-2 hover:bg-primary/5 border border-transparent hover:border-primary/20 transition-all duration-200"
              >
                <div className="w-full aspect-square rounded-lg overflow-hidden bg-foreground/5 border border-border">
                  <img
                    src={`${baseUrl}/samples/${sample.file}`}
                    alt={sample.label}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 filter grayscale"
                  />
                </div>
                <span className="text-[11px] font-inter text-muted-foreground group-hover:text-foreground transition-colors text-center leading-tight">
                  {sample.label}
                </span>
              </button>
            ))}
          </div>

          <div className="mt-4 pt-3 border-t border-border">
            <p className="text-[11px] text-muted-foreground/50 font-inter">
              Source: NIH Clinical Center Chest X-ray Dataset (Wang et al., CVPR 2017)
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
