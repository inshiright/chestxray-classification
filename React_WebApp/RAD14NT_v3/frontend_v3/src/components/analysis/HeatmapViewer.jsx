import React, { useState } from "react";
import { motion } from "framer-motion";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Eye, Layers, BarChart3, AlertCircle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";

const chartColors = [
  "hsl(0, 72%, 51%)",    // red — highest
  "hsl(27, 87%, 67%)",   // orange
  "hsl(43, 74%, 66%)",   // amber
  "hsl(160, 60%, 45%)",  // teal
  "hsl(199, 89%, 48%)",  // sky
  "hsl(172, 66%, 50%)",  // cyan
  "hsl(215, 25%, 45%)",  // steel
  "hsl(280, 65%, 60%)",  // purple
  "hsl(215, 25%, 27%)",  // dark blue
  "hsl(160, 60%, 35%)",  // dark teal
  "hsl(199, 89%, 35%)",  // dark sky
  "hsl(27, 60%, 50%)",   // dark orange
  "hsl(43, 60%, 50%)",   // dark amber
  "hsl(0, 50%, 40%)",    // dark red
];

// Custom tooltip for the chart
function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{
        background: "hsl(var(--card))",
        border: "1px solid hsl(var(--border))",
        borderRadius: "10px",
        padding: "8px 12px",
        fontSize: "12px",
        boxShadow: "0 8px 30px rgba(0,0,0,0.08)",
      }}>
        <p style={{ fontWeight: 600, color: "hsl(var(--foreground))", marginBottom: 2 }}>
          {d.fullName}
        </p>
        <p style={{ color: payload[0].fill }}>
          Confidence: <strong>{d.confidence}%</strong>
        </p>
      </div>
    );
  }
  return null;
}

// Heatmap tab content — shows GradCAM for CNN models, Attention Rollout for ViT models
function HeatmapTab({ results, originalImage, explainResults, heatmapOpacity, setHeatmapOpacity, isVit }) {
  // For CNN: prefer freshly-run gradcam, fall back to /predict's heatmapUrl
  // For ViT: prefer freshly-run attention_rollout overlay
  const cnnHeatmap =
    explainResults?.gradcam?.image ||
    (!isVit ? results?.heatmapUrl : null) ||
    null;

  const vitHeatmap =
    explainResults?.attention_rollout?.overlay ||
    (isVit ? results?.heatmapUrl : null) ||
    null;

  const activeHeatmap = isVit ? vitHeatmap : cnnHeatmap;
  const heatmapLabel = isVit ? "Attention Rollout" : "Grad-CAM";

  return (
    <div className="space-y-4">
      {/* Label pill */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-semibold px-2 py-1 rounded-lg bg-primary/10 text-primary font-space uppercase tracking-wider">
          {heatmapLabel}
        </span>
        {isVit && (
          <span className="text-[11px] text-muted-foreground font-inter">
            ViT models use attention rollout (not Grad-CAM)
          </span>
        )}
      </div>

      <div className="relative rounded-2xl overflow-hidden border border-border bg-card shadow-sm">
        {/* Original X-Ray */}
        <img
          src={originalImage}
          alt="X-Ray"
          className="w-full h-auto max-h-[400px] object-contain"
        />

        {/* Overlay heatmap */}
        {activeHeatmap && (
          <img
            src={activeHeatmap}
            alt={`${heatmapLabel} heatmap`}
            className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
            style={{ opacity: heatmapOpacity[0] }}
          />
        )}

        {/* Loading / not available state */}
        {!activeHeatmap && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
            <p className="text-xs text-muted-foreground/60 font-inter bg-card/90 px-3 py-1.5 rounded-full">
              {isVit
                ? "Run Attention Rollout in Explainability Tools to see the heatmap"
                : "Heatmap loading…"}
            </p>
          </div>
        )}
      </div>

      {/* Opacity Slider */}
      {activeHeatmap && (
        <>
          <div className="flex items-center gap-4 px-1">
            <Layers className="w-4 h-4 text-muted-foreground shrink-0" />
            <div className="flex-1">
              <Slider
                value={heatmapOpacity}
                onValueChange={setHeatmapOpacity}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
              />
            </div>
            <span className="text-xs text-muted-foreground font-inter w-10 text-right">
              {Math.round(heatmapOpacity[0] * 100)}%
            </span>
          </div>
          <p className="text-xs text-muted-foreground/60 font-inter text-center">
            Adjust slider to control overlay intensity
          </p>
        </>
      )}
    </div>
  );
}

export default function HeatmapViewer({ results, originalImage, explainResults = {}, isVit = false }) {
  const [heatmapOpacity, setHeatmapOpacity] = useState([0.6]);

  if (!results) return null;

  // All 14 diseases, sorted by confidence descending
  const chartData = [...results.predictions]
    .sort((a, b) => b.confidence - a.confidence)
    .map((p, i) => ({
      name: p.disease.length > 14 ? p.disease.substring(0, 13) + "…" : p.disease,
      fullName: p.disease,
      confidence: +(p.confidence * 100).toFixed(1),
      rank: i,
    }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <Tabs defaultValue="heatmap" className="w-full">
        <TabsList className="w-full bg-secondary/50 p-1 rounded-xl">
          <TabsTrigger value="heatmap" className="flex-1 gap-2 rounded-lg font-inter text-sm">
            <Eye className="w-4 h-4" />
            Heatmap
          </TabsTrigger>
          <TabsTrigger value="chart" className="flex-1 gap-2 rounded-lg font-inter text-sm">
            <BarChart3 className="w-4 h-4" />
            All Diseases
          </TabsTrigger>
        </TabsList>

        <TabsContent value="heatmap" className="mt-4">
          <HeatmapTab
            results={results}
            originalImage={originalImage}
            explainResults={explainResults}
            heatmapOpacity={heatmapOpacity}
            setHeatmapOpacity={setHeatmapOpacity}
            isVit={isVit}
          />
        </TabsContent>

        <TabsContent value="chart" className="mt-4">
          <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold font-space text-sm text-foreground">
                All 14 Disease Confidence Scores
              </h4>
              <span className="text-[10px] text-muted-foreground font-inter bg-muted/50 px-2 py-1 rounded-lg">
                sorted by confidence
              </span>
            </div>
            <div className="h-[420px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ left: 8, right: 32, top: 4, bottom: 4 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    horizontal={false}
                  />
                  <XAxis
                    type="number"
                    domain={[0, 100]}
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    tickFormatter={(v) => `${v}%`}
                    tickCount={6}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={108}
                    tick={{ fontSize: 11, fill: "hsl(var(--foreground))" }}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(var(--muted))", opacity: 0.3 }} />
                  {/* 5% noise floor reference line */}
                  <ReferenceLine
                    x={5}
                    stroke="hsl(var(--muted-foreground))"
                    strokeDasharray="4 4"
                    strokeOpacity={0.4}
                    label={{ value: "5%", position: "top", fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                  />
                  <Bar dataKey="confidence" radius={[0, 6, 6, 0]} barSize={20}>
                    {chartData.map((entry, index) => (
                      <Cell key={index} fill={chartColors[index % chartColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-muted-foreground/50 font-inter mt-3 text-center">
              Dashed line at 5% marks the noise floor threshold
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </motion.div>
  );
}
