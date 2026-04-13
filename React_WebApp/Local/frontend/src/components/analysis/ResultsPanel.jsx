import React from "react";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle2, Activity } from "lucide-react";

// Colour tier based on relative rank within this scan's results, not fixed thresholds
const barColor = (rank) => {
  if (rank === 0) return "bg-destructive";
  if (rank === 1) return "bg-chart-4";
  return "bg-primary/60";
};

const badgeColor = (rank) => {
  if (rank === 0) return "bg-destructive/10 text-destructive border-destructive/20";
  if (rank === 1) return "bg-chart-4/10 text-chart-4 border-chart-4/20";
  return "bg-primary/10 text-primary border-primary/20";
};

/**
 * Determine which predictions count as "significant findings".
 *
 * Strategy: relative significance — a finding is flagged if its score is
 * meaningfully above the background noise of the other predictions.
 * Specifically, a prediction is significant if:
 *   1. Its score is above a minimum noise floor (5%), AND
 *   2. It is at least 1.5× the mean score of all predictions
 *      (i.e. it genuinely stands out), AND
 *   3. At most the top 5 are ever flagged (keeps the summary readable)
 *
 * This adapts naturally to models that score low across the board —
 * if the top result is 18% and everything else is 4%, that 18% is flagged.
 * If everything scores similarly low and bunched together, nothing is flagged.
 */
function getSignificantFindings(predictions) {
  const NOISE_FLOOR = 0.05;        // below 5% is noise regardless
  const RELATIVE_FACTOR = 1.5;     // must be 1.5× the mean to stand out
  const MAX_FLAGGED = 5;

  const mean = predictions.reduce((s, p) => s + p.confidence, 0) / predictions.length;
  const threshold = Math.max(NOISE_FLOOR, mean * RELATIVE_FACTOR);

  return predictions
    .filter(p => p.confidence >= threshold)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, MAX_FLAGGED);
}

export default function ResultsPanel({ results }) {
  if (!results) return null;

  // Show all predictions above 3% noise floor, sorted by confidence
  const allFindings = results.predictions
    .filter(p => p.confidence > 0.03)
    .sort((a, b) => b.confidence - a.confidence);

  const significant = getSignificantFindings(results.predictions);
  const hasFindings = significant.length > 0;
  const topScore = allFindings[0]?.confidence ?? 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="space-y-6"
    >
      {/* Summary Banner */}
      <div className={`rounded-2xl p-5 border ${hasFindings ? "bg-destructive/5 border-destructive/20" : "bg-accent/5 border-accent/20"}`}>
        <div className="flex items-start gap-3">
          {hasFindings
            ? <AlertTriangle className="w-5 h-5 text-destructive mt-0.5 shrink-0" />
            : <CheckCircle2 className="w-5 h-5 text-accent mt-0.5 shrink-0" />
          }
          <div>
            <p className="font-semibold font-space text-foreground">
              {hasFindings
                ? `${significant.length} Significant Finding${significant.length > 1 ? "s" : ""}`
                : "No Significant Findings"
              }
            </p>
            <p className="text-sm text-muted-foreground font-inter mt-1">
              {hasFindings
                ? `Elevated signals detected for: ${significant.map(p => p.disease).join(", ")}`
                : "No condition stands out significantly above baseline across all 14 categories."
              }
            </p>
          </div>
        </div>

        {/* Top score indicator — gives context even when nothing clears 50% */}
        {hasFindings && (
          <div className="mt-3 pt-3 border-t border-destructive/10 flex items-center gap-2">
            <span className="text-xs text-muted-foreground font-inter">Top score:</span>
            <span className="text-xs font-semibold font-space text-foreground">
              {(topScore * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-muted-foreground font-inter">— {significant[0]?.disease}</span>
          </div>
        )}
      </div>

      {/* Detailed Results */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-4 h-4 text-primary" />
          <h3 className="font-semibold font-space text-foreground">Detailed Predictions</h3>
          <span className="text-xs text-muted-foreground font-inter ml-auto">
            Showing {allFindings.length} of 14 conditions
          </span>
        </div>

        <div className="space-y-3">
          {allFindings.map((prediction, index) => {
            const isSig = significant.some(s => s.disease === prediction.disease);
            const sigRank = significant.findIndex(s => s.disease === prediction.disease);
            return (
              <motion.div
                key={prediction.disease}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.05 * index }}
                className="group"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium font-inter ${isSig ? "text-foreground" : "text-muted-foreground"}`}>
                      {prediction.disease}
                    </span>
                    {isSig && (
                      <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md bg-destructive/10 text-destructive font-space">
                        significant
                      </span>
                    )}
                  </div>
                  <Badge
                    variant="outline"
                    className={`text-xs font-inter ${isSig ? badgeColor(sigRank) : "bg-muted/40 text-muted-foreground border-border"}`}
                  >
                    {(prediction.confidence * 100).toFixed(1)}%
                  </Badge>
                </div>
                <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${prediction.confidence * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.08 * index, ease: "easeOut" }}
                    className={`h-full rounded-full ${isSig ? barColor(sigRank) : "bg-muted-foreground/30"}`}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-muted-foreground/60 font-inter leading-relaxed border-t border-border pt-4">
        Significant findings are determined by relative score distribution, not fixed thresholds.
        This is an AI-assisted tool and should not replace professional medical diagnosis.
      </p>
    </motion.div>
  );
}
