import raw from "./research.json";
export type Strategy = "uniform" | "rare4x" | "synthetic" | "hybrid";
export type Metric = "macro_f1" | "micro_f1" | "macro_average_precision";
export type RareLabel = "grief" | "nervousness" | "pride" | "relief";
export interface LabelMetric {
  f1: number;
  precision: number;
  recall: number;
  support: number;
  average_precision?: number;
}
export interface Run {
  strategy: Strategy;
  seed: number;
  id: string;
  threshold: number;
  metrics: Record<Metric, number>;
  perLabel: Record<string, LabelMetric>;
  checkpoint: string;
  optimizerSteps: number;
  provenance: {
    model_id: string;
    model_revision: string;
    seed: number;
    base_config_sha256: string;
  };
  sources: string[];
}
export const research = raw;
export const runs = raw.runs as Run[];
export const strategies: Strategy[] = [
  "uniform",
  "rare4x",
  "synthetic",
  "hybrid",
];
export const rareLabels: RareLabel[] = [
  "grief",
  "nervousness",
  "pride",
  "relief",
];
export const strategyNames: Record<Strategy, string> = {
  uniform: "Uniform",
  rare4x: "Real 4×",
  synthetic: "Synthetic",
  hybrid: "Hybrid",
};
export const metricNames: Record<Metric, string> = {
  macro_f1: "Macro F1",
  micro_f1: "Micro F1",
  macro_average_precision: "Macro AP",
};
export const fmt = (value: number, digits = 4) => value.toFixed(digits);
export function statistics(values: number[]) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return {
    mean,
    sd: Math.sqrt(
      values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1),
    ),
  };
}
