export const motionTokens = {
  easeOut: "cubic-bezier(.23,1,.32,1)",
  easeInOut: "cubic-bezier(.77,0,.175,1)",
  heading: 280,
  workspace: 320,
  result: 280,
  signal: 450,
  signalCycle: 10000,
  signalRamp: 650,
  signalActiveRate: 3.5,
  signalActiveAmplitude: 1.75,
  analysisPresentation: 1500,
  resultDelays: [0, 50, 100, 150, 230],
} as const;
// Supervisor-approved full-motion application policy (2026-11-04).
// System/browser preferences are not modified; no media-query suppression is applied.
export const keyboardInput = () =>
  document.documentElement.dataset.input === "keyboard";
