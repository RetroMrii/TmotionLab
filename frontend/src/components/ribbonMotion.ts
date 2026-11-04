import { motionTokens } from "../motion";

const TAU = Math.PI * 2;

// Keep the original 61 ribbons and 85 samples per path. Spatial terms are
// computed once; a frame shares four temporal coefficients across every path.
function ribbon(i: number, count: number, offset: number) {
  const spread = (i - (count - 1) / 2) / ((count - 1) / 2);
  return Array.from({ length: 85 }, (_, step) => {
    const t = step / 84;
    const envelope = step === 0 || step === 84 ? 0 : Math.sin(t * Math.PI) ** 0.8;
    // Outer ribbons have less room to bend: taper smoothly rather than clamp
    // coordinates, which would flatten crests or create a velocity discontinuity.
    const waveEnvelope = envelope * (1 - .45 * spread * spread);
    const a = t * TAU * 1.25 + spread * .9 + offset * .4;
    const b = t * TAU * 2 + spread * .65 - offset;
    return {
      prefix: `${step ? "L" : "M"}${(34 + t * 372).toFixed(2)},`,
      y: 170 + spread * 106 * envelope + Math.sin(t * TAU + spread * 2.6 + offset) * 40 * envelope,
      sinA: 14 * waveEnvelope * Math.sin(a), cosA: 14 * waveEnvelope * Math.cos(a),
      sinB: 6 * waveEnvelope * Math.sin(b), cosB: 6 * waveEnvelope * Math.cos(b),
    };
  });
}

const ribbons = [
  ...Array.from({ length: 36 }, (_, i) => ribbon(i, 36, 0)),
  ...Array.from({ length: 25 }, (_, i) => ribbon(i, 25, 1.4)),
];

export function ribbonFrame(phase: number, amplitude: number): string[] {
  const cosA = Math.cos(phase), sinA = Math.sin(phase);
  const cosB = Math.cos(phase * .67), sinB = Math.sin(phase * .67);
  return ribbons.map(points => points.map(p =>
    p.prefix + (p.y + amplitude * (p.sinA * cosA - p.cosA * sinA + p.sinB * cosB + p.cosB * sinB)).toFixed(2),
  ).join(" "));
}

// The undeformed source is retained for initial markup and geometry regression.
export const REST_RIBBONS = ribbonFrame(0, 0);
export type WaveMotion = { phase: number; intensity: number; velocity: number };
export const waveRate = (intensity: number) => 1 + (motionTokens.signalActiveRate - 1) * intensity;
export const waveAmplitude = (intensity: number) => 1 + (motionTokens.signalActiveAmplitude - 1) * intensity;

export function advanceWave(current: WaveMotion, target: 0 | 1, seconds: number): WaveMotion {
  // Analytic critically damped response: retains position AND velocity on a
  // reversal. ~99.95% settled in signalRamp; no phase restart on state changes.
  const omega = 10 / (motionTokens.signalRamp / 1000);
  const error = current.intensity - target;
  const decay = Math.exp(-omega * seconds);
  const travel = (current.velocity + omega * error) * seconds;
  let intensity = target + (error + travel) * decay;
  let velocity = (current.velocity - omega * travel) * decay;
  if (Math.abs(intensity - target) < .0001 && Math.abs(velocity) < .001) {
    intensity = target;
    velocity = 0;
  }
  const rate = (waveRate(current.intensity) + waveRate(intensity)) / 2;
  return { intensity, velocity, phase: current.phase + seconds * TAU * rate / (motionTokens.signalCycle / 1000) };
}
