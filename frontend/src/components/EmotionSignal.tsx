import { memo, useEffect, useId, useRef } from "react";
import { advanceWave, ribbonFrame, waveAmplitude, waveRate, type WaveMotion } from "./ribbonMotion";

const INITIAL_RIBBONS = ribbonFrame(0, 1);
const MAIN = INITIAL_RIBBONS.slice(0, 36);
const ECHO = INITIAL_RIBBONS.slice(36);
export const EmotionSignal = memo(function EmotionSignal({
  state,
}: {
  state: "idle" | "pending" | "resolved" | "error";
}) {
  const id = useId().replaceAll(":", "");
  const ribbonGroup = useRef<HTMLDivElement>(null);
  const target = useRef<0 | 1>(0);
  useEffect(() => { target.current = state === "pending" ? 1 : 0; }, [state]);
  useEffect(() => {
    const node = ribbonGroup.current;
    if (!node) return;
    const paths = [...node.querySelectorAll("path")];
    let motion: WaveMotion = { phase: 0, intensity: 0, velocity: 0 };
    let frame = 0;
    let previous = 0;
    let nextPaint = 0;
    let visible = true;
    let active = true;
    let running = false;
    const draw = (now: number) => {
      if (!running) return;
      frame = requestAnimationFrame(draw);
      if (now < nextPaint) return;
      const seconds = previous ? Math.min((now - previous) / 1000, .064) : 0;
      previous = now;
      nextPaint = Math.max(nextPaint + 1000 / 60, now);
      motion = advanceWave(motion, target.current, seconds);
      const amplitude = waveAmplitude(motion.intensity);
      const shape = ribbonFrame(motion.phase, amplitude);
      paths.forEach((path, index) => path.setAttribute("d", shape[index]));
      // Bounded-rate diagnostics for inspecting state transitions; never UI progress.
      const rate = waveRate(motion.intensity).toFixed(3);
      const amount = amplitude.toFixed(3);
      if (node.dataset.waveRate !== rate) node.dataset.waveRate = rate;
      if (node.dataset.waveAmplitude !== amount) node.dataset.waveAmplitude = amount;
    };
    const syncVisibility = () => {
      if (!active) return;
      const shouldRun = !document.hidden && visible;
      if (shouldRun === running) return;
      running = shouldRun;
      if (running) {
        previous = 0;
        nextPaint = 0;
        frame = requestAnimationFrame(draw);
      } else cancelAnimationFrame(frame);
    };
    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      syncVisibility();
    });
    observer.observe(node);
    document.addEventListener("visibilitychange", syncVisibility);
    syncVisibility();
    return () => {
      active = false;
      running = false;
      observer.disconnect();
      document.removeEventListener("visibilitychange", syncVisibility);
      cancelAnimationFrame(frame);
    };
  }, []);
  return (
    <div
      className="emotion-signal"
      data-state={state}
      aria-hidden="true"
    >
      <div className="signal-breathe" ref={ribbonGroup}>
      <svg className="signal-art" viewBox="0 0 440 340" aria-hidden="true">
        <defs>
          <linearGradient id={`${id}-cool`}>
            <stop stopColor="var(--signal-cool)" stopOpacity=".13" />
            <stop
              offset=".48"
              stopColor="var(--signal-cool)"
              stopOpacity=".8"
            />
            <stop offset="1" stopColor="var(--signal-warm)" stopOpacity=".2" />
          </linearGradient>
          <linearGradient id={`${id}-warm`}>
            <stop stopColor="var(--signal-warm)" stopOpacity=".1" />
            <stop
              offset=".5"
              stopColor="var(--signal-warm)"
              stopOpacity=".45"
            />
            <stop offset="1" stopColor="var(--signal-cool)" stopOpacity=".16" />
          </linearGradient>
        </defs>
        <g
          className="signal-settle"
          transform="rotate(-3 220 170)"
        >
          <g>
            <g fill="none" stroke={`url(#${id}-cool)`} strokeWidth=".65">
              {MAIN.map((d, i) => (
                <path d={d} key={i} />
              ))}
            </g>
            <g
              fill="none"
              stroke={`url(#${id}-warm)`}
              strokeWidth=".55"
              transform="rotate(164 220 170)"
            >
              {ECHO.map((d, i) => (
                <path d={d} key={i} />
              ))}
            </g>
          </g>
        </g>
      </svg>
      </div>
    </div>
  );
});
