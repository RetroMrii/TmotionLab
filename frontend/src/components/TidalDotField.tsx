import { memo, useEffect, useRef } from "react";
import "./tidalDotField.css";

type Dot = { x: number; y: number; sin: number; cos: number; crossSin: number; crossCos: number };

/** Decorative environment only. Request state changes emphasis, never data or progress. */
export const TidalDotField = memo(function TidalDotField({ analyzing }: { analyzing: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const target = useRef(false);
  useEffect(() => { target.current = analyzing; }, [analyzing]);

  useEffect(() => {
    const element = host.current;
    const surface = canvas.current;
    if (!element || !surface) return;
    let context: CanvasRenderingContext2D | null;
    try { context = surface.getContext("2d"); } catch { return; }
    if (!context) return; // The CSS dot field remains visible without Canvas.
    const ctx = context;
    const finePointer = matchMedia("(hover: hover) and (pointer: fine)");
    let disposed = false, failed = false, visible = true, running = false;
    let frame = 0, previous = 0, nextPaint = 0;
    let width = 0, height = 0, mobile = false, phase = 0, emphasis = 0;
    let pointerX = 0, pointerY = 0, pointerTime = -Infinity, pointerStrength = 0;
    let dots: Dot[] = [];
    const buckets: Dot[][] = Array.from({ length: 8 }, () => []);
    let colors: string[] = [];

    function stop() {
      running = false;
      cancelAnimationFrame(frame);
      previous = 0;
    }
    function fail() {
      failed = true;
      stop();
      delete element!.dataset.ready;
    }
    function palette() {
      const style = getComputedStyle(element!);
      const teal = style.getPropertyValue("--tidal-teal").trim();
      const slate = style.getPropertyValue("--tidal-slate").trim();
      colors = buckets.map((_, i) => `rgba(${i < 4 ? slate : teal},${.16 + i * .055})`);
    }
    function paint(now: number, seconds = 0) {
      if (disposed || failed || !width || !height) return;
      try {
        emphasis += ((target.current ? 1 : 0) - emphasis) * (1 - Math.exp(-seconds / 1.1));
        phase += seconds * (Math.PI * 2 / 14) * 1.5 * (1 + emphasis * .18);
        const p = phase;
        const sin = Math.sin(p), cos = Math.cos(p);
        const crossSin = Math.sin(p * .73), crossCos = Math.cos(p * .73);
        const pointerTarget = !mobile && finePointer.matches && now - pointerTime < 1200 ? 1 : 0;
        pointerStrength += (pointerTarget - pointerStrength) * (1 - Math.exp(-seconds / .45));
        const usePointer = !mobile && pointerStrength > .01;
        buckets.forEach(bucket => { bucket.length = 0; });
        for (const dot of dots) {
          const wave = (dot.sin * cos - dot.cos * sin + 1) / 2;
          const cross = (dot.crossSin * crossCos - dot.crossCos * crossSin + 1) / 2;
          let value = wave * wave * .72 + cross * cross * .28;
          if (usePointer) {
            const distance = ((dot.x - pointerX) ** 2 + (dot.y - pointerY) ** 2) / (160 * 160);
            if (distance < 1) value += (1 - distance) ** 2 * pointerStrength * .12;
          }
          value = Math.min(1, value * (1 + emphasis * .12));
          buckets[Math.min(7, Math.floor(value * 8))].push(dot);
        }
        ctx.clearRect(0, 0, width, height);
        buckets.forEach((bucket, i) => {
          ctx.beginPath();
          const radius = .85 + i * .18;
          for (const dot of bucket) {
            ctx.moveTo(dot.x + radius, dot.y);
            ctx.arc(dot.x, dot.y, radius, 0, Math.PI * 2);
          }
          ctx.fillStyle = colors[i];
          ctx.fill();
        });
        if (element!.dataset.ready !== "true") element!.dataset.ready = "true";
      } catch { fail(); }
    }
    function tick(now: number) {
      if (!running) return;
      if (now >= nextPaint) {
        const seconds = previous ? Math.min((now - previous) / 1000, .1) : 0;
        previous = now;
        nextPaint = now + 1000 / (mobile ? 10 : 12) - 1;
        paint(now, seconds);
      }
      if (running) frame = requestAnimationFrame(tick);
    }
    function sync() {
      if (disposed || failed) return;
    
      const shouldRun = !document.hidden && visible;
    
      if (!shouldRun) {
        stop();
      } else if (!running) {
        running = true;
        nextPaint = 0;
        frame = requestAnimationFrame(tick);
      }
    }
    function resize() {
      if (disposed || failed) return;
      const bounds = element!.getBoundingClientRect();
      width = bounds.width; height = bounds.height;
      mobile = width <= 768;
      // The environment is deliberately softer than the foreground, including on HiDPI.
      const dpr = Math.min(devicePixelRatio || 1, 1);
      surface!.width = Math.round(width * dpr);
      surface!.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // Bound work on ultrawide screens too; no per-dot DOM or per-frame trigonometry.
      const spacing = Math.max(mobile ? 36 : 30, Math.sqrt(width * height / 1600));
      dots = [];
      for (let y = 13; y < height; y += spacing) for (let x = 13; x < width; x += spacing) {
        const wave = x * .007 + y * .009;
        const cross = x * -.004 + y * .006;
        dots.push({ x, y, sin: Math.sin(wave), cos: Math.cos(wave), crossSin: Math.sin(cross), crossCos: Math.cos(cross) });
      }
      if (!document.hidden && visible) paint(performance.now());
    }
    function pointer(event: PointerEvent) {
      if (mobile || !finePointer.matches || event.pointerType !== "mouse") return;
      pointerX = event.clientX; pointerY = event.clientY; pointerTime = performance.now();
    }
    function resetPointer() { pointerTime = -Infinity; }
    function themeChanged() {
      if (disposed || failed) return;
      palette();
      if (!document.hidden && visible) paint(performance.now());
    }
    palette();
    resize();
    const sizeObserver = new ResizeObserver(resize);
    sizeObserver.observe(element);
    const visibilityObserver = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    visibilityObserver.observe(element);
    const themeObserver = new MutationObserver(themeChanged);
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    finePointer.addEventListener("change", resetPointer);
    document.addEventListener("visibilitychange", sync);
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", pointer, { passive: true });
    document.addEventListener("pointerleave", resetPointer);
    window.addEventListener("blur", resetPointer);
    surface.addEventListener("contextlost", fail);
    sync();
    return () => {
      disposed = true;
      stop();
      sizeObserver.disconnect(); visibilityObserver.disconnect(); themeObserver.disconnect();
      finePointer.removeEventListener("change", resetPointer);
      document.removeEventListener("visibilitychange", sync);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", pointer);
      document.removeEventListener("pointerleave", resetPointer);
      window.removeEventListener("blur", resetPointer);
      surface.removeEventListener("contextlost", fail);
    };
  }, []);

  return <div className="tidal-field" ref={host} aria-hidden="true">
    <canvas ref={canvas} />
  </div>;
});
