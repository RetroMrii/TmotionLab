import { test, expect, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import { advanceWave, REST_RIBBONS, ribbonFrame, waveAmplitude, waveRate, type WaveMotion } from "../src/components/ribbonMotion";

const output = "../reports/engineering/wave-motion/screenshots";
mkdirSync(output, { recursive: true });
const endpoint = "**/v1/goemotions/predict";
const pathData = (page: Page) => page.locator(".emotion-signal path").evaluateAll(nodes => nodes.map(node => node.getAttribute("d")));
const values = (page: Page) => page.locator(".signal-breathe").evaluate(node => ({ rate: Number((node as HTMLElement).dataset.waveRate), amplitude: Number((node as HTMLElement).dataset.waveAmplitude) }));

test("wave field preserves the source artwork and pinned endpoints, with continuous interruption", () => {
  const baseline = JSON.parse(readFileSync("tests/fixtures/ribbon-geometry.json", "utf8"));
  expect(createHash("sha256").update(REST_RIBBONS.join("\n")).digest("hex")).toBe(baseline.sha256);
  const points = (path: string) => [...path.matchAll(/[ML]([\d.-]+),([\d.-]+)/g)].map(match => [Number(match[1]), Number(match[2])]);
  let motion: WaveMotion = { phase: 0, intensity: 0, velocity: 0 };
  let previous = ribbonFrame(motion.phase, 1).map(points);
  let greatestFrameChange = 0;
  for (let frame = 0; frame < 600; frame++) {
    const target = frame >= 60 && frame < 180 || frame >= 195 && frame < 205 ? 1 : 0;
    const next = advanceWave(motion, target, 1 / 60);
    expect(next.phase).toBeGreaterThan(motion.phase);
    expect(next.intensity).toBeGreaterThanOrEqual(0);
    expect(next.intensity).toBeLessThanOrEqual(1);
    motion = next;
    const shape = ribbonFrame(motion.phase, waveAmplitude(motion.intensity)).map(points);
    if (shape.length !== 61) throw new Error("Ribbon count changed");
    shape.forEach((path, i) => {
      if (path.length !== 85 || path[0].join() !== "34,170" || path.at(-1)!.join() !== "406,170") throw new Error("Ribbon samples/endpoints changed");
      const angle = (i < 36 ? -3 : 161) * Math.PI / 180;
      path.forEach(([x,y], j) => {
        greatestFrameChange = Math.max(greatestFrameChange, Math.abs(y - previous[i][j][1]));
        const renderedX=220+Math.cos(angle)*(x-220)-Math.sin(angle)*(y-170);
        const renderedY=170+Math.sin(angle)*(x-220)+Math.cos(angle)*(y-170);
        if (x < 34 || x > 406 || renderedX < 0 || renderedX > 440 || renderedY < 0 || renderedY > 340) throw new Error(`Ribbon ${i} escaped its original viewBox at frame ${frame}: ${renderedX},${renderedY}`);
      });
    });
    previous = shape;
  }
  expect(greatestFrameChange).toBeGreaterThan(.5);
  expect(greatestFrameChange).toBeLessThan(4);
  expect(waveRate(motion.intensity)).toBe(1);
  expect(waveAmplitude(motion.intensity)).toBe(1);
  // At a matched phase, analyzing is exactly the approved 1.75x deformation.
  const rest = REST_RIBBONS.map(points), idle = ribbonFrame(2, 1).map(points), active = ribbonFrame(2, 1.75).map(points);
  let idleEnergy = 0, activeEnergy = 0;
  idle.forEach((path,i) => path.forEach(([,y],j) => {
    idleEnergy += (y-rest[i][j][1])**2; activeEnergy += (active[i][j][1]-rest[i][j][1])**2;
  }));
  expect(Math.sqrt(activeEnergy/idleEnergy)).toBeCloseTo(1.75, 2);
});

for (const width of [1440,390]) for (const theme of ["dark","light"]) {
  test(`flowing geometry idle, analyzing, resolved, error and cancellation ${width} ${theme}`, async ({page}) => {
    test.setTimeout(45000);
    await page.setViewportSize({width,height:900});
    await page.addInitScript(value=>localStorage.setItem("tmotionlab-theme",value),theme);
    await page.emulateMedia({reducedMotion:"reduce"});
    await page.goto("/");
    const signal=page.locator(".emotion-signal");
    await signal.scrollIntoViewIfNeeded();
    await expect.poll(()=>values(page)).toEqual({rate:1,amplitude:1});
    const idle=await pathData(page);
    await page.waitForTimeout(500);
    const evolved=await pathData(page);
    expect(evolved.filter((path,i)=>path!==idle[i])).toHaveLength(61);
    await expect(page.locator(".signal-settle")).toHaveAttribute("transform","rotate(-3 220 170)");
    expect(await page.locator(".signal-breathe").evaluate(node=>getComputedStyle(node).transform)).toBe("none");
    await page.screenshot({path:`${output}/tested-${width}-${theme}-idle.png`});
    let release!:()=>void; let gate=new Promise<void>(resolve=>release=resolve);
    await page.route(endpoint,async route=>{await gate;await route.continue();});
    await page.getByLabel("English text",{exact:true}).fill("I am grateful and happy to meet my friends again.");
    await page.getByRole("button",{name:"Analyze text",exact:true}).click();
    await expect.poll(()=>values(page)).toEqual({rate:3.5,amplitude:1.75});
    await expect(signal).toHaveAttribute("data-state","pending");
    const active=await pathData(page);
    await page.waitForTimeout(300);
    expect(await pathData(page)).not.toEqual(active);
    await page.screenshot({path:`${output}/tested-${width}-${theme}-analyzing.png`});
    release(); await expect(page.locator(".reading")).toBeVisible();
    await signal.scrollIntoViewIfNeeded();
    await expect.poll(()=>values(page)).toEqual({rate:1,amplitude:1});
    await expect(signal).toHaveAttribute("data-state","resolved");
    await page.screenshot({path:`${output}/tested-${width}-${theme}-resolved.png`});
    await page.unroute(endpoint);
    await page.route(endpoint, route=>route.fulfill({status:503,json:{error:{code:"http_503",message:"Unavailable"}}}));
    await page.getByRole("button",{name:"Analyze text",exact:true}).click();
    await expect(page.getByRole("alert")).toBeVisible();
    await expect.poll(()=>values(page)).toEqual({rate:1,amplitude:1});
    await expect(signal).toHaveAttribute("data-state","error");
    await page.screenshot({path:`${output}/tested-${width}-${theme}-error.png`});
    await page.unroute(endpoint);
    gate=new Promise<void>(resolve=>release=resolve);
    await page.route(endpoint,async route=>{await gate;await route.continue().catch(()=>{});});
    await page.getByRole("button",{name:"Analyze text",exact:true}).click();
    await expect.poll(()=>values(page)).toEqual({rate:3.5,amplitude:1.75});
    await page.getByRole("button",{name:"Clear",exact:true}).click();
    release();
    await expect.poll(()=>values(page)).toEqual({rate:1,amplitude:1});
    await expect(signal).toHaveAttribute("data-state","idle");
    await expect(page.locator(".reading")).toHaveCount(0);
    const canceled=await pathData(page);await page.waitForTimeout(300);
    expect(await pathData(page)).not.toEqual(canceled);
    await page.screenshot({path:`${output}/tested-${width}-${theme}-canceled.png`});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  });
}

test("wave work pauses offscreen and stops on unmount without resuming a stale request",async({page})=>{
  await page.setViewportSize({width:1440,height:300});await page.goto("/");
  await page.locator(".emotion-signal").scrollIntoViewIfNeeded();
  await expect.poll(()=>values(page)).toEqual({rate:1,amplitude:1});
  await page.evaluate(()=>window.scrollTo(0,document.body.scrollHeight));await page.waitForTimeout(150);
  const offscreen=await pathData(page);await page.waitForTimeout(300);
  expect(await pathData(page)).toEqual(offscreen);
  await page.locator(".emotion-signal").scrollIntoViewIfNeeded();await page.waitForTimeout(200);
  expect(await pathData(page)).not.toEqual(offscreen);
  // Keep a handle to a detached path: a leaked RAF would keep mutating it.
  const path=await page.locator(".emotion-signal path").first().elementHandle();
  await page.getByRole("button",{name:"Explore",exact:true}).click();
  await page.getByRole("link",{name:/Research overview/}).click();
  await expect(page).toHaveURL(/\/research$/);
  const stopped=await path!.getAttribute("d");await page.waitForTimeout(300);
  expect(await path!.getAttribute("d")).toBe(stopped);
  await path!.dispose();
});
