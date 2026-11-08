import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const output = "../reports/engineering/tidal-dot-field";
mkdirSync(`${output}/screenshots`, { recursive: true });
const endpoint = "**/v1/goemotions/predict";
const pixels = (page: Page) => page.locator(".tidal-field canvas").evaluate(node => (node as HTMLCanvasElement).toDataURL());

for (const width of [1440, 1024, 768, 390]) for (const theme of ["dark", "light"]) {
  test(`tidal environment, input, request, result and navigation ${width} ${theme}`, async ({ page }) => {
    test.setTimeout(45000);
    await page.setViewportSize({ width, height: 900 });
    await page.addInitScript(value => localStorage.setItem("tmotionlab-theme", value), theme);
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto("/");
    await expect(page.locator(".tidal-field")).toHaveAttribute("data-ready", "true");
    const shot = async (state: string) => {
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
      await page.screenshot({ path: `${output}/screenshots/${width}-${theme}-${state}.png`, fullPage: state === "research" });
    };
    await shot("empty");
    const first = await pixels(page);
    await page.waitForTimeout(250);
    expect(await pixels(page)).not.toBe(first);
    expect(await page.locator(".tidal-field").evaluate(node => getComputedStyle(node).pointerEvents)).toBe("none");
    await page.getByLabel("English text", { exact: true }).fill("I feel grateful and happy to see my friends again.");
    await expect(page.getByLabel("English text", { exact: true })).toBeFocused();
    await shot("input");
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    await page.route(endpoint, async route => { await gate; await route.continue(); });
    await page.getByRole("button", { name: "Analyze text", exact: true }).click();
    await page.waitForTimeout(900);
    await expect(page.locator(".emotion-signal")).toHaveAttribute("data-state", "pending");
    await shot("analyzing");
    release();
    await expect(page.locator(".reading")).toBeVisible();
    await page.waitForTimeout(600);
    await shot("result");
    expect(await page.locator(".emotion-signal path").count()).toBe(61);
    const accessibility = await new AxeBuilder({ page }).analyze();
    writeFileSync(`${output}/axe-${width}-${theme}.json`, JSON.stringify(accessibility, null, 2));
    expect(accessibility.violations).toEqual([]);
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
    if (width < 560) await page.getByRole("button", { name: /navigation/i }).click();
    const explore = page.getByRole("button", { name: "Explore", exact: true });
    if (width < 560) await explore.click(); else await explore.hover();
    await expect(explore).toHaveAttribute("aria-expanded", "true");
    await page.waitForTimeout(250); // Capture the settled menu, not its existing entrance fade.
    await shot("menu");
    await page.getByRole("link", { name: /Research overview/ }).click();
    await expect(page).toHaveURL(/\/research$/);
    await expect(page.locator(".tidal-field")).toHaveCount(0);
    await shot("research");
  });
}

test("reduced motion is static, theme changes repaint, resize caps backing resolution", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".tidal-field")).toHaveAttribute("data-ready", "true");
  await page.waitForTimeout(150);
  const first = await pixels(page);
  const ribbon = await page.locator(".emotion-signal path").first().getAttribute("d");
  await page.mouse.move(1100, 250);
  await page.waitForTimeout(400);
  expect(await pixels(page)).toBe(first);
  expect(await page.locator(".emotion-signal path").first().getAttribute("d")).not.toBe(ribbon);
  await page.locator(".theme-toggle").click();
  expect(await pixels(page)).not.toBe(first);
  const second = await pixels(page);
  await page.waitForTimeout(200);
  expect(await pixels(page)).toBe(second);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.waitForTimeout(250);
  expect(await pixels(page)).not.toBe(second);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.locator(".tidal-field canvas").evaluate(node => ({ width: (node as HTMLCanvasElement).width, height: (node as HTMLCanvasElement).height }))).toEqual({ width: 390, height: 844 });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 3, mobile: true });
  expect(await page.evaluate(() => devicePixelRatio)).toBe(3);
  expect(await page.locator(".tidal-field canvas").evaluate(node => (node as HTMLCanvasElement).width)).toBe(390);
});

test("hidden, offscreen and detached canvases stop painting; return mounts one field", async ({ page }) => {
  await page.addInitScript(() => {
    const paints = new Map<HTMLCanvasElement, number>();
    Object.assign(window, { tidalPaints: paints });
    const original = CanvasRenderingContext2D.prototype.clearRect;
    CanvasRenderingContext2D.prototype.clearRect = function (...args) {
      paints.set(this.canvas, (paints.get(this.canvas) ?? 0) + 1);
      return original.apply(this, args);
    };
  });
  await page.goto("/");
  await expect(page.locator(".tidal-field")).toHaveAttribute("data-ready", "true");
  expect(await page.locator(".tidal-field").evaluate(async node => {
    let mutations = 0;
    const observer = new MutationObserver(records => { mutations += records.length; });
    observer.observe(node, { attributes: true, childList: true, subtree: true });
    await new Promise(resolve => setTimeout(resolve, 350));
    observer.disconnect();
    return mutations;
  })).toBe(0); // Canvas animation must not cause per-frame DOM or React updates.
  const count = () => page.evaluate(() => [...(window as unknown as { tidalPaints: Map<HTMLCanvasElement, number> }).tidalPaints.values()].reduce((a, b) => a + b, 0));
  await page.evaluate(() => { Object.defineProperty(document, "hidden", { configurable: true, value: true }); document.dispatchEvent(new Event("visibilitychange")); });
  const hidden = await count(); await page.waitForTimeout(250); expect(await count()).toBe(hidden);
  await page.evaluate(() => { Object.defineProperty(document, "hidden", { configurable: true, value: false }); document.dispatchEvent(new Event("visibilitychange")); });
  await expect.poll(count).toBeGreaterThan(hidden);
  await page.locator(".tidal-field").evaluate(node => { (node as HTMLElement).style.transform = "translateY(200vh)"; });
  await page.waitForTimeout(100);
  const offscreen = await count(); await page.waitForTimeout(250); expect(await count()).toBe(offscreen);
  await page.locator(".tidal-field").evaluate(node => { (node as HTMLElement).style.transform = ""; });
  await expect.poll(count).toBeGreaterThan(offscreen);
  await page.getByRole("link", { name: "About", exact: true }).click();
  const detached = await count(); await page.waitForTimeout(250); expect(await count()).toBe(detached);
  await page.getByRole("link", { name: "Analyze", exact: true }).click();
  await expect(page.locator(".tidal-field")).toHaveCount(1);
  await expect.poll(count).toBeGreaterThan(detached);
});

test("unavailable Canvas and lost context fall back without blocking input", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".tidal-field")).toHaveAttribute("data-ready", "true");
  await page.locator(".tidal-field canvas").dispatchEvent("contextlost");
  await expect(page.locator(".tidal-field")).not.toHaveAttribute("data-ready");
  await page.waitForTimeout(200);
  await expect(page.locator(".tidal-field")).not.toHaveAttribute("data-ready");
  await page.addInitScript(() => { HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext; });
  await page.reload();
  await expect(page.locator(".tidal-field")).not.toHaveAttribute("data-ready");
  expect(await page.locator(".tidal-field").evaluate(node => getComputedStyle(node).backgroundImage)).toContain("radial-gradient");
  await page.getByLabel("English text", { exact: true }).fill("Still usable without Canvas.");
  await expect(page.getByRole("button", { name: "Analyze text", exact: true })).toBeEnabled();
});
