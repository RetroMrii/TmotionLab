import { test, expect, chromium, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { REST_RIBBONS } from "../src/components/ribbonMotion";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const output = "../reports/engineering/ux-refinement";
mkdirSync(`${output}/screenshots`, { recursive: true });
const endpoint = "**/v1/goemotions/predict";
const labels: string[] = JSON.parse(readFileSync("../data/processed/goemotions_v001/manifest.json", "utf8")).label_names;
function fixture(text: string) {
  const scores = labels.map((label, label_id) => ({ label, label_id, score: (label_id + 30) / 100 }));
  return {
    schema_version: "1", task: "multi_label", model_id: "roberta_goemotions_rare4x_v001",
    dataset: "goemotions_v001", seed: 42, threshold: .25, score_type: "sigmoid", device: "cpu",
    input: { text, tokens: ["<s>", text, "</s>"], token_count: 3, original_token_count: 3, max_tokens: 48, truncated: false },
    scores, ranked_scores: [...scores].sort((a,b) => b.score-a.score).slice(0,5),
    detected_labels: [...labels], score_note: "Independent, uncalibrated sigmoid scores.", availability_notes: [],
  };
}
const approved = [
  ["uplift", "amusement excitement joy optimism pride relief", "#D7C38F", "#756019"],
  ["connection", "admiration caring gratitude love", "#A9C6B7", "#396D5C"],
  ["appraisal", "approval disapproval", "#C5BACF", "#6B527E"],
  ["longing", "desire", "#D3B2C3", "#86516F"],
  ["reflection", "disappointment grief remorse sadness", "#B0BCD7", "#52658B"],
  ["alarm", "embarrassment fear nervousness", "#D1B297", "#875B38"],
  ["friction", "anger annoyance disgust", "#D3ABA4", "#8F5148"],
  ["orientation", "confusion curiosity realization surprise", "#9FC6CB", "#386E76"],
  ["neutral", "neutral", "#ACB7BF", "#52636D"],
];
declare global {
  interface Window {
    __ux: {
      submitted: number; requested: number; completed: number; reveals: number[];
      scrolls: { delta: number; expected: number; before: number; behavior: string | undefined }[];
      frames: { t: number; y: number }[]; focusCalls: (FocusOptions | undefined)[];
    };
  }
}
async function instrument(page: Page) {
  await page.addInitScript(() => {
    const log: Window["__ux"] = { submitted: 0, requested: 0, completed: 0, reveals: [], scrolls: [], frames: [], focusCalls: [] };
    window.__ux = log;
    document.addEventListener("submit", () => { if (!log.submitted) log.submitted = performance.now(); }, true);
    const fetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const prediction = String(args[0]).includes("/v1/goemotions/predict");
      if (prediction) log.requested = performance.now();
      try { return await fetch(...args); }
      finally { if (prediction) log.completed = performance.now(); }
    };
    const scroll = window.scrollBy.bind(window);
    window.scrollBy = ((options: ScrollToOptions) => {
      const rects = [...document.querySelectorAll(".leading-reading h3, .leading-score strong, .empty-ranking")].map(el => el.getBoundingClientRect());
      const top = Math.min(...rects.map(r => r.top));
      const bottom = Math.max(...rects.map(r => r.bottom));
      const viewTop = (visualViewport?.offsetTop ?? 0) + 24;
      const viewBottom = (visualViewport?.offsetTop ?? 0) + (visualViewport?.height ?? innerHeight) - 24;
      const expected = top < viewTop || bottom - top > viewBottom-viewTop ? top-viewTop : Math.max(0, bottom-viewBottom);
      log.scrolls.push({ delta: options.top ?? 0, expected, before: scrollY, behavior: options.behavior });
      scroll(options);
    }) as typeof window.scrollBy;
    const focus = HTMLElement.prototype.focus;
    HTMLElement.prototype.focus = function(options) {
      if (this.classList.contains("reading")) log.focusCalls.push(options);
      return focus.call(this, options);
    };
    let observed: Element | null = null;
    new MutationObserver(() => {
      const section = document.querySelector(".reading");
      if (section && section !== observed) {
        const start = performance.now();
        log.reveals.push(start);
        log.frames.push({ t: start, y: scrollY });
        const sample = (t: number) => {
          log.frames.push({ t, y: scrollY });
          if (t-start < 1100) requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
      }
      observed = section;
    }).observe(document, { childList: true, subtree: true });
  });
}
async function submit(page: Page, text = "A test passage") {
  await page.getByLabel("English text", { exact: true }).fill(text);
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
}
const rate = (page: Page) => page.locator(".signal-breathe").evaluate(el => Number((el as HTMLElement).dataset.waveRate));

test("fast inference starts immediately and reveals at 1.5s, including duplicate keyboard activation", async ({ page }) => {
  await instrument(page);
  let requests = 0;
  await page.route(endpoint, route => { requests++; return route.fulfill({ json: fixture("fast") }); });
  await page.goto("/");
  await page.getByLabel("English text", { exact: true }).fill("fast");
  const button = page.locator('.editor button[type="submit"]');
  await button.focus();
  await page.keyboard.press("Enter");
  await expect(button).toBeFocused();
  await page.keyboard.press("Enter");
  await page.waitForTimeout(350);
  await expect(page.locator(".reading")).toHaveCount(0);
  await expect(page.locator(".emotion-signal")).toHaveAttribute("data-state", "pending");
  await expect(button).toHaveAttribute("aria-disabled", "true");
  await expect(page.locator(".reading")).toBeFocused();
  const log = await page.evaluate(() => window.__ux);
  expect(requests).toBe(1);
  expect(log.requested - log.submitted).toBeLessThan(150);
  expect(log.completed - log.submitted).toBeLessThan(1000);
  expect(log.reveals[0] - log.submitted).toBeGreaterThanOrEqual(1490);
  expect(log.reveals[0] - log.submitted).toBeLessThan(2200);
  expect(log.focusCalls).toEqual([{ preventScroll: true }]);
  writeFileSync(`${output}/timing-fast.json`, JSON.stringify(log, null, 2));
});

test("slow inference reveals upon completion without another presentation delay", async ({ page }) => {
  await instrument(page);
  let release!: () => void;
  const gate = new Promise<void>(r => release=r);
  await page.route(endpoint, async route => { await gate; await route.fulfill({ json: fixture("slow") }); });
  await page.goto("/"); await submit(page, "slow");
  await page.waitForTimeout(1700);
  await expect(page.locator(".reading")).toHaveCount(0);
  await expect(page.locator(".emotion-signal")).toHaveAttribute("data-state", "pending");
  release();
  await expect(page.locator(".reading")).toBeVisible();
  const log = await page.evaluate(() => window.__ux);
  expect(log.reveals[0] - log.submitted).toBeGreaterThan(1700);
  expect(log.reveals[0] - log.completed).toBeLessThan(250);
  writeFileSync(`${output}/timing-slow.json`, JSON.stringify(log, null, 2));
});

for (const action of ["edit", "clear", "preset", "navigate", "replacement-error"]) {
  test(`cancels the completed-inference presentation wait on ${action}`, async ({ page }) => {
    await instrument(page);
    let count = 0;
    await page.route(endpoint, route => {
      count++;
      return count > 1 && action === "replacement-error"
        ? route.fulfill({ status: 503, json: { error: { message: "Unavailable" } } })
        : route.fulfill({ json: fixture(route.request().postDataJSON().text) });
    });
    await page.goto("/");
    const response = page.waitForResponse(/\/v1\/goemotions\/predict/);
    await submit(page, "old"); await response;
    await expect.poll(() => page.evaluate(() => window.__ux.completed)).toBeGreaterThan(0);
    await expect(page.locator(".reading")).toHaveCount(0);
    if (action === "clear") await page.getByRole("button", { name: "Clear", exact: true }).click();
    else if (action === "preset") await page.getByRole("button", { name: "A small joy", exact: true }).click();
    else if (action === "navigate") await page.getByRole("link", { name: "About", exact: true }).click();
    else await page.getByLabel("English text", { exact: true }).fill("new");
    if (action === "replacement-error") await page.getByRole("button", { name: "Analyze text", exact: true }).click();
    await page.waitForTimeout(1700);
    await expect(page.locator(".reading")).toHaveCount(0);
    if (action === "replacement-error") await expect(page.getByRole("alert")).toBeVisible();
    else if (action !== "navigate") {
      await expect(page.locator(".emotion-signal")).toHaveAttribute("data-state", "idle");
      await expect.poll(() => rate(page)).toBe(1);
      await submit(page, "fresh");
      await expect(page.locator(".reading")).toBeVisible();
      await page.locator(".reading-details > summary").click();
      await expect(page.locator("blockquote.exact-text")).toHaveText("fresh");
    }
  });
}

test("Compare has one contextual entry; deep links and compatibility redirects survive", async ({ page }) => {
  for (const path of ["/", "/predict", "/about", "/research", "/research/experiments", "/interpret", "/compare?example=dair_test_001837"]) {
    await page.goto(path);
    const links = page.locator('a[href^="/compare"]');
    await expect(links).toHaveCount(path === "/" || path === "/predict" ? 1 : 0);
    if (path === "/" || path === "/predict") {
      await expect(links).toHaveText("Compare historical benchmarks");
      await expect(links).toHaveAttribute("aria-describedby", "historical-context");
    }
  }
  await expect(page.getByLabel("Saved example ID")).toHaveValue("dair_test_001837");
  await page.getByRole("button", { name: "Load comparison", exact: true }).click();
  await expect(page.locator(".comparison-row")).toHaveCount(4);
  for (const route of ["/errors", "/metrics"]) {
    await page.goto(route);
    await expect(page).toHaveURL("/research/experiments#historical");
  }
});

test("hover corridor, focus retention, pinned click and Escape suppression", async ({ page }) => {
  await page.setViewportSize({ width:1440, height:900 });
  await page.goto("/");
  const trigger = page.getByRole("button", { name:"Explore", exact:true });
  await trigger.hover();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".explore-menu a")).toHaveCount(2);
  const box = (await trigger.boundingBox())!;
  await page.mouse.move(box.x+box.width/2, box.y+box.height+3, { steps:10 });
  await page.waitForTimeout(230);
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("link", { name:/Research overview/ }).hover();
  await page.getByRole("link", { name:/Research overview/ }).focus();
  await page.mouse.move(100, 300);
  await page.waitForTimeout(230);
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await trigger.hover();
  await page.keyboard.press("Escape");
  await expect(trigger).toBeFocused();
  await page.waitForTimeout(230);
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await page.mouse.move(100, 300);
  await trigger.hover();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await trigger.click(); // pin the hover-open panel
  await page.mouse.move(100,300);
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("link", { name:"About", exact:true }).focus();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await trigger.focus(); await page.keyboard.press("Space");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name:/Research overview/ })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL("/research");
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
});

test.describe("touch navigation", () => {
  test.use({ viewport:{width:390,height:844}, hasTouch:true, isMobile:true });
  for (const color of ["dark","light"]) test(`touch toggle, outside dismissal and navigation ${color}`, async ({ page }) => {
    await page.addInitScript(value => localStorage.setItem("tmotionlab-theme",value),color);
    await page.goto("/");
    const outer = page.getByRole("button", {name:"Toggle navigation"});
    const trigger = page.getByRole("button", {name:"Explore",exact:true});
    await outer.tap(); await trigger.tap();
    await expect(trigger).toHaveAttribute("aria-expanded","true");
    await trigger.tap(); await expect(trigger).toHaveAttribute("aria-expanded","false");
    await trigger.tap();
    await page.screenshot({path:`${output}/screenshots/touch-menu-${color}.png`});
    await page.locator(".editor-rail").tap();
    await expect(outer).toHaveAttribute("aria-expanded","false");
    await outer.tap(); await trigger.tap();
    await page.getByRole("link", {name:/Experiment explorer/}).tap();
    await expect(page).toHaveURL("/research/experiments");
    await expect(outer).toHaveAttribute("aria-expanded","false");
  });
});

for (const reduced of [false,true]) test(`61 original paths; continuous phase and full motion with system reduction ${reduced}`, async ({ page }) => {
  await page.emulateMedia({reducedMotion:reduced?"reduce":"no-preference"});
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-motion","full");
  const paths=await page.locator(".emotion-signal path").evaluateAll(xs=>xs.map(x=>x.getAttribute("d")));
  const before=JSON.parse(readFileSync("tests/fixtures/ribbon-geometry.json","utf8"));
  expect(paths).toHaveLength(61);
  expect(createHash("sha256").update(REST_RIBBONS.join("\n")).digest("hex")).toBe(before.sha256);
  await expect(page.locator(".signal-copy, .signal-caption, .signal-status")).toHaveCount(0);
  await expect(page.locator(".signal-disclaimer")).toContainText("not model activity");
  const firstPath=page.locator(".emotion-signal path").first();
  const initial=await firstPath.getAttribute("d");
  const transform=await page.locator(".signal-breathe").evaluate(el=>getComputedStyle(el).transform);
  await page.waitForTimeout(350);
  expect(await firstPath.getAttribute("d")).not.toBe(initial);
  expect(await page.locator(".signal-breathe").evaluate(el=>getComputedStyle(el).transform)).toBe(transform);
  await page.keyboard.press("Tab");
  const afterKey=await firstPath.getAttribute("d");
  await page.waitForTimeout(200);
  expect(await firstPath.getAttribute("d")).not.toBe(afterKey);
  let release!:()=>void; const gate=new Promise<void>(r=>release=r);
  await page.route(endpoint, async route=>{await gate;await route.fulfill({json:fixture("signal")});});
  await submit(page,"signal");
  await page.waitForTimeout(200);
  const middle=await rate(page); expect(middle).toBeGreaterThan(1); expect(middle).toBeLessThan(3.5);
  await expect.poll(()=>rate(page)).toBe(3.5);
  await page.emulateMedia({reducedMotion:reduced?"no-preference":"reduce"});
  expect(await rate(page)).toBe(3.5);
  await page.getByRole("button",{name:"Clear",exact:true}).click();
  await expect.poll(()=>rate(page)).toBe(1);
  release();
  await page.waitForTimeout(1600);
  await expect(page.locator(".reading")).toHaveCount(0);
});

for (const color of ["dark","light"]) test(`all 28 approved colors and unchanged numerical semantics ${color}`, async ({page})=>{
  await page.addInitScript(value=>localStorage.setItem("tmotionlab-theme",value),color);
  await page.route(endpoint,route=>route.fulfill({json:fixture("colors")}));
  await page.goto("/"); await submit(page,"colors");
  await page.locator(".reading-details > summary").click();
  const rows=await page.locator(".score-row").evaluateAll(xs=>xs.map(el=>({
    label:el.firstElementChild?.textContent,
    family:el.getAttribute("data-emotion-family"),
    color:getComputedStyle(el.firstElementChild!).color,
    bar:getComputedStyle(el.querySelector(".score-track > span")!).backgroundColor,
    value:el.querySelector("b")?.textContent,
    width:(el.querySelector(".score-track > span") as HTMLElement).style.width,
  })));
  expect(rows.map(r=>r.label)).toEqual(labels);
  const rgb=(hex:string)=>`rgb(${hex.slice(1).match(/../g)!.map(x=>parseInt(x,16)).join(", ")})`;
  for(const [i,row] of rows.entries()){
    const group=approved.find(([,members])=>members.split(" ").includes(row.label!))!;
    expect(row.family).toBe(group[0]);
    expect(row.color).toBe(rgb(group[color==="dark"?2:3]));
    expect(row.bar).toBe(row.color);
    expect(row.value).toBe(`${(i+30).toFixed(1)}%`);
    expect(parseFloat(row.width)).toBeCloseTo(i+30,5);
  }
  const names=await page.locator(".detected-labels [data-emotion-family]").allTextContents();
  expect(names).toEqual(labels);
  const coloredLabels=await page.locator(".leading-reading h3, .alternative strong, .detected-labels [data-emotion-family]").evaluateAll(xs=>xs.map(el=>({label:el.textContent,color:getComputedStyle(el).color})));
  for(const item of coloredLabels){
    const group=approved.find(([,members])=>members.split(" ").includes(item.label!))!;
    expect(item.color).toBe(rgb(group[color==="dark"?2:3]));
  }
  await expect(page.locator(".detections")).toContainText("threshold 0.25");
  await expect(page.locator(".leading-reading h3")).toHaveText("neutral");
  await expect(page.locator(".leading-reading .detection-tag")).toHaveText("Detected");
  await page.getByText("About label colors",{exact:true}).click();
  await page.screenshot({animations:"disabled",path:`${output}/screenshots/all-colors-${color}.png`,fullPage:true});
  writeFileSync(`${output}/colors-${color}.json`,JSON.stringify(rows,null,2));
});

for(const color of ["dark","light"]) test(`narrow long emotion names remain readable ${color}`,async({page})=>{
  await page.setViewportSize({width:320,height:700});
  await page.addInitScript(value=>localStorage.setItem("tmotionlab-theme",value),color);
  const data=fixture("long labels");
  const leaders=["disappointment","embarrassment","nervousness"];
  data.scores=data.scores.map(item=>({...item,score:leaders.includes(item.label)?.8-leaders.indexOf(item.label)*.1:item.label==="neutral"?.35:.05}));
  data.ranked_scores=[...data.scores].sort((a,b)=>b.score-a.score).slice(0,5);
  data.detected_labels=data.scores.filter(item=>item.score>=.25).map(item=>item.label);
  await page.route(endpoint,route=>route.fulfill({json:data}));
  await page.goto("/");await submit(page,"long labels");
  await expect(page.locator(".leading-reading h3")).toHaveText("disappointment");
  await page.waitForTimeout(500);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
  const labelHeights=await page.locator(".leading-reading h3, .alternative strong").evaluateAll(nodes=>nodes.map(node=>({height:node.getBoundingClientRect().height,lineHeight:parseFloat(getComputedStyle(node).lineHeight)})));
  for(const label of labelHeights) expect(label.height).toBeLessThanOrEqual(label.lineHeight+1);
  await page.screenshot({animations:"disabled",path:`${output}/screenshots/long-labels-320-${color}.png`,fullPage:true});
});

for (const width of [1440,390]) for (const mode of ["pointer","keyboard","already-visible"]) {
  test(`minimum automatic result scroll ${width} ${mode}`, async ({page})=>{
    await page.setViewportSize({width,height:mode==="already-visible"?1400:844});
    await instrument(page);
    await page.route(endpoint,route=>route.fulfill({json:fixture("scroll")}));
    await page.goto("/");
    await page.getByLabel("English text",{exact:true}).fill("scroll");
    const button=page.getByRole("button",{name:"Analyze text",exact:true});
    if(mode==="keyboard"){await button.focus();await page.keyboard.press("Enter");}
    else await button.click();
    await expect(page.locator(".reading")).toBeVisible();
    await page.waitForTimeout(1150);
    const log=await page.evaluate(()=>window.__ux);
    expect(log.scrolls).toHaveLength(mode==="already-visible"?0:1);
    for(const scroll of log.scrolls){expect(scroll.behavior).toBe("smooth");expect(scroll.delta).toBeCloseTo(scroll.expected,1);expect(scroll.delta).toBeLessThan(550);}
    if(mode==="keyboard")expect(log.focusCalls).toEqual([{preventScroll:true}]);
    else expect(log.focusCalls).toEqual([]);
    const rects=await page.locator(".leading-reading h3, .leading-score strong").evaluateAll(xs=>xs.map(x=>({top:x.getBoundingClientRect().top,bottom:x.getBoundingClientRect().bottom})));
    for(const rect of rects){expect(rect.top).toBeGreaterThanOrEqual(23);expect(rect.bottom).toBeLessThanOrEqual((mode==="already-visible"?1400:844)-23);}
    await expect(page.getByRole("button",{name:/View reading/i})).toHaveCount(0);
    writeFileSync(`${output}/scroll-${width}-${mode}.json`,JSON.stringify(log,null,2));
    await page.screenshot({path:`${output}/screenshots/viewport-${width}-${mode}.png`});
  });
}

test("native smooth scrolling also traverses intermediate positions when Chrome enables it",async()=>{
  const browser=await chromium.launch({channel:"chrome",headless:true,args:["--enable-smooth-scrolling"]});
  try{
    const context=await browser.newContext({viewport:{width:390,height:844},reducedMotion:"reduce"});
    const page=await context.newPage();await instrument(page);
    await page.route(endpoint,route=>route.fulfill({json:fixture("native smooth")}));
    await page.goto("http://127.0.0.1:5173/");
    await page.evaluate(()=>Promise.all(document.getAnimations().map(animation=>animation.finished.catch(()=>{}))));
    const placement=[await page.evaluate(()=>({stage:"loaded",y:scrollY,time:performance.now()}))];
    await page.getByLabel("English text",{exact:true}).fill("native smooth");
    placement.push(await page.evaluate(()=>({stage:"filled",y:scrollY,time:performance.now()})));
    // Exercise a real visible-pointer click without Playwright's protocol-level
    // scroll-into-view prelude, which can relocate this smooth-enabled viewport.
    const button=await page.getByRole("button",{name:"Analyze text",exact:true}).boundingBox();
    expect(button).not.toBeNull();
    expect(button!.y).toBeGreaterThan(0);expect(button!.y+button!.height).toBeLessThan(844);
    await page.mouse.click(button!.x+button!.width/2,button!.y+button!.height/2);
    placement.push(await page.evaluate(()=>({stage:"clicked",y:scrollY,time:performance.now()})));
    await expect(page.locator(".reading")).toBeVisible();await page.waitForTimeout(1150);
    const log=await page.evaluate(()=>window.__ux);
    writeFileSync(`${output}/scroll-native-enabled.json`,JSON.stringify({...log, placement, viewport:await page.evaluate(()=>({width:innerWidth,height:innerHeight,hidden:document.hidden,scrollY,anchors:[...document.querySelectorAll(".leading-reading h3, .leading-score strong")].map(node=>({top:node.getBoundingClientRect().top,bottom:node.getBoundingClientRect().bottom}))}))},null,2));
    expect(log.scrolls).toHaveLength(1);
    expect(new Set(log.frames.map(f=>f.y)).size).toBeGreaterThan(5);
  }finally{await browser.close();}
});
