import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
const endpoint = "**/v1/goemotions/predict";
const labels = [
  "admiration",
  "amusement",
  "anger",
  "annoyance",
  "approval",
  "caring",
  "confusion",
  "curiosity",
  "desire",
  "disappointment",
  "disapproval",
  "disgust",
  "embarrassment",
  "excitement",
  "fear",
  "gratitude",
  "grief",
  "joy",
  "love",
  "nervousness",
  "optimism",
  "pride",
  "realization",
  "relief",
  "remorse",
  "sadness",
  "surprise",
  "neutral",
];
// Isolated browser fixtures, never imported by the production application.
function fixture(text: string, detected: string[] = ["joy", "neutral"]) {
  const scores = labels.map((label, label_id) => ({
    label,
    label_id,
    score: detected.includes(label) ? 0.5 : 0.1,
  }));
  return {
    schema_version: "1",
    task: "multi_label",
    model_id: "roberta_goemotions_rare4x_v001",
    dataset: "goemotions_v001",
    seed: 42,
    threshold: 0.25,
    score_type: "sigmoid",
    device: "cpu",
    input: {
      text,
      tokens: ["<s>", text, "</s>"],
      token_count: 3,
      original_token_count: 3,
      max_tokens: 48,
      truncated: false,
    },
    scores,
    detected_labels: detected,
    ranked_scores: [...scores].sort((a, b) => b.score - a.score).slice(0, 5),
    score_note: "Independent sigmoid scores are not calibrated confidence.",
    availability_notes: [],
  };
}
async function submit(page: Page, text = "I love this, thank you so much!") {
  await page.getByLabel("English text", { exact: true }).fill(text);
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
}
async function theme(page: Page, value: string) {
  await page.addInitScript(
    (v) => localStorage.setItem("tmotionlab-theme", v),
    value,
  );
}
async function overflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
}
const output = "../reports/engineering/ux-refinement/screenshots";
mkdirSync(output, { recursive: true });

async function signalRate(page: Page) {
  return page.locator(".signal-breathe").evaluate(el => Number((el as HTMLElement).dataset.waveRate));
}

for (const color of ["dark", "light"])
  for (const width of [1440, 1024, 768, 390]) {
    test(`matrix ${width} ${color}: all six primary routes and live result`, async ({
      page,
    }) => {
      test.setTimeout(60000);
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(e.message));
      await page.setViewportSize({ width, height: 900 });
      await theme(page, color);
      for (const path of [
        "/",
        "/predict",
        "/compare",
        "/research",
        "/research/experiments",
        "/about",
      ]) {
        await page.goto(path);
        await expect(page.locator("h1")).toBeVisible();
        await page.evaluate(() => document.fonts.ready);
        await overflow(page);
        if (path === "/compare") {
          await page
            .getByRole("button", { name: "Load comparison", exact: true })
            .click();
          await expect(page.locator(".comparison-row")).toHaveCount(4);
        }
        if (path === "/research/experiments")
          await expect(page.locator(".run-table tbody tr")).toHaveCount(12);
        await page.screenshot({
          animations: "disabled",
          path: `${output}/matrix-${path.replaceAll("/", "-") || "home"}-${width}-${color}.png`,
          fullPage: true,
        });
        if (path === "/research" && [1440, 390].includes(width)) {
          await page
            .locator(".paper-heading")
            .screenshot({
              path: `${output}/paper-opening-${width}-${color}.png`,
            });
          await page
            .locator("#results")
            .screenshot({
              path: `${output}/paper-results-${width}-${color}.png`,
            });
          await page
            .locator("#rare-labels")
            .screenshot({ path: `${output}/paper-rare-${width}-${color}.png` });
        }
        if (path === "/research/experiments" && [1440, 390].includes(width)) {
          await page.getByLabel("Mean ± SD", { exact: true }).check();
          await page
            .locator("#validation")
            .screenshot({ path: `${output}/aggregate-${width}-${color}.png` });
          await page
            .locator("#rare-labels")
            .screenshot({ path: `${output}/rare-${width}-${color}.png` });
          await page
            .locator("#heldout")
            .screenshot({ path: `${output}/heldout-${width}-${color}.png` });
        }
      }
      await page.goto("/");
      await submit(page);
      await expect(page.locator(".reading")).toBeVisible();
      await expect(page.locator(".detected-labels")).toContainText("gratitude");
      await expect(page.locator(".detected-labels")).toContainText("love");
      await overflow(page);
      await page.screenshot({
        animations: "disabled",
        path: `${output}/result-${width}-${color}.png`,
        fullPage: true,
      });
      await page.locator(".reading-details > summary").click();
      await expect(page.locator(".score-row")).toHaveCount(28);
      await overflow(page);
      await page.screenshot({
        animations: "disabled",
        path: `${output}/details-${width}-${color}.png`,
        fullPage: true,
      });
      expect(errors).toEqual([]);
    });
  }
test("initial empty, presets fill only, clear focuses editor and blank stays disabled", async ({
  page,
}) => {
  let requests = 0;
  await page.route(endpoint, (route) => {
    requests++;
    return route.fulfill({ json: fixture("test") });
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Analyze text", exact: true }),
  ).toBeDisabled();
  await expect(page.getByLabel("English text", { exact: true })).toHaveValue(
    "",
  );
  await page.getByRole("button", { name: "A small joy", exact: true }).click();
  expect(requests).toBe(0);
  await expect(page.getByLabel("English text", { exact: true })).toContainText(
    "",
  );
  await expect(page.getByLabel("English text", { exact: true })).toHaveValue(
    "I’m so glad we finally made time for this. It means more than you know.",
  );
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await expect(page.getByLabel("English text", { exact: true })).toBeFocused();
  await page.getByLabel("English text", { exact: true }).fill("   ");
  await expect(
    page.getByRole("button", { name: "Analyze text", exact: true }),
  ).toBeDisabled();
});
for (const detected of [[], ["joy"], ["joy", "neutral"]])
  test(`detection semantics ${detected.join(",") || "none"}`, async ({
    page,
  }) => {
    await page.route(endpoint, (route) =>
      route.fulfill({ json: fixture("fixture", detected) }),
    );
    await page.goto("/");
    await submit(page, "fixture");
    await expect(page.locator(".reading")).toBeVisible();
    if (detected.length)
      await expect(page.locator(".detected-labels")).toHaveText(
        detected.join(" · "),
      );
    else
      await expect(
        page.getByText("No labels reached the threshold.", { exact: false }),
      ).toBeVisible();
    await expect(page.getByRole("link", { name: /interpret/i })).toHaveCount(0);
  });
test("ranked response sorts a copy; filters invalid values; does not pad empty or fewer results", async ({
  page,
}) => {
  let mode = 0;
  await page.route(endpoint, (route) => {
    const data = fixture("test", []);
    data.ranked_scores =
      mode === 0
        ? [
            { label: "joy", label_id: 17, score: 0.1 },
            { label: "neutral", label_id: 27, score: 0.24 },
            { label: "anger", label_id: 2, score: -1 },
            { label: "love", label_id: 18, score: 2 },
            { label: "fear", label_id: 14, score: null as unknown as number },
          ]
        : [];
    return route.fulfill({ json: data });
  });
  await page.goto("/");
  await submit(page, "test");
  await expect(page.locator(".leading-reading h3")).toHaveText("neutral");
  await expect(page.locator(".alternative")).toHaveCount(1);
  await expect(page.locator(".leading-reading")).toContainText(
    "Ranked alternative",
  );
  mode = 1;
  await submit(page, "empty response ranking");
  await expect(page.locator(".empty-ranking")).toContainText(
    "No valid ranked scores",
  );
  await expect(page.locator(".leading-reading")).toHaveCount(0);
});
test("malformed response fails honestly without showing stale result", async ({
  page,
}) => {
  await page.route(endpoint, (route) => route.fulfill({ json: { bad: true } }));
  await page.goto("/");
  await submit(page, "Preserve this");
  await expect(page.getByRole("alert")).toContainText("usable reading");
  await expect(page.getByLabel("English text", { exact: true })).toHaveValue(
    "Preserve this",
  );
  await expect(page.locator(".reading")).toHaveCount(0);
});
test("live truncation shows actual analyzed tokens and original text", async ({
  page,
}) => {
  await page.goto("/");
  const text = "wonderful ".repeat(500);
  await submit(page, text);
  await page.locator(".reading-details > summary").click();
  await expect(page.getByRole("note")).toContainText("truncated to 48 tokens");
  await expect(page.getByLabel("English text", { exact: true })).toHaveValue(
    text,
  );
  await expect(page.locator("blockquote.exact-text")).toHaveText(text.trim());
});
for (const action of ["edit", "clear", "preset", "navigate", "new-failure"])
  test(`stale response guard after ${action}`, async ({ page }) => {
    let release!: () => void, received!: () => void, finished!: () => void;
    const gate = new Promise<void>((r) => (release = r)),
      incoming = new Promise<void>((r) => (received = r)),
      done = new Promise<void>((r) => (finished = r));
    await page.route(endpoint, async (route) => {
      const text = route.request().postDataJSON().text;
      if (text === "Old passage") {
        received();
        await gate;
        await route.fulfill({ json: fixture(text, ["anger"]) }).catch(() => {});
        finished();
      } else if (action === "new-failure") {
        await route.fulfill({
          status: 503,
          json: { error: { message: "Unavailable", code: "http_503" } },
        });
      } else await route.fulfill({ json: fixture(text, ["joy"]) });
    });
    await page.goto("/");
    await submit(page, "Old passage");
    await incoming;
    await expect(page.locator(".emotion-signal")).toHaveAttribute(
      "data-state",
      "pending",
    );
    if (action === "clear")
      await page.getByRole("button", { name: "Clear", exact: true }).click();
    else if (action === "preset")
      await page
        .getByRole("button", { name: "Mixed feelings", exact: true })
        .click();
    else if (action === "navigate")
      await page.getByRole("link", { name: "About", exact: true }).click();
    else {
      await page
        .getByLabel("English text", { exact: true })
        .fill("New passage");
      await expect(page.locator(".reading")).toHaveCount(0);
      await page
        .getByRole("button", { name: "Analyze text", exact: true })
        .click();
      if (action === "new-failure")
        await expect(page.getByRole("alert")).toBeVisible();
      else await expect(page.locator(".detected-labels")).toHaveText("joy");
    }
    release();
    await done;
    if (action === "edit")
      await expect(page.locator(".detected-labels")).toHaveText("joy");
    else await expect(page.locator(".reading")).toHaveCount(0);
  });
test("ambient signal accelerates during request and returns to idle after failure", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  let count = 0;
  await page.route(endpoint, async (route) => {
    count++;
    await gate;
    await route.fulfill({
      status: 503,
      json: { error: { code: "http_503", message: "Unavailable" } },
    });
  });
  await page.goto("/");
  await expect.poll(() => signalRate(page)).toBe(1);
  await submit(page, "Keep my words");
  await expect(
    page.getByRole("button", { name: "Reading your text…", exact: true }),
  ).toBeDisabled();
  await expect.poll(() => signalRate(page)).toBe(3.5);
  await page.screenshot({
    animations: "disabled",
    path: `${output}/pending-desktop-dark.png`,
    fullPage: true,
  });
  release();
  await expect(page.getByRole("alert")).toContainText("Your text is preserved");
  expect(count).toBe(1);
  await expect.poll(() => signalRate(page)).toBe(1);
  await expect(page.getByLabel("English text", { exact: true })).toHaveValue(
    "Keep my words",
  );
  await page.unroute(endpoint);
  await page.route(endpoint, (r) => r.abort());
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("couldn’t reach");
});
for (const width of [390, 1440])
  test(`keyboard navigation focus Escape and theme persistence ${width}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/");
    if (width === 390) {
      await page.getByRole("button", { name: "Toggle navigation" }).focus();
      await page.keyboard.press("Enter");
    }
    const trigger = page.getByRole("button", { name: "Explore", exact: true });
    await trigger.focus();
    await page.keyboard.press("Enter");
    await expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(
      await page
        .locator(".explore-menu")
        .evaluate((el) => getComputedStyle(el).transitionDuration),
    ).toBe("0s");
    await page.keyboard.press("Tab");
    await expect(
      page.getByRole("link", { name: /Research overview/ }),
    ).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(trigger).toBeFocused();
    await expect(page.locator(".explore-menu")).toHaveAttribute("inert", "");
    if (width === 390) {
      await page.keyboard.press("Escape");
      await expect(
        page.getByRole("button", { name: "Toggle navigation" }),
      ).toBeFocused();
      await page.getByRole("button", { name: "Toggle navigation" }).click();
    }
    await trigger.click();
    await page.getByRole("link", { name: /Research overview/ }).click();
    await expect(page).toHaveURL("/research");
    await expect(page.locator("main")).toBeFocused();
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  });
test("outside click and focus leaving close Explore", async ({ page }) => {
  await page.goto("/");
  const trigger = page.getByRole("button", { name: "Explore", exact: true });
  await trigger.click();
  await page.locator("h1").click();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await trigger.click();
  await page.getByLabel("English text", { exact: true }).focus();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
});
for (const reduced of [true, false])
  test(`keyboard result focus with system reduced preference ${reduced}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 700 });
    await page.emulateMedia({
      reducedMotion: reduced ? "reduce" : "no-preference",
    });
    await page.route(endpoint, (r) => r.fulfill({ json: fixture("keyboard") }));
    await page.goto("/");
    await page.getByLabel("English text", { exact: true }).fill("keyboard");
    await page
      .getByRole("button", { name: "Analyze text", exact: true })
      .focus();
    await page.keyboard.press("Enter");
    await expect(page.locator(".reading")).toBeFocused();
    expect(
      await page
        .locator(".reading")
        .evaluate((el) => el.getBoundingClientRect().top),
    ).toBeGreaterThanOrEqual(0);
    await expect(page.locator(".emotion-signal path")).toHaveCount(61);
    expect(
      await page
        .locator(".reading")
        .evaluate((el) => el.getAnimations({ subtree: true }).length),
    ).toBe(0);
  });
test("research works without API, chart/table controls use actual run data", async ({
  page,
}) => {
  await page.route("http://127.0.0.1:8000/**", (r) => r.abort());
  await page.goto("/research");
  await expect(
    page.getByRole("heading", { name: "One final evaluation." }),
  ).toBeVisible();
  await page.goto("/research/experiments");
  await expect(page.locator(".run-table tbody tr")).toHaveCount(12);
  await expect(page.locator("#historical")).toContainText("unavailable");
  const data = JSON.parse(readFileSync("src/data/research.json", "utf8"));
  await page.getByLabel("Metric", { exact: true }).selectOption("micro_f1");
  await expect(page.locator(".run-table tbody tr").first()).toContainText(
    data.runs[0].metrics.micro_f1.toFixed(6),
  );
  await page.getByLabel("Mean ± SD", { exact: true }).check();
  await expect(page.locator(".run-table tbody tr")).toHaveCount(4);
  await page.getByRole("button", { name: "Compare Real 4× & Hybrid" }).click();
  await expect(page.locator(".run-table tbody tr")).toHaveCount(2);
  await expect(page.locator(".run-table tbody tr").first()).toContainText(
    data.summary.rare4x.micro_f1.mean.toFixed(6),
  );
  await expect(page.locator(".run-table tbody tr").first()).toContainText(
    data.summary.rare4x.micro_f1.sample_std.toFixed(6),
  );
  await page.getByLabel("Emotion", { exact: true }).selectOption("grief");
  await expect(
    page
      .getByRole("region", { name: "Rare-label per-seed results" })
      .locator("tbody tr"),
  ).toHaveCount(6);
  await page.screenshot({
    animations: "disabled",
    path: `${output}/explorer-aggregate-offline.png`,
    fullPage: true,
  });
  for (const label of ["Real 4×", "Hybrid"])
    await page.getByRole("checkbox", { name: label, exact: true }).uncheck();
  await expect(
    page.getByText("Select a strategy to inspect its runs."),
  ).toBeVisible();
  await page.goto(
    "/research/sources/reports/goemotions/final_heldout_test_v001.json",
  );
  await expect(page.locator("body")).toContainText("0.5236535323415689");
});
test("compatibility routes and unavailable interpretation", async ({
  page,
}) => {
  for (const path of ["/errors", "/metrics"]) {
    await page.goto(path);
    await expect(page).toHaveURL("/research/experiments#historical");
    await expect(
      page.getByRole("heading", { name: "Historical evidence." }),
    ).toBeVisible();
  }
  await page.goto("/interpret");
  await expect(page.locator("h1")).toHaveText(
    "This view has reached its limit.",
  );
  await page.goto("/not-a-route");
  await expect(page.locator("h1")).toHaveText("There’s no reading here.");
  await page.screenshot({
    animations: "disabled",
    path: `${output}/fallback.png`,
    fullPage: true,
  });
});
test("comparison clears old results on edit and unavailable replacements", async ({
  page,
}) => {
  await page.goto("/compare");
  await page
    .getByRole("button", { name: "Load comparison", exact: true })
    .click();
  await expect(page.locator(".comparison-row")).toHaveCount(4);
  await page.getByLabel("Saved example ID").fill("not-an-example");
  await expect(page.locator(".comparison-result")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Load comparison", exact: true })
    .click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.locator(".comparison-result")).toHaveCount(0);
});
for (const color of ["dark", "light"])
  test(`axe populated routes and menu ${color}`, async ({ page }) => {
    test.setTimeout(120000);
    await theme(page, color);
    await page.goto("/");
    await submit(page);
    await page.locator(".reading-details > summary").click();
    async function scan(name: string) {
      await page.evaluate(() =>
        Promise.all(
          document
            .getAnimations()
            .filter(
              (a) => a.effect?.getComputedTiming().iterations !== Infinity,
            )
            .map((a) => a.finished.catch(() => {})),
        ),
      );
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      writeFileSync(
        `${output}/axe-${name.replaceAll("/", "-").replaceAll(" ", "-")}-${color}.json`,
        JSON.stringify(
          {
            url: page.url(),
            violations: results.violations,
            passes: results.passes.length,
            incomplete: results.incomplete.map((item) => ({
              id: item.id,
              nodes: item.nodes.map((node) => node.target),
            })),
          },
          null,
          2,
        ),
      );
      expect(
        results.violations,
        `${name}: ${JSON.stringify(results.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })))}`,
      ).toEqual([]);
    }
    await scan("reading disclosure");
    await page.getByRole("button", { name: "Explore", exact: true }).click();
    await scan("open navigation");
    await page.screenshot({
      animations: "disabled",
      path: `${output}/menu-${color}.png`,
    });
    for (const path of [
      "/compare",
      "/research",
      "/research/experiments",
      "/about",
    ]) {
      await page.goto(path);
      if (path === "/compare") {
        await page
          .getByRole("button", { name: "Load comparison", exact: true })
          .click();
        await expect(page.locator(".comparison-row")).toHaveCount(4);
      }
      if (path === "/research/experiments") {
        await page.getByText("Inspect 42 / Uniform", { exact: true }).click();
      }
      await scan(path);
    }
  });

test("real connection refusal preserves text and static research remains available", async ({
  page,
}) => {
  await page.goto("http://127.0.0.1:5174/");
  await expect(page.locator(".service-status")).toContainText(
    "status unavailable",
  );
  await submit(page, "This text stays here.");
  await expect(page.getByRole("alert")).toContainText(
    "couldn’t reach the reading service",
  );
  await expect(page.getByLabel("English text", { exact: true })).toHaveValue(
    "This text stays here.",
  );
  await page.screenshot({
    path: `${output}/real-service-unavailable.png`,
    fullPage: true,
  });
  await page.goto("http://127.0.0.1:5174/research/experiments");
  await expect(page.locator(".run-table tbody tr")).toHaveCount(12);
  await expect(page.locator("#historical")).toContainText("unavailable");
});
for (const color of ["dark", "light"])
  test(`mobile Explore and full pointer motion despite system reduction ${color}`, async ({
    page,
  }) => {
    await theme(page, color);
    await page.setViewportSize({ width: 390, height: 900 });
    await page.goto("/");
    await page.getByRole("button", { name: "Toggle navigation" }).click();
    await page.getByRole("button", { name: "Explore", exact: true }).click();
    await overflow(page);
    await page.screenshot({
      animations: "disabled",
      path: `${output}/mobile-menu-${color}.png`,
    });
    await page.keyboard.press("Escape");
    await page.keyboard.press("Escape");
    await page.emulateMedia({ reducedMotion: "reduce" });
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    await page.route(endpoint, async (route) => {
      await gate;
      await route.fulfill({ json: fixture("reduced") });
    });
    await submit(page, "reduced");
    await expect(page.locator(".emotion-signal")).toHaveAttribute(
      "data-state",
      "pending",
    );
    await expect.poll(() => signalRate(page)).toBe(3.5);
    await page.screenshot({
      path: `${output}/pending-mobile-reduced-${color}.png`,
      fullPage: true,
    });
    release();
    await expect(page.locator(".reading")).toBeVisible();
    expect(
      await page
        .locator(".reading")
        .evaluate((el) => el.getAnimations({ subtree: true }).length),
    ).toBeGreaterThan(0);
  });
