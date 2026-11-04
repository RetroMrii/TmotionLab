import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  workers: 1,
  reporter: "list",
  globalSetup: "./tests/setup.ts",
  use: {
    baseURL: "http://127.0.0.1:5173",
    channel: "chrome",
    headless: true,
    screenshot: "only-on-failure",
  },
});
