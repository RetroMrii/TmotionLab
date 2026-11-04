import { createServer } from "vite";
// In-process QA server avoids Windows child-process teardown limitations.
// It exercises a real unreachable API origin without stopping the live API.
export default async function setup() {
  const server = await createServer({
    server: { host: "127.0.0.1", port: 5174, strictPort: true, open: false },
    define: {
      "import.meta.env.VITE_API_BASE_URL": JSON.stringify(
        "http://127.0.0.1:65530",
      ),
    },
  });
  await server.listen();
  return async () => {
    await server.close();
  };
}
