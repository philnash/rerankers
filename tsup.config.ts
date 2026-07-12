import { defineConfig } from "tsup";

export default defineConfig({
  clean: true,
  dts: true,
  entry: ["src/index.ts", "src/ai-sdk.ts"],
  format: ["esm"],
  sourcemap: true,
  splitting: false,
});
