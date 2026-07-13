import { rm } from "node:fs/promises";

import { modelCacheDirectory } from "./model-benchmark-config.ts";

await rm(modelCacheDirectory, { force: true, recursive: true });
console.log(`Removed benchmark model cache: ${modelCacheDirectory}`);
