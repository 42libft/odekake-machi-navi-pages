import assert from "node:assert/strict";
import { readdir, readFile, stat } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
const text = (path) => readFile(new URL(path, root), "utf8");
const bytes = (path) => stat(new URL(path, root)).then((result) => result.size);

const publicSourceFiles = [
  "index.html",
  "src/main.tsx",
  "src/App.tsx",
  "src/styles.css",
  "vite.config.ts",
  "package.json",
  "README.md",
  "THIRD-PARTY-NOTICES.md",
  ".github/workflows/pages.yml",
];

test("builds a static entry point with relative asset URLs", async () => {
  const html = await text("dist/index.html");
  assert.match(html, /<div id="root"><\/div>/);
  assert.match(html, /\.\/assets\//);
  assert.doesNotMatch(html, /https?:\/\//i);
});

test("keeps the public source free of private infrastructure and external runtime calls", async () => {
  const source = await Promise.all(publicSourceFiles.map(text));
  const joined = source.join("\n");
  assert.doesNotMatch(joined, /https?:\/\/|fetch\s*\(|XMLHttpRequest|sendBeacon|localStorage|sessionStorage|indexedDB|document\.cookie/i);
  assert.match(joined, /スタッフ作成／くら寿司公式ではありません／公式予約・順番待ちには接続しません/);
  assert.match(joined, /TextDetector/);
  assert.match(joined, /cacheMethod: "none"/);
  assert.match(joined, /import\.meta\.env\.BASE_URL/);
  assert.match(joined, /MIN_GROUP_GAP = 39/);
  assert.match(joined, /MAX_GROUP_GAP = 45/);
  assert.match(joined, /Math\.random\(\)/);
});

test("ships the bundled OCR worker, core, data, and license notices", async () => {
  const [worker, core, simdCore, relaxedSimdCore, languageData] = await Promise.all([
    bytes("public/ocr/worker.min.js"),
    bytes("public/ocr/tesseract-core-lstm.wasm.js"),
    bytes("public/ocr/tesseract-core-simd-lstm.wasm.js"),
    bytes("public/ocr/tesseract-core-relaxedsimd-lstm.wasm.js"),
    bytes("public/ocr/eng.traineddata.gz"),
  ]);
  assert.ok(worker > 50_000);
  assert.ok(core > 1_000_000);
  assert.ok(simdCore > 1_000_000);
  assert.ok(relaxedSimdCore > 1_000_000);
  assert.ok(languageData > 1_000_000);
  assert.match(await text("LICENSES/APACHE-2.0.txt"), /Apache License/i);
  assert.match(await text("LICENSES/MIT.txt"), /MIT License/);
  assert.match(await text("LICENSES/tesseract-worker.min.js.LICENSE.txt"), /ieee754/);
});

test("contains only the approved public project surfaces", async () => {
  const allowed = new Set([
    ".github",
    ".gitignore",
    "LICENSES",
    "README.md",
    "THIRD-PARTY-NOTICES.md",
    "eslint.config.mjs",
    "index.html",
    "package-lock.json",
    "package.json",
    "public",
    "src",
    "tests",
    "tsconfig.json",
    "vite.config.ts",
  ]);
  const entries = new Set((await readdir(root)).filter((entry) => ![".git", "dist", "node_modules"].includes(entry)));
  assert.deepEqual([...entries].sort(), [...allowed].sort());
});
