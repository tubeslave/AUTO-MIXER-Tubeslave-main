import express from "express";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, basename, extname } from "node:path";
import { tmpdir } from "node:os";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { analyzeTrack, ensureFfmpeg } from "./analyzer.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, "..");
const UI_URI = "ui://gpt-mixing-cloud/control-panel-v0.5";
const VERSION = "0.5.0";

const OpenAIFile = z.object({
  download_url: z.string().url(),
  file_id: z.string().min(1),
  mime_type: z.string().optional(),
  file_name: z.string().optional(),
}).strict();

type OpenAIFileValue = z.infer<typeof OpenAIFile>;

const ANALYSIS_OUTPUT = z.object({
  schemaVersion: z.string(),
  status: z.string(),
  readinessScore: z.number(),
  tracks: z.array(z.any()),
  summary: z.object({
    tracksAnalyzed: z.number(),
    tracksWithoutSignal: z.array(z.string()),
    clippingTracks: z.array(z.string()),
    suspectedBleedTracks: z.array(z.string()),
  }),
});

function safeName(name: string | undefined, index: number) {
  const raw = basename(name || `track-${index + 1}.wav`);
  const cleaned = raw.replace(/[^\p{L}\p{N}._()\- ]/gu, "_").slice(0, 180);
  return cleaned || `track-${index + 1}.wav`;
}

async function downloadToFile(source: OpenAIFileValue, dest: string) {
  const response = await fetch(source.download_url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`Не удалось получить ${source.file_name || source.file_id}: HTTP ${response.status}`);
  }
  const length = Number(response.headers.get("content-length") || "0");
  const max = Number(process.env.MAX_AUDIO_BYTES || 1024 * 1024 * 1024);
  if (length > max) throw new Error(`Файл больше лимита ${Math.round(max / 1024 / 1024)} MB.`);
  await pipeline(Readable.fromWeb(response.body as any), createWriteStream(dest));
}

function readinessFromTracks(tracks: any[]) {
  if (!tracks.length) return 0;
  let score = 100;
  const noSignal = tracks.filter((t) => !t.signal?.hasSignal);
  const clipping = tracks.filter((t) => t.signal?.clippingSuspected);
  const bleed = tracks.filter((t) => t.bleed?.status === "suspected");
  score -= noSignal.length * 20;
  score -= clipping.length * 12;
  score -= bleed.length * 3;
  return Math.max(0, Math.min(100, score));
}

function createServer() {
  const server = new McpServer(
    { name: "GPT MIXING ОБЛАКО", version: VERSION },
    { instructions: "Use open_mixing_panel to show the mobile UI. Use analyze_files for non-destructive analysis of user audio files. Never claim DSP edits occurred unless a dedicated editing tool reports success." },
  );

  server.registerResource("mixing-control-panel", UI_URI, {}, async () => ({
    contents: [{
      uri: UI_URI,
      mimeType: "text/html;profile=mcp-app",
      text: await readFile(join(ROOT, "public", "index.html"), "utf8"),
      _meta: {
        "openai/widgetDescription": "Мобильная панель GPT MIXING ОБЛАКО для выбора аудиофайлов и запуска анализа.",
        ui: { prefersBorder: false, csp: { connectDomains: [], resourceDomains: [] } },
      },
    }],
  }));

  server.registerTool("open_mixing_panel", {
    title: "Открыть панель сведения",
    description: "Use this when the user wants to open the GPT MIXING ОБЛАКО control panel.",
    inputSchema: {},
    outputSchema: { status: z.string(), version: z.string(), readinessScore: z.number() },
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    _meta: { ui: { resourceUri: UI_URI } },
  }, async () => ({
    structuredContent: { status: "ready", version: VERSION, readinessScore: 0 },
    content: [{ type: "text", text: "Панель GPT MIXING ОБЛАКО открыта." }],
    _meta: { ui: { resourceUri: UI_URI } },
  }));

  server.registerTool("analyze_files", {
    title: "Анализировать аудиодорожки",
    description: "Use this when the user presses АНАЛИЗ or asks to analyze attached audio tracks. Measures signal presence, clipping, EBU R128 loudness, dynamics, silence, spectral statistics, tentative source class, and conservative bleed risk without modifying source files.",
    inputSchema: { files: z.array(OpenAIFile).min(1).max(32) },
    outputSchema: ANALYSIS_OUTPUT.shape,
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    _meta: {
      "openai/fileParams": ["files"],
      ui: { resourceUri: UI_URI },
      "openai/toolInvocation/invoking": "Анализирую дорожки…",
      "openai/toolInvocation/invoked": "Анализ завершён",
    },
  }, async ({ files }) => {
    const workDir = await mkdtemp(join(tmpdir(), "gpt-mixing-cloud-"));
    try {
      const localPaths: string[] = [];
      for (let i = 0; i < files.length; i++) {
        const source = files[i];
        const name = safeName(source.file_name, i);
        const extension = extname(name) || ".wav";
        const dest = join(workDir, `${String(i + 1).padStart(2, "0")}-${basename(name, extension)}${extension}`);
        await downloadToFile(source, dest);
        localPaths.push(dest);
      }
      const tracks = [];
      for (const path of localPaths) tracks.push(await analyzeTrack(path));
      const publicTracks = tracks.map((t, i) => ({ ...t, path: undefined, file: files[i]?.file_name || t.file, fileId: files[i]?.file_id }));
      const readinessScore = readinessFromTracks(publicTracks);
      const summary = {
        tracksAnalyzed: publicTracks.length,
        tracksWithoutSignal: publicTracks.filter((t) => !t.signal?.hasSignal).map((t) => t.file),
        clippingTracks: publicTracks.filter((t) => t.signal?.clippingSuspected).map((t) => t.file),
        suspectedBleedTracks: publicTracks.filter((t) => t.bleed?.status === "suspected").map((t) => t.file),
      };
      const structuredContent = { schemaVersion: VERSION, status: "complete", readinessScore, tracks: publicTracks, summary };
      return {
        structuredContent,
        content: [{ type: "text", text: `Анализ завершён: ${summary.tracksAnalyzed} дорожек. Готовность ${readinessScore}/100. Клиппинг: ${summary.clippingTracks.length}; без сигнала: ${summary.tracksWithoutSignal.length}.` }],
      };
    } catch (error: any) {
      return {
        structuredContent: { schemaVersion: VERSION, status: "error", readinessScore: 0, tracks: [], summary: { tracksAnalyzed: 0, tracksWithoutSignal: [], clippingTracks: [], suspectedBleedTracks: [] } },
        content: [{ type: "text", text: `Анализ не выполнен: ${String(error?.message || error)}` }],
        isError: true,
      };
    } finally {
      await rm(workDir, { recursive: true, force: true });
    }
  });

  return server;
}

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "2mb" }));
app.get("/health", (_req, res) => res.json({ ok: true, app: "GPT MIXING ОБЛАКО", version: VERSION, ffmpegRequired: true }));
app.all("/mcp", async (req, res) => {
  const server = createServer();
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on("close", () => {
    transport.close().catch(() => undefined);
    server.close().catch(() => undefined);
  });
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

await ensureFfmpeg();
const port = Number(process.env.PORT || 3000);
app.listen(port, "0.0.0.0", () => console.log(`GPT MIXING ОБЛАКО ${VERSION}: http://0.0.0.0:${port}/mcp`));
