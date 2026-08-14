"use client";

import { useEffect, useMemo, useState } from "react";

type DetectedText = { rawValue?: string };
type LocalTextDetector = {
  detect: (source: ImageBitmap) => Promise<DetectedText[]>;
};
type OcrWord = {
  text?: string;
  confidence?: number;
  bbox?: { x0: number; y0: number; x1: number; y1: number };
  symbols?: OcrSymbol[];
};
type OcrSymbol = { text?: string; confidence?: number; bbox?: OcrWord["bbox"] };
type OcrLine = { text?: string; words?: OcrWord[]; bbox?: OcrWord["bbox"] };
type OcrBlock = { text?: string; paragraphs?: Array<{ lines?: OcrLine[] }>; bbox?: OcrWord["bbox"] };
type BundledOcrResult = {
  data?: { text?: string; blocks?: OcrBlock[] };
};
type BundledOcrWorker = {
  setParameters: (params: Record<string, string>) => Promise<unknown>;
  recognize: (
    image: File,
    options?: Record<string, unknown>,
    output?: Record<string, boolean>,
  ) => Promise<BundledOcrResult>;
};

declare global {
  interface Window {
    TextDetector?: new () => LocalTextDetector;
  }
}

const digitsOnly = (value: string) => value.replace(/[^0-9]/g, "").slice(0, 4);

const OCR_BASE_PATH = import.meta.env.BASE_URL;
const OCR_WORKER_PATH = `${OCR_BASE_PATH}ocr/worker.min.js`;
const OCR_CORE_PATH = `${OCR_BASE_PATH}ocr`;
const OCR_LANGUAGE_PATH = `${OCR_BASE_PATH}ocr`;

let bundledOcrWorkerPromise: Promise<BundledOcrWorker> | null = null;

const extractCandidate = (rawText: string) => {
  const candidates = rawText.match(/\d{1,4}/g) ?? [];
  return candidates.find((value) => Number(value) > 0) ?? null;
};

const selectBundledCandidate = (result: BundledOcrResult) => {
  const lineCandidates = (result.data?.blocks ?? [])
    .flatMap((block) => block.paragraphs?.flatMap((paragraph) => paragraph.lines ?? []) ?? [])
    .map((line) => {
      const symbols = (line.words ?? [])
        .flatMap((word) => word.symbols ?? [])
        .filter((symbol) => /^\d$/.test(symbol.text ?? ""));
      if (symbols.length > 0) {
        const maxSymbolHeight = Math.max(...symbols.map((symbol) => (
          symbol.bbox ? Math.max(1, symbol.bbox.y1 - symbol.bbox.y0) : 1
        )));
        const minSymbolTop = Math.min(...symbols.map((symbol) => symbol.bbox?.y0 ?? 0));
        const mainSymbols = symbols
          .filter((symbol) => {
            const height = symbol.bbox ? Math.max(1, symbol.bbox.y1 - symbol.bbox.y0) : 1;
            const top = symbol.bbox?.y0 ?? minSymbolTop;
            return height >= maxSymbolHeight * 0.55 && top <= minSymbolTop + maxSymbolHeight * 0.45;
          })
          .sort((left, right) => (left.bbox?.x0 ?? 0) - (right.bbox?.x0 ?? 0));
        const value = mainSymbols.map((symbol) => symbol.text ?? "").join("");
        if (value.length > 0 && value.length <= 4) {
          const width = mainSymbols.reduce((sum, symbol) => sum + (
            symbol.bbox ? Math.max(1, symbol.bbox.x1 - symbol.bbox.x0) : 1
          ), 0);
          const confidence = mainSymbols.reduce((sum, symbol) => sum + (
            Number.isFinite(symbol.confidence) ? Math.max(1, symbol.confidence ?? 1) : 1
          ), 0) / mainSymbols.length;
          return { value, score: maxSymbolHeight * width * confidence * value.length };
        }
      }

      const words = (line.words ?? [])
        .filter((word) => /^\d{1,4}$/.test(word.text ?? ""))
        .sort((left, right) => (left.bbox?.x0 ?? 0) - (right.bbox?.x0 ?? 0));
      if (words.length === 0) return null;

      const maxHeight = Math.max(...words.map((word) => (
        word.bbox ? Math.max(1, word.bbox.y1 - word.bbox.y0) : 1
      )));
      const mainWords = words.filter((word) => {
        const height = word.bbox ? Math.max(1, word.bbox.y1 - word.bbox.y0) : 1;
        return height >= maxHeight * 0.55;
      });
      const value = mainWords.map((word) => word.text ?? "").join("");
      if (!value || value.length > 4) return null;
      const width = mainWords.reduce((sum, word) => sum + (
        word.bbox ? Math.max(1, word.bbox.x1 - word.bbox.x0) : 1
      ), 0);
      const confidence = mainWords.reduce((sum, word) => sum + (
        Number.isFinite(word.confidence) ? Math.max(1, word.confidence ?? 1) : 1
      ), 0) / mainWords.length;
      return { value, score: maxHeight * width * confidence * value.length };
    })
    .filter((candidate): candidate is { value: string; score: number } => candidate !== null)
    .sort((left, right) => right.score - left.score || right.value.length - left.value.length);

  return lineCandidates[0]?.value ?? extractCandidate(result.data?.text ?? "");
};

const createBundledOcrWorker = async (): Promise<BundledOcrWorker> => {
  const { createWorker } = await import("tesseract.js");
  const worker = await createWorker("eng", 1, {
    workerPath: OCR_WORKER_PATH,
    corePath: OCR_CORE_PATH,
    langPath: OCR_LANGUAGE_PATH,
    gzip: true,
    cacheMethod: "none",
    workerBlobURL: false,
    logger: () => {},
  });

  await worker.setParameters({
    tessedit_char_whitelist: "0123456789",
    tessedit_pageseg_mode: "11",
    user_defined_dpi: "300",
  });
  return worker;
};

const getBundledOcrWorker = () => {
  if (!bundledOcrWorkerPromise) {
    bundledOcrWorkerPromise = createBundledOcrWorker().catch((error) => {
      bundledOcrWorkerPromise = null;
      throw error;
    });
  }
  return bundledOcrWorkerPromise;
};

const runBundledOcr = async (file: File) => {
  const worker = await getBundledOcrWorker();
  const result = await worker.recognize(file, {}, { text: true, blocks: true });
  return selectBundledCandidate(result);
};

const MIN_GROUP_GAP = 39;
const MAX_GROUP_GAP = 45;
const MINUTES_PER_NUMBER = 10;
type InputMode = "auto" | "manual";

const toNumber = (value: string) => {
  if (!value) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const formatWait = (minutes: number) => {
  if (minutes <= 0) return "まもなく";
  if (minutes < 60) return `約${minutes}分`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `約${hours}時間${rest}分` : `約${hours}時間`;
};

const pickInitialGap = () =>
  Math.floor(Math.random() * (MAX_GROUP_GAP - MIN_GROUP_GAP + 1)) + MIN_GROUP_GAP;
const estimateCurrent = (ticket: number, gap: number) => Math.max(0, ticket - gap);
const estimateWait = (ticket: number, current: number) =>
  Math.max(0, ticket - current) * MINUTES_PER_NUMBER;

export default function Home() {
  const [ticketNumber, setTicketNumber] = useState("");
  const [currentNumber, setCurrentNumber] = useState("");
  const [directWait, setDirectWait] = useState("");
  const [currentMode, setCurrentMode] = useState<InputMode>("auto");
  const [waitMode, setWaitMode] = useState<InputMode>("auto");
  const [initialGap, setInitialGap] = useState<number | null>(null);
  const [photoName, setPhotoName] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [ocrMessage, setOcrMessage] = useState(
    "端末内で受付番号を読み取ります。",
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const updateTicket = (value: string) => {
    const nextTicket = digitsOnly(value);
    setTicketNumber(nextTicket);

    const ticket = toNumber(nextTicket);
    if (ticket === null) {
      setInitialGap(null);
      if (currentMode === "auto") setCurrentNumber("");
      if (waitMode === "auto") setDirectWait("");
      return;
    }

    const gap = initialGap !== null && ticketNumber === nextTicket ? initialGap : pickInitialGap();
    if (gap !== initialGap) setInitialGap(gap);
    const defaultCurrent = estimateCurrent(ticket, gap);
    const effectiveCurrent = currentMode === "auto" ? defaultCurrent : toNumber(currentNumber);

    if (currentMode === "auto") setCurrentNumber(String(defaultCurrent));
    if (waitMode === "auto") {
      setDirectWait(
        currentMode === "auto"
          ? String(gap * MINUTES_PER_NUMBER)
          : effectiveCurrent === null
            ? ""
            : String(estimateWait(ticket, effectiveCurrent)),
      );
    }
  };

  const updateCurrent = (value: string) => {
    const nextCurrent = digitsOnly(value);
    setCurrentNumber(nextCurrent);
    setCurrentMode("manual");

    if (waitMode === "auto") {
      const ticket = toNumber(ticketNumber);
      const current = toNumber(nextCurrent);
      setDirectWait(
        ticket !== null && current !== null ? String(estimateWait(ticket, current)) : "",
      );
    }
  };

  const updateWait = (value: string) => {
    setDirectWait(digitsOnly(value));
    setWaitMode("manual");
  };

  const runLocalOcr = async (file: File) => {
    setOcrMessage("画像を読み取り中…");

    let nativeCandidate: string | null = null;
    try {
      if (window.TextDetector && typeof createImageBitmap !== "undefined") {
        const bitmap = await createImageBitmap(file);
        try {
          const detector = new window.TextDetector();
          const results = await detector.detect(bitmap);
          nativeCandidate = extractCandidate(results.map((item) => item.rawValue ?? "").join(" "));
        } finally {
          bitmap.close();
        }
      }
      if (nativeCandidate && nativeCandidate.length > 1) {
        updateTicket(nativeCandidate);
        setOcrMessage(`受付番号候補「${nativeCandidate}」を入力しました。確認してください。`);
        return;
      }
    } catch {
      // TextDetectorの失敗時は、同梱の端末内OCRへ進みます。
    }

    try {
      const candidate = await runBundledOcr(file);
      if (candidate) {
        updateTicket(candidate);
        setOcrMessage(`受付番号候補「${candidate}」を入力しました。確認してください。`);
        return;
      }
    } catch {
      // 同梱OCRが使えない場合も、手入力へ案内します。
    }

    if (nativeCandidate) {
      updateTicket(nativeCandidate);
      setOcrMessage(`受付番号候補「${nativeCandidate}」を入力しました。確認してください。`);
      return;
    }

    setOcrMessage("読み取れませんでした。受付番号を手入力してください");
  };

  const handlePhoto = (file: File | undefined) => {
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPhotoName(file.name);
    setPreviewUrl(URL.createObjectURL(file));
    void runLocalOcr(file);
  };

  const result = useMemo(() => {
    const ticket = toNumber(ticketNumber);
    const current = toNumber(currentNumber);
    const wait = toNumber(directWait);
    const groups = ticket !== null && current !== null ? Math.max(0, ticket - current) : null;
    const isCalled = groups === 0 && ticket !== null && current !== null;
    const hasTicket = ticket !== null;
    const hasStaffInput = current !== null || wait !== null;
    const canEstimate = hasTicket && hasStaffInput;

    let guidance = "入力が必要です。";
    if (!hasTicket) {
      guidance = "受付番号を入力してください。";
    } else if (!hasStaffInput) {
      guidance = "現在呼出番号または待ち時間を入力してください。";
    } else if (current !== null && wait === null) {
      guidance = "待ち時間を入力してください。";
    } else if (isCalled) {
      guidance = "呼出番号に達しています。";
    } else if (canEstimate) {
      guidance = waitMode === "auto" ? "目安" : "スタッフ入力";
    }

    return {
      ticket,
      current,
      wait,
      groups,
      isCalled,
      hasTicket,
      hasStaffInput,
      canEstimate,
      guidance,
    };
  }, [currentNumber, directWait, ticketNumber, waitMode]);

  const clearAll = () => {
    setTicketNumber("");
    setCurrentNumber("");
    setDirectWait("");
    setCurrentMode("auto");
    setWaitMode("auto");
    setInitialGap(null);
    setPhotoName("");
    setOcrMessage("端末内で受付番号を読み取ります。");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  return (
    <main className="page-shell">
      <header className="site-header">
        <div className="header-title-block">
          <span className="header-ticket" aria-hidden="true">待</span>
          <h1 id="page-title" className="page-title">くら寿司 待ち時間</h1>
        </div>
        <div className="privacy-chip">端末内のみ</div>
      </header>

      <section className="result-shell">
        <section className={`status-card ${result.canEstimate ? "is-ready" : "is-pending"}`} aria-live="polite" aria-label="待ち時間の結果">
          <div className="status-card-topline">
            <span>待ち時間の目安</span>
            <span className="status-dot" aria-hidden="true" />
          </div>
          <div className="ticket-display">
            <p>受付番号</p>
            <strong>{result.ticket ?? "—"}</strong>
          </div>
          <div className="status-divider" />
          <div className="result-grid">
            <div>
              <p>あと何組</p>
              <strong>{result.groups !== null ? `${result.groups}` : "—"}<small>{result.groups !== null ? "組" : ""}</small></strong>
            </div>
            <div>
              <p>待ち時間</p>
              <strong>{result.wait !== null ? formatWait(result.wait) : "—"}</strong>
            </div>
          </div>
          <p className={`status-guidance ${result.isCalled ? "is-called" : ""}`}>{result.guidance}</p>
        </section>
      </section>

      <section className="workspace" aria-label="入力">
        <div className="panel panel-photo">
          <div className="panel-heading">
            <h2>受付番号</h2>
          </div>
          <label className="photo-drop" htmlFor="ticket-photo">
            <span className="photo-icon" aria-hidden="true">＋</span>
            <span><strong>待ち札の写真を選ぶ／撮影</strong><small>端末内OCR。失敗時は手入力</small></span>
            <input id="ticket-photo" type="file" accept="image/*" capture="environment" onChange={(event) => handlePhoto(event.target.files?.[0])} />
          </label>
          {previewUrl ? (
            <div className="photo-preview">
              <img src={previewUrl} alt="選択した待ち札のプレビュー" />
              <span>{photoName}</span>
            </div>
          ) : null}
          <p className="ocr-message" role="status"><span className="message-dot" aria-hidden="true" />{ocrMessage}</p>
          <label className="field-label" htmlFor="ticket-number">受付番号</label>
          <input className="number-input" id="ticket-number" inputMode="numeric" pattern="[0-9]*" maxLength={4} value={ticketNumber} onChange={(event) => updateTicket(event.target.value)} placeholder="例 128" aria-describedby="ticket-help" />
          <p className="field-help" id="ticket-help">数字のみ</p>
        </div>

        <div className="panel panel-staff">
          <div className="panel-heading">
            <h2>スタッフ入力</h2>
          </div>
          <p className="panel-note">現在呼出番号・待ち時間</p>
          <div className="staff-fields">
            <label className="field-label" htmlFor="current-number">
              現在呼出番号
              <em className={`field-state ${currentMode === "auto" ? "is-auto" : "is-manual"}`}>
                {currentMode === "auto" ? "自動の初期値" : "スタッフ上書き"}
              </em>
            </label>
            <input className="number-input" id="current-number" inputMode="numeric" pattern="[0-9]*" maxLength={4} value={currentNumber} onChange={(event) => updateCurrent(event.target.value)} placeholder="例 115" />
            <label className="field-label" htmlFor="direct-wait">
              待ち時間（分）
              <em className={`field-state ${waitMode === "auto" ? "is-auto" : "is-manual"}`}>
                {waitMode === "auto" ? "初期の目安" : "スタッフ上書き"}
              </em>
            </label>
            <div className="unit-input"><input className="number-input" id="direct-wait" inputMode="numeric" pattern="[0-9]*" maxLength={3} value={directWait} onChange={(event) => updateWait(event.target.value)} placeholder="例 40" /><span>分</span></div>
          </div>
        </div>
      </section>

      <footer className="site-footer">
        <p className="footer-disclaimer" role="note">スタッフ作成／くら寿司公式ではありません／公式予約・順番待ちには接続しません。</p>
        <button type="button" className="clear-button" onClick={clearAll}>入力を消す</button>
      </footer>
    </main>
  );
}
