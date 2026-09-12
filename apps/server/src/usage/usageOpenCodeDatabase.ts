// @effect-diagnostics nodeBuiltinImport:off
/**
 * OpenCode usage source.
 *
 * OpenCode keeps its usage in a SQLite database (`opencode.db` under its XDG
 * data directory) rather than the JSONL transcript directories every other
 * provider scans, so it cannot flow through the byte-offset resume machinery
 * in `usageTranscriptReader` — "resume from an offset" is meaningless for a
 * database. This module is the seam between the two source kinds: a pure
 * row-to-record mapper, and a stateful reader that opens the database
 * read-only and re-reads only what a high-water mark says it must.
 *
 * Rows in `message` are immutable once written, so a high-water mark on
 * `time_created` is safe: a row already read can never come back changed.
 * The mark plus the database file's `(size, mtime)` form the cheap
 * "nothing changed" gate that lets a warm scan skip reopening the database.
 *
 * Only the `message` table is ever queried, and only four columns of it.
 * OpenCode's `part` table (tool output) is the bulk of the file by far and
 * never carries usage; touching it would make every scan enormously slow.
 *
 * @module usageOpenCodeDatabase
 */
import * as NodeFSP from "node:fs/promises";
import type * as NodeFS from "node:fs";
import * as NodePath from "node:path";

import type { UsageTokenTotals } from "@t3tools/contracts";

import { totalTokens, type UsageRecord } from "./usageTranscripts.ts";
import { expandHomePath } from "../pathExpansion.ts";

/**
 * How far behind the high-water mark each re-read reaches.
 *
 * `time_created` comes from the writer's wall clock and several OpenCode
 * sessions can be in flight at once, so it is not monotonic across rows: a
 * message can commit with a timestamp slightly behind one already scanned,
 * and clock adjustment does the same. The mark therefore queries from
 * `mark - overlap` rather than `mark`. The overlap re-reads are free: every
 * record carries the message id as its dedupe key, and the aggregator
 * de-duplicates the whole scan globally.
 */
export const OPENCODE_HWM_OVERLAP_MS = 5 * 60 * 1000;

/** Longest a read waits out an active writer before degrading. */
const BUSY_TIMEOUT_MS = 3000;

const MESSAGE_ROWS_SQL =
  "SELECT id, session_id, time_created, data FROM message WHERE time_created > ?";

/**
 * Raw shape of one `message` row, narrowed to the four columns this source
 * reads. The fields are typed loosely so the mapper can be tested against
 * malformed values the database could never hand back.
 */
export interface OpenCodeMessageRow {
  readonly id: unknown;
  readonly session_id: unknown;
  readonly time_created: unknown;
  readonly data: unknown;
}

function tokenCount(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.trunc(value) : 0;
}

/**
 * Maps one `message` row onto a usage record, or `null` when the row carries
 * no reportable usage.
 *
 * OpenCode reports `output` and `reasoning` as separate, non-overlapping
 * counts — unlike Claude and Codex, whose reasoning is a subset of output —
 * so the two are combined into `outputTokens` here. That preserves the
 * `UsageTokenTotals` invariant that `reasoningTokens` is a subset of
 * `outputTokens`, which `totalTokens` relies on to count each generated
 * token exactly once.
 */
export function parseOpenCodeMessageRow(row: OpenCodeMessageRow): UsageRecord | null {
  // The message id is the record's dedupe key, and the overlap re-reads
  // depend on it collapsing: a row without one must not be emitted at all,
  // or every warm scan would double count it.
  if (typeof row.id !== "string" || row.id.length === 0) return null;

  if (typeof row.data !== "string") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(row.data);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const data = parsed as Record<string, unknown>;

  if (data["role"] !== "assistant") return null;

  // Prefer the high-resolution clock in the payload; the column is the row
  // insert time and the fallback of the two.
  let timestampMs: number | null = null;
  const time = data["time"];
  if (typeof time === "object" && time !== null) {
    const created = (time as Record<string, unknown>)["created"];
    if (typeof created === "number" && Number.isFinite(created) && created > 0) {
      timestampMs = created;
    }
  }
  if (
    timestampMs === null &&
    typeof row.time_created === "number" &&
    Number.isFinite(row.time_created) &&
    row.time_created > 0
  ) {
    timestampMs = row.time_created;
  }
  if (timestampMs === null) return null;

  const model = data["modelID"];
  // Kept raw: the model breakdown and rate lookup key on exactly this string.
  if (typeof model !== "string" || model.length === 0) return null;

  const tokens = data["tokens"];
  const tokensRecord =
    typeof tokens === "object" && tokens !== null ? (tokens as Record<string, unknown>) : {};
  const cache = tokensRecord["cache"];
  const cacheRecord =
    typeof cache === "object" && cache !== null ? (cache as Record<string, unknown>) : {};

  const output = tokenCount(tokensRecord["output"]);
  const reasoning = tokenCount(tokensRecord["reasoning"]);
  // OpenCode's `input` is already non-cached (cache reads/writes sit beside
  // it), so the three input counts are disjoint, matching UsageTokenTotals.
  const totals: UsageTokenTotals = {
    uncachedInputTokens: tokenCount(tokensRecord["input"]),
    cachedInputTokens: tokenCount(cacheRecord["read"]),
    cacheCreationTokens: tokenCount(cacheRecord["write"]),
    outputTokens: output + reasoning,
    reasoningTokens: reasoning,
  };

  const cost = data["cost"];
  const reportedCostUsd = typeof cost === "number" && Number.isFinite(cost) ? cost : null;

  // Rows with neither tokens nor cost are empty or cancelled turns, not usage.
  // A reported zero is no cost for this purpose — OpenCode writes `cost: 0` on
  // turns that produced nothing — but a zero cost on a row that does have
  // tokens is kept as reported rather than repriced from the rate table.
  if (totalTokens(totals) === 0 && (reportedCostUsd === null || reportedCostUsd === 0)) {
    return null;
  }

  return {
    provider: "opencode",
    timestampMs,
    model,
    sessionId: typeof row.session_id === "string" ? row.session_id : "",
    totals,
    reportedCostUsd,
    dedupeKey: row.id,
  };
}

/**
 * Resolves OpenCode's usage database the way OpenCode itself resolves its
 * data directory: `$XDG_DATA_HOME/opencode` when that variable is set, else
 * the platform default `~/.local/share/opencode`. T3 Code's OpenCode settings
 * expose no home override to honour.
 */
export function resolveOpenCodeDatabasePath(input: {
  readonly xdgDataHome: string | undefined;
  readonly homedir: string;
}): string {
  // Whitespace must fall through to the default: coalescing an empty string
  // as a path would resolve against the process cwd.
  const configured = input.xdgDataHome?.trim() ?? "";
  const dataHome =
    configured.length > 0
      ? NodePath.resolve(expandHomePath(configured))
      : NodePath.join(input.homedir, ".local", "share");
  return NodePath.join(dataHome, "opencode", "opencode.db");
}

/**
 * The reader's incremental state, held by the service alongside its file
 * scan cache.
 */
export interface OpenCodeScanState {
  /** Every record read so far, keyed by message id (the dedupe key). */
  readonly records: Map<string, UsageRecord>;
  /**
   * Highest `message.time_created` column value read so far. Rows are
   * immutable, so everything at or below the mark is already on hand.
   */
  highWaterMarkMs: number;
  /** `(size, mtime)` of the database file the records came from. */
  size: number;
  mtimeMs: number;
  /** Filesystem identity of the database file, as `device:inode`. */
  volumeId: string;
  /** False until a read succeeds, so the first scan cannot take the fast path. */
  hasRead: boolean;
}

export function createOpenCodeScanState(): OpenCodeScanState {
  return {
    records: new Map(),
    highWaterMarkMs: 0,
    size: -1,
    mtimeMs: -1,
    volumeId: "",
    hasRead: false,
  };
}

export type OpenCodeScanOutcome =
  | { readonly status: "ok"; readonly volumeId: string; readonly records: readonly UsageRecord[] }
  | { readonly status: "missing"; readonly volumeId: string }
  | { readonly status: "failed"; readonly volumeId: string; readonly detail: string };

export interface OpenCodeScanOptions {
  /**
   * Cached records older than this are dropped, mirroring the file scan
   * cache's retention: they sit behind the high-water mark and beyond the
   * longest window the UI offers, so they would only cost memory.
   */
  readonly retentionCutoffMs?: number;
}

/**
 * Reads OpenCode's database and returns the source's full record set.
 *
 * Incremental by high-water mark: a warm scan reopens the database only when
 * its `(size, mtime)` changed, and then queries only rows newer than
 * `mark - overlap`, merging them into the records already held. This never
 * rejects — a locked, busy, corrupt or missing database degrades to a
 * `missing`/`failed` outcome so the usage read keeps working for every other
 * provider, exactly as a missing transcript directory does.
 */
export async function scanOpenCodeDatabase(
  dbPath: string,
  state: OpenCodeScanState,
  options: OpenCodeScanOptions = {},
): Promise<OpenCodeScanOutcome> {
  let volumeId = "";
  try {
    let stats: NodeFS.Stats;
    try {
      stats = await NodeFSP.stat(dbPath);
    } catch {
      return { status: "missing", volumeId };
    }
    volumeId = `${stats.dev}:${stats.ino}`;

    // A replaced database file (a restore, a VACUUM rewrite) shares nothing
    // with the rows already held: forget the cache and the mark rather than
    // stitching two different databases together.
    if (state.hasRead && state.volumeId !== volumeId) {
      state.records.clear();
      state.highWaterMarkMs = 0;
      state.hasRead = false;
    }

    // Cheap "nothing changed" gate. OpenCode only ever adds rows, so an
    // unchanged `(size, mtime)` means the held records are still complete —
    // the same gate the file pipeline caches parsed files on.
    if (state.hasRead && state.size === stats.size && state.mtimeMs === stats.mtimeMs) {
      pruneOpenCodeRecords(state, options.retentionCutoffMs);
      return { status: "ok", volumeId, records: [...state.records.values()] };
    }

    const read = await readOpenCodeMessageRows(
      dbPath,
      state.highWaterMarkMs - OPENCODE_HWM_OVERLAP_MS,
    );
    for (const record of read.records) {
      // parseOpenCodeMessageRow only emits records that carry the message id
      // as their dedupe key, so this merge is where overlap re-reads collapse.
      if (record.dedupeKey !== null) state.records.set(record.dedupeKey, record);
    }
    // Never backwards: a scan that read nothing (or only overlap repeats)
    // must not undo the mark a previous scan advanced.
    state.highWaterMarkMs = Math.max(state.highWaterMarkMs, read.maxRowTimeCreatedMs);
    state.size = stats.size;
    state.mtimeMs = stats.mtimeMs;
    state.volumeId = volumeId;
    state.hasRead = true;
    pruneOpenCodeRecords(state, options.retentionCutoffMs);

    return { status: "ok", volumeId, records: [...state.records.values()] };
  } catch (cause) {
    return { status: "failed", volumeId, detail: boundedDetail(cause) };
  }
}

function pruneOpenCodeRecords(state: OpenCodeScanState, retentionCutoffMs: number | undefined) {
  if (retentionCutoffMs === undefined) return;
  for (const [key, record] of state.records) {
    if (record.timestampMs < retentionCutoffMs) state.records.delete(key);
  }
}

function boundedDetail(cause: unknown): string {
  const message = cause instanceof Error ? cause.message : String(cause);
  const trimmed = message.trim();
  return trimmed.length > 0 ? trimmed.slice(0, 200) : "OpenCode database could not be read.";
}

/**
 * The two SQLite bindings the server supports, behind one read-only face.
 * `persistence/Layers/Sqlite.ts` makes the same runtime split for T3 Code's
 * own database.
 */
interface ReadOnlyOpenCodeDatabase {
  setBusyTimeout(ms: number): void;
  queryMessageRows(sinceExclusiveMs: number): readonly unknown[];
  close(): void;
}

/**
 * Opens the database read-only. Read-only matters twice over: OpenCode may be
 * running and writing to this file through WAL while the scan reads it, and a
 * usage scan must never create, migrate or otherwise mutate another process's
 * database. Both bindings refuse to create the file in read-only mode.
 */
async function openOpenCodeDatabase(dbPath: string): Promise<ReadOnlyOpenCodeDatabase> {
  if (process.versions.bun !== undefined) {
    const { Database } = await import("bun:sqlite");
    const db = new Database(dbPath, { readonly: true, create: false });
    return {
      setBusyTimeout: (ms) => db.exec(`PRAGMA busy_timeout = ${ms};`),
      queryMessageRows: (sinceExclusiveMs) =>
        db.query(MESSAGE_ROWS_SQL).all(sinceExclusiveMs) as readonly unknown[],
      close: () => db.close(),
    };
  }
  const { DatabaseSync } = await import("node:sqlite");
  const db = new DatabaseSync(dbPath, { readOnly: true });
  return {
    setBusyTimeout: (ms) => db.exec(`PRAGMA busy_timeout = ${ms};`),
    queryMessageRows: (sinceExclusiveMs) =>
      db.prepare(MESSAGE_ROWS_SQL).all(sinceExclusiveMs) as readonly unknown[],
    close: () => db.close(),
  };
}

async function readOpenCodeMessageRows(
  dbPath: string,
  sinceExclusiveMs: number,
): Promise<{ readonly records: readonly UsageRecord[]; readonly maxRowTimeCreatedMs: number }> {
  const db = await openOpenCodeDatabase(dbPath);
  try {
    // Best effort: a busy timeout lets a scan ride out a brief writer lock
    // instead of degrading for that scan. The pragma writes nothing.
    try {
      db.setBusyTimeout(BUSY_TIMEOUT_MS);
    } catch {
      // Ignore; the read below still degrades cleanly on a real lock.
    }
    const rows = db.queryMessageRows(sinceExclusiveMs);

    const records: UsageRecord[] = [];
    let maxRowTimeCreatedMs = 0;
    for (const row of rows) {
      const values = row as Record<string, unknown>;
      // The mark advances from the raw column, not from parsed records: rows
      // skipped as non-usage still prove the scan reached their timestamp.
      const rowTimeCreated = values["time_created"];
      if (
        typeof rowTimeCreated === "number" &&
        Number.isFinite(rowTimeCreated) &&
        rowTimeCreated > maxRowTimeCreatedMs
      ) {
        maxRowTimeCreatedMs = rowTimeCreated;
      }
      const record = parseOpenCodeMessageRow({
        id: values["id"],
        session_id: values["session_id"],
        time_created: values["time_created"],
        data: values["data"],
      });
      if (record !== null) records.push(record);
    }
    return { records, maxRowTimeCreatedMs };
  } finally {
    try {
      db.close();
    } catch {
      // A failed close must not shadow the read's outcome.
    }
  }
}
