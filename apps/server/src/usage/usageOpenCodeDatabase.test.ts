// @effect-diagnostics nodeBuiltinImport:off - the suite builds real temporary
// SQLite databases on disk. It never touches the developer's own OpenCode
// database: every database is created inside a fresh temp directory.
import * as NodeFSP from "node:fs/promises";
import * as NodeOS from "node:os";
import * as NodePath from "node:path";
import * as NodeSqlite from "node:sqlite";

import { describe, expect, it } from "@effect/vitest";

import { totalTokens } from "./usageTranscripts.ts";
import {
  OPENCODE_HWM_OVERLAP_MS,
  createOpenCodeScanState,
  parseOpenCodeMessageRow,
  resolveOpenCodeDatabasePath,
  scanOpenCodeDatabase,
  type OpenCodeMessageRow,
} from "./usageOpenCodeDatabase.ts";

/** Shaped after a real OpenCode assistant `message` row's `data` payload. */
function assistantPayload(
  overrides: {
    role?: string;
    createdMs?: number;
    modelID?: string;
    cost?: unknown;
    tokens?: Record<string, unknown>;
  } = {},
): string {
  const payload: Record<string, unknown> = {
    role: "assistant",
    time: { created: 1_764_000_000_000, completed: 1_764_000_010_000 },
    modelID: "gpt-5.1-codex-max",
    providerID: "openai",
    mode: "build",
    path: { cwd: "/tmp/project", root: "/tmp/project" },
    cost: 0.005434,
    tokens: { input: 1032, output: 48, reasoning: 0, cache: { read: 29312, write: 0 } },
    finish: "tool-calls",
  };
  if (overrides.role !== undefined) payload["role"] = overrides.role;
  if (overrides.createdMs !== undefined) payload["time"] = { created: overrides.createdMs };
  if (overrides.modelID !== undefined) payload["modelID"] = overrides.modelID;
  if ("cost" in overrides) payload["cost"] = overrides.cost;
  if (overrides.tokens !== undefined) payload["tokens"] = overrides.tokens;
  return JSON.stringify(payload);
}

function row(overrides: Partial<OpenCodeMessageRow> = {}): OpenCodeMessageRow {
  return {
    id: "msg_1",
    session_id: "ses_1",
    time_created: 1_764_000_000_500,
    data: assistantPayload(),
    ...overrides,
  };
}

const MESSAGE_TABLE_SQL = `CREATE TABLE IF NOT EXISTS message (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  time_created INTEGER NOT NULL,
  time_updated INTEGER NOT NULL,
  data TEXT NOT NULL
)`;

async function createOpenCodeDatabase(dbPath: string): Promise<NodeSqlite.DatabaseSync> {
  await NodeFSP.mkdir(NodePath.dirname(dbPath), { recursive: true });
  const db = new NodeSqlite.DatabaseSync(dbPath);
  db.exec(MESSAGE_TABLE_SQL);
  return db;
}

function insertMessage(
  db: NodeSqlite.DatabaseSync,
  id: string,
  sessionId: string,
  timeCreated: number,
  data: string,
): void {
  db.prepare(
    "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
  ).run(id, sessionId, timeCreated, timeCreated, data);
}

async function withTempDir(
  run: (dir: string) => Promise<void>,
): Promise<void> {
  const dir = await NodeFSP.mkdtemp(NodePath.join(NodeOS.tmpdir(), "opencode-usage-test-"));
  try {
    await run(dir);
  } finally {
    await NodeFSP.rm(dir, { recursive: true, force: true });
  }
}

describe("parseOpenCodeMessageRow", () => {
  it("maps a full assistant row onto a usage record", () => {
    const record = parseOpenCodeMessageRow(
      row({ time_created: 1_764_000_000_500, data: assistantPayload() }),
    );
    expect(record).not.toBeNull();
    expect(record?.provider).toBe("opencode");
    expect(record?.timestampMs).toBe(1_764_000_000_000);
    expect(record?.model).toBe("gpt-5.1-codex-max");
    expect(record?.sessionId).toBe("ses_1");
    expect(record?.totals).toEqual({
      uncachedInputTokens: 1032,
      cachedInputTokens: 29312,
      cacheCreationTokens: 0,
      outputTokens: 48,
      reasoningTokens: 0,
    });
    expect(record?.reportedCostUsd).toBe(0.005434);
    expect(record?.dedupeKey).toBe("msg_1");
  });

  it("combines reasoning into output to preserve the subset invariant", () => {
    // OpenCode reports output and reasoning as separate, non-overlapping
    // counts, so a row can carry more reasoning than output.
    const record = parseOpenCodeMessageRow(
      row({
        data: assistantPayload({
          tokens: { input: 0, output: 100, reasoning: 250, cache: { read: 0, write: 0 } },
        }),
      }),
    );
    expect(record?.totals.outputTokens).toBe(350);
    expect(record?.totals.reasoningTokens).toBe(250);
    // Counted once: 350, not 100 (reasoning dropped) and not 600 (added on top).
    expect(totalTokens(record!.totals)).toBe(350);
  });

  it("falls back to the row's time_created when the payload has no usable time", () => {
    const payload = JSON.stringify({
      role: "assistant",
      modelID: "gpt-5.1-codex-max",
      cost: 0.01,
      tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
    });
    const record = parseOpenCodeMessageRow(
      row({ time_created: 1_764_000_000_500, data: payload }),
    );
    expect(record?.timestampMs).toBe(1_764_000_000_500);

    // A zero or non-finite payload time is no better than the column.
    const zeroed = parseOpenCodeMessageRow(
      row({ time_created: 1_764_000_000_500, data: assistantPayload({ createdMs: 0 }) }),
    );
    expect(zeroed?.timestampMs).toBe(1_764_000_000_500);

    const unstampeded = parseOpenCodeMessageRow(
      row({ time_created: "not-a-number", data: payload }),
    );
    expect(unstampeded).toBeNull();
  });

  it("skips rows whose role is not assistant", () => {
    expect(parseOpenCodeMessageRow(row({ data: assistantPayload({ role: "user" }) }))).toBeNull();
  });

  it("skips rows with no usable tokens and no cost", () => {
    const emptyTokens = assistantPayload({ tokens: {}, cost: 0 });
    expect(parseOpenCodeMessageRow(row({ data: emptyTokens }))).toBeNull();

    const noTokens = JSON.stringify({ role: "assistant", modelID: "m", time: { created: 1 } });
    expect(parseOpenCodeMessageRow(row({ data: noTokens }))).toBeNull();
  });

  it("keeps a zero-token row that reports a cost", () => {
    const record = parseOpenCodeMessageRow(
      row({ data: assistantPayload({ tokens: {}, cost: 0.005 }) }),
    );
    expect(record?.reportedCostUsd).toBe(0.005);
    expect(record?.totals).toEqual({
      uncachedInputTokens: 0,
      cachedInputTokens: 0,
      cacheCreationTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
    });
  });

  it("tolerates partial token shapes", () => {
    const record = parseOpenCodeMessageRow(
      row({ data: assistantPayload({ tokens: { output: 48 } }) }),
    );
    expect(record?.totals).toEqual({
      uncachedInputTokens: 0,
      cachedInputTokens: 0,
      cacheCreationTokens: 0,
      outputTokens: 48,
      reasoningTokens: 0,
    });

    const hostileCache = parseOpenCodeMessageRow(
      row({
        data: assistantPayload({
          tokens: { input: 5, output: 1, reasoning: 0, cache: "nope" },
        }),
      }),
    );
    expect(hostileCache?.totals.cachedInputTokens).toBe(0);
    expect(hostileCache?.totals.cacheCreationTokens).toBe(0);
  });

  it("rejects malformed payloads", () => {
    expect(parseOpenCodeMessageRow(row({ data: "not json{" }))).toBeNull();
    expect(parseOpenCodeMessageRow(row({ data: 42 }))).toBeNull();
    expect(parseOpenCodeMessageRow(row({ data: '"just a string"' }))).toBeNull();
    expect(parseOpenCodeMessageRow(row({ data: assistantPayload({ modelID: "" }) }))).toBeNull();
  });

  it("requires a message id so overlap re-reads can collapse", () => {
    expect(parseOpenCodeMessageRow(row({ id: "" }))).toBeNull();
    expect(parseOpenCodeMessageRow(row({ id: 42 }))).toBeNull();
  });

  it("treats a non-finite cost as no cost", () => {
    const record = parseOpenCodeMessageRow(
      row({ data: assistantPayload({ cost: "0.5" }) }),
    );
    expect(record?.reportedCostUsd).toBeNull();

    const unpriced = parseOpenCodeMessageRow(
      row({ data: assistantPayload({ cost: Number.NaN }) }),
    );
    expect(unpriced?.reportedCostUsd).toBeNull();
  });

  it("clamps negative and fractional token counts", () => {
    const record = parseOpenCodeMessageRow(
      row({
        data: assistantPayload({
          tokens: { input: -5, output: 10.9, reasoning: 3, cache: { read: -2, write: 1.5 } },
        }),
      }),
    );
    expect(record?.totals).toEqual({
      uncachedInputTokens: 0,
      cachedInputTokens: 0,
      cacheCreationTokens: 1,
      outputTokens: 13,
      reasoningTokens: 3,
    });
  });
});

describe("resolveOpenCodeDatabasePath", () => {
  it("prefers $XDG_DATA_HOME/opencode", () => {
    expect(
      resolveOpenCodeDatabasePath({ xdgDataHome: "/custom/data", homedir: "/home/u" }),
    ).toBe(NodePath.join("/custom/data", "opencode", "opencode.db"));
  });

  it("falls back to ~/.local/share/opencode", () => {
    const expected = NodePath.join("/home/u", ".local", "share", "opencode", "opencode.db");
    expect(resolveOpenCodeDatabasePath({ xdgDataHome: undefined, homedir: "/home/u" })).toBe(
      expected,
    );
    expect(resolveOpenCodeDatabasePath({ xdgDataHome: "", homedir: "/home/u" })).toBe(expected);
    expect(resolveOpenCodeDatabasePath({ xdgDataHome: "   ", homedir: "/home/u" })).toBe(expected);
  });

  it("expands a leading ~ in $XDG_DATA_HOME", () => {
    // expandHomePath expands against the process home, so the test's homedir
    // override must agree with it for this case.
    expect(
      resolveOpenCodeDatabasePath({ xdgDataHome: "~/xdg", homedir: NodeOS.homedir() }),
    ).toBe(NodePath.join(NodeOS.homedir(), "xdg", "opencode", "opencode.db"));
  });
});

describe("scanOpenCodeDatabase", () => {
  it("reads assistant rows and advances the high-water mark from the raw column", async () => {
    await withTempDir(async (dir) => {
      const dbPath = NodePath.join(dir, "opencode.db");
      const db = await createOpenCodeDatabase(dbPath);
      insertMessage(db, "msg_1", "ses_1", 1000, assistantPayload({ createdMs: 900 }));
      insertMessage(db, "msg_2", "ses_2", 2000, assistantPayload({ createdMs: 2100 }));
      insertMessage(db, "msg_3", "ses_3", 3000, assistantPayload({ role: "user" }));
      insertMessage(
        db,
        "msg_4",
        "ses_4",
        4000,
        assistantPayload({ tokens: {}, cost: 0 }),
      );
      await db.close();

      const state = createOpenCodeScanState();
      const outcome = await scanOpenCodeDatabase(dbPath, state);

      expect(outcome.status).toBe("ok");
      if (outcome.status !== "ok") return;
      // The user row and the zero-usage row are not records...
      expect(outcome.records).toHaveLength(2);
      expect(outcome.volumeId).not.toBe("");
      // ...but they still prove the scan reached their timestamps.
      expect(state.highWaterMarkMs).toBe(4000);
      expect(state.hasRead).toBe(true);
      expect(state.records.get("msg_1")?.timestampMs).toBe(900);
      expect(state.records.get("msg_2")?.timestampMs).toBe(2100);
    });
  });

  it("re-reads the overlap window on later scans and collapses the repeats", async () => {
    await withTempDir(async (dir) => {
      const dbPath = NodePath.join(dir, "opencode.db");
      const db = await createOpenCodeDatabase(dbPath);
      insertMessage(db, "msg_1", "ses_1", 1000, assistantPayload());
      insertMessage(db, "msg_2", "ses_2", 2000, assistantPayload());
      await db.close();

      const state = createOpenCodeScanState();
      const first = await scanOpenCodeDatabase(dbPath, state);
      expect(first.status === "ok" ? first.records : []).toHaveLength(2);

      // A concurrent session commits a message whose timestamp lands slightly
      // behind the mark, the shape the overlap exists to catch.
      const db2 = new NodeSqlite.DatabaseSync(dbPath);
      insertMessage(db2, "msg_3", "ses_3", 2000 - OPENCODE_HWM_OVERLAP_MS / 2, assistantPayload());
      await db2.close();

      const second = await scanOpenCodeDatabase(dbPath, state);
      expect(second.status).toBe("ok");
      if (second.status !== "ok") return;
      expect(second.records).toHaveLength(3);
      expect(state.highWaterMarkMs).toBe(2000);
      // The overlap re-read msg_1/msg_2 by timestamp; the ids collapse them.
      expect(new Set(second.records.map((record) => record.dedupeKey)).size).toBe(3);
    });
  });

  it("does not reopen an unchanged database", async () => {
    await withTempDir(async (dir) => {
      const dbPath = NodePath.join(dir, "opencode.db");
      const db = await createOpenCodeDatabase(dbPath);
      insertMessage(db, "msg_1", "ses_1", 1000, assistantPayload());
      await db.close();

      // Pin a whole-millisecond mtime so the gate comparison is exact.
      const pinnedMs = 1_764_000_000_000;
      await NodeFSP.utimes(dbPath, pinnedMs, pinnedMs);

      const state = createOpenCodeScanState();
      const first = await scanOpenCodeDatabase(dbPath, state);
      expect(first.status === "ok" ? first.records : []).toHaveLength(1);

      // Corrupt the file beyond any recognition without changing its size or
      // mtime: if the gate missed, this scan would fail to read it.
      const { size } = await NodeFSP.stat(dbPath);
      await NodeFSP.writeFile(dbPath, Buffer.alloc(size, 0x21));
      await NodeFSP.utimes(dbPath, pinnedMs, pinnedMs);

      const second = await scanOpenCodeDatabase(dbPath, state);
      expect(second.status).toBe("ok");
      if (second.status !== "ok") return;
      expect(second.records).toHaveLength(1);
      expect(second.records[0]?.dedupeKey).toBe("msg_1");
    });
  });

  it("forgets its state when the database file is replaced", async () => {
    await withTempDir(async (dir) => {
      const dbPath = NodePath.join(dir, "opencode.db");
      const db = await createOpenCodeDatabase(dbPath);
      insertMessage(db, "msg_1", "ses_1", 1000, assistantPayload());
      await db.close();

      const state = createOpenCodeScanState();
      const first = await scanOpenCodeDatabase(dbPath, state);
      expect(first.status === "ok" ? first.records : []).toHaveLength(1);

      // A different database moved over the path keeps its own inode, so the
      // volume identity changes and the state must reset.
      const replacementPath = NodePath.join(dir, "replacement.db");
      const replacement = await createOpenCodeDatabase(replacementPath);
      insertMessage(replacement, "msg_9", "ses_9", 500, assistantPayload());
      await replacement.close();
      await NodeFSP.rename(replacementPath, dbPath);

      const second = await scanOpenCodeDatabase(dbPath, state);
      expect(second.status).toBe("ok");
      if (second.status !== "ok") return;
      expect(second.records).toHaveLength(1);
      expect(second.records[0]?.dedupeKey).toBe("msg_9");
      expect(state.highWaterMarkMs).toBe(500);
    });
  });

  it("degrades a file that is not a database to a failed outcome", async () => {
    await withTempDir(async (dir) => {
      const dbPath = NodePath.join(dir, "opencode.db");
      await NodeFSP.writeFile(dbPath, "this is not a database, it is just text");

      const state = createOpenCodeScanState();
      const first = await scanOpenCodeDatabase(dbPath, state);
      expect(first.status).toBe("failed");
      if (first.status !== "failed") return;
      expect(first.detail.length).toBeGreaterThan(0);
      expect(state.hasRead).toBe(false);
      expect(state.records.size).toBe(0);

      // And a retry neither throws nor poisons the state.
      const second = await scanOpenCodeDatabase(dbPath, state);
      expect(second.status).toBe("failed");
      expect(state.highWaterMarkMs).toBe(0);
    });
  });

  it("reports missing when the database does not exist", async () => {
    await withTempDir(async (dir) => {
      const state = createOpenCodeScanState();
      const outcome = await scanOpenCodeDatabase(NodePath.join(dir, "absent.db"), state);
      expect(outcome.status).toBe("missing");
      expect(state.hasRead).toBe(false);
    });
  });

  it("drops cached records older than the retention cutoff", async () => {
    await withTempDir(async (dir) => {
      const dbPath = NodePath.join(dir, "opencode.db");
      const db = await createOpenCodeDatabase(dbPath);
      // The record's timestamp comes from the payload clock, so pin it low
      // enough to sit under the cutoff.
      insertMessage(db, "msg_1", "ses_1", 1000, assistantPayload({ createdMs: 1000 }));
      await db.close();

      const state = createOpenCodeScanState();
      const pruned = await scanOpenCodeDatabase(dbPath, state, { retentionCutoffMs: 5000 });
      expect(pruned.status).toBe("ok");
      expect(pruned.status === "ok" ? pruned.records : []).toHaveLength(0);
    });
  });
});
