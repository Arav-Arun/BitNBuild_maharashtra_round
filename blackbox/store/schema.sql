PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    parent_run_id TEXT REFERENCES runs(run_id),
    fork_id TEXT,
    agent TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT 'dev',
    task_id TEXT NOT NULL,
    model TEXT,
    seed INTEGER,
    mode TEXT NOT NULL,
    outcome TEXT,
    score REAL,
    checker_reason TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    addr TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    agent_role TEXT,
    request_key TEXT,
    input_hash TEXT,
    output_hash TEXT,
    reasoning_hash TEXT,
    state_before TEXT NOT NULL,
    state_after TEXT,
    reads_json TEXT NOT NULL DEFAULT '[]',
    writes_json TEXT NOT NULL DEFAULT '[]',
    cache_status TEXT NOT NULL DEFAULT 'live',
    tokens_in INTEGER,
    tokens_out INTEGER,
    tokens_cached INTEGER,
    latency_ms REAL NOT NULL DEFAULT 0,
    finish_reason TEXT,
    error_type TEXT,
    retries INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, addr)
);

CREATE INDEX IF NOT EXISTS idx_steps_run_seq ON steps(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_steps_request_key ON steps(request_key);

CREATE TABLE IF NOT EXISTS edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    src_addr TEXT NOT NULL,
    src_pointer TEXT,
    dst_addr TEXT NOT NULL,
    dst_pointer TEXT,
    value_hash TEXT,
    key TEXT,
    version INTEGER,
    kind TEXT NOT NULL CHECK(kind IN ('state', 'message', 'inferred')),
    UNIQUE(run_id, src_addr, src_pointer, dst_addr, dst_pointer, key, version, kind)
);

CREATE INDEX IF NOT EXISTS idx_edges_run_src ON edges(run_id, src_addr);
CREATE INDEX IF NOT EXISTS idx_edges_run_dst ON edges(run_id, dst_addr);

CREATE TABLE IF NOT EXISTS cassette (
    request_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forks (
    fork_id TEXT PRIMARY KEY,
    base_run_id TEXT NOT NULL REFERENCES runs(run_id),
    branch_name TEXT NOT NULL,
    edits_json TEXT NOT NULL,
    mode TEXT NOT NULL,
    samples INTEGER NOT NULL,
    reexec_steps INTEGER NOT NULL DEFAULT 0,
    cached_steps INTEGER NOT NULL DEFAULT 0,
    invalidated_steps INTEGER NOT NULL DEFAULT 0,
    tokens_saved INTEGER NOT NULL DEFAULT 0,
    ms_saved REAL NOT NULL DEFAULT 0,
    fix_pass_rate REAL,
    fix_ci_low REAL,
    fix_ci_high REAL,
    control_pass_rate REAL,
    control_ci_low REAL,
    control_ci_high REAL,
    verdict TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    root_addr TEXT NOT NULL,
    fault_type TEXT,
    source TEXT NOT NULL CHECK(source IN ('injected', 'natural_auto', 'human', 'verified')),
    recovered INTEGER NOT NULL DEFAULT 0,
    manifest_addr TEXT,
    verified INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS diagnoses (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    ranking_json TEXT NOT NULL,
    conformal_set_json TEXT,
    abstain INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, model_version)
);

CREATE TABLE IF NOT EXISTS regression_exports (
    export_id TEXT PRIMARY KEY,
    fork_id TEXT NOT NULL REFERENCES forks(fork_id),
    test_hash TEXT NOT NULL,
    fixture_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
