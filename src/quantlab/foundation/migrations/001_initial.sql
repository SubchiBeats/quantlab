-- QuantLab schema v1 (Phase 1: research vault, data registry, audit, journal).
-- Append-only tables get BEFORE UPDATE / BEFORE DELETE triggers that abort.
-- "The platform must never forget failed experiments" is enforced here.

-- ============================== data domain ==============================

CREATE TABLE instruments (
    instrument_id  INTEGER PRIMARY KEY,
    symbol         TEXT NOT NULL UNIQUE,
    name           TEXT,
    asset_class    TEXT NOT NULL DEFAULT 'equity' CHECK (asset_class = 'equity'),
    currency       TEXT NOT NULL DEFAULT 'USD',
    source         TEXT NOT NULL,
    first_bar      TEXT,
    last_bar       TEXT,
    is_benchmark   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

CREATE TABLE data_quality_log (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT NOT NULL,
    range_start TEXT,
    range_end   TEXT,
    check_name  TEXT NOT NULL,
    severity    TEXT NOT NULL CHECK (severity IN ('info','warn','fail')),
    detail      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TRIGGER dql_no_update BEFORE UPDATE ON data_quality_log
BEGIN SELECT RAISE(ABORT, 'data_quality_log is append-only'); END;
CREATE TRIGGER dql_no_delete BEFORE DELETE ON data_quality_log
BEGIN SELECT RAISE(ABORT, 'data_quality_log is append-only'); END;

CREATE TABLE snapshots (
    snapshot_id    TEXT PRIMARY KEY,           -- sha256 of the manifest
    created_at     TEXT NOT NULL,
    description    TEXT NOT NULL,
    frequency      TEXT NOT NULL,
    manifest_json  TEXT NOT NULL,              -- files + per-file sha256
    coverage_start TEXT,
    coverage_end   TEXT
);
CREATE TRIGGER snapshots_no_update BEFORE UPDATE ON snapshots
BEGIN SELECT RAISE(ABORT, 'snapshots are immutable'); END;
CREATE TRIGGER snapshots_no_delete BEFORE DELETE ON snapshots
BEGIN SELECT RAISE(ABORT, 'snapshots are immutable'); END;

-- ============================ governance domain ==========================

CREATE TABLE journal_entries (
    journal_id  INTEGER PRIMARY KEY,
    ts          TEXT NOT NULL,
    author      TEXT NOT NULL,
    body        TEXT NOT NULL,
    tags        TEXT,
    linked_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TRIGGER journal_no_update BEFORE UPDATE ON journal_entries
BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;
CREATE TRIGGER journal_no_delete BEFORE DELETE ON journal_entries
BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;

CREATE TABLE audit_log (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    actor        TEXT NOT NULL CHECK (actor IN ('human','system')),
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    this_hash    TEXT NOT NULL
);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;

-- ============================ research domain ============================

CREATE TABLE hypotheses (
    hypothesis_id          INTEGER PRIMARY KEY,
    family                 TEXT NOT NULL,      -- groups related tests for multiple-testing accounting
    statement              TEXT NOT NULL,
    mechanism              TEXT NOT NULL,      -- why should this edge exist?
    success_criteria       TEXT NOT NULL,      -- declared BEFORE any run
    falsification_criteria TEXT NOT NULL,
    author                 TEXT NOT NULL DEFAULT 'human',
    created_at             TEXT NOT NULL
);
CREATE TRIGGER hypotheses_no_update BEFORE UPDATE ON hypotheses
BEGIN SELECT RAISE(ABORT, 'hypotheses are append-only'); END;
CREATE TRIGGER hypotheses_no_delete BEFORE DELETE ON hypotheses
BEGIN SELECT RAISE(ABORT, 'hypotheses are append-only'); END;

CREATE VIRTUAL TABLE hypotheses_fts USING fts5(
    statement, mechanism,
    content='hypotheses', content_rowid='hypothesis_id'
);
CREATE TRIGGER hypotheses_fts_sync AFTER INSERT ON hypotheses
BEGIN
    INSERT INTO hypotheses_fts(rowid, statement, mechanism)
    VALUES (new.hypothesis_id, new.statement, new.mechanism);
END;

CREATE TABLE experiments (
    experiment_id         INTEGER PRIMARY KEY,
    hypothesis_id         INTEGER NOT NULL REFERENCES hypotheses(hypothesis_id),
    strategy_name         TEXT NOT NULL,
    strategy_version      INTEGER NOT NULL,
    snapshot_id           TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    benchmark_symbol      TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'registered'
                          CHECK (status IN ('registered','running','complete')),
    verdict               TEXT CHECK (verdict IN ('pass','fail','inconclusive')),
    verdict_reason        TEXT,
    similar_failures_json TEXT NOT NULL DEFAULT '[]',
    ack_journal_id        INTEGER REFERENCES journal_entries(journal_id),
    created_at            TEXT NOT NULL,
    completed_at          TEXT
);
-- Identity columns are frozen; only status/verdict fields may change.
CREATE TRIGGER experiments_guard BEFORE UPDATE ON experiments
WHEN NEW.experiment_id         IS NOT OLD.experiment_id
  OR NEW.hypothesis_id         IS NOT OLD.hypothesis_id
  OR NEW.strategy_name         IS NOT OLD.strategy_name
  OR NEW.strategy_version      IS NOT OLD.strategy_version
  OR NEW.snapshot_id           IS NOT OLD.snapshot_id
  OR NEW.benchmark_symbol      IS NOT OLD.benchmark_symbol
  OR NEW.similar_failures_json IS NOT OLD.similar_failures_json
  OR NEW.ack_journal_id        IS NOT OLD.ack_journal_id
  OR NEW.created_at            IS NOT OLD.created_at
BEGIN SELECT RAISE(ABORT, 'experiment identity columns are immutable'); END;
-- A recorded verdict is final: complete experiments cannot be re-opened or re-judged.
CREATE TRIGGER experiments_verdict_final BEFORE UPDATE ON experiments
WHEN OLD.status = 'complete'
BEGIN SELECT RAISE(ABORT, 'completed experiments are immutable - register a new one'); END;
CREATE TRIGGER experiments_no_delete BEFORE DELETE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are never deleted'); END;

CREATE TABLE runs (
    run_id         TEXT PRIMARY KEY,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(experiment_id),
    run_type       TEXT NOT NULL CHECK
                   (run_type IN ('backtest','walkforward','montecarlo','benchmark','stress')),
    config_hash    TEXT NOT NULL,
    git_sha        TEXT NOT NULL,
    dirty_tree     INTEGER NOT NULL,
    seed           INTEGER NOT NULL,
    engine_version TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT NOT NULL,
    artifact_dir   TEXT,
    promotable     INTEGER NOT NULL            -- 0 if dirty tree etc.; advisory for Phase 2 gates
);
CREATE TRIGGER runs_no_update BEFORE UPDATE ON runs
BEGIN SELECT RAISE(ABORT, 'runs are append-only'); END;
CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'runs are append-only'); END;

CREATE TABLE run_metrics (
    run_id  TEXT NOT NULL REFERENCES runs(run_id),
    metric  TEXT NOT NULL,
    period  TEXT NOT NULL,                     -- 'full' | 'oos' | 'is' | 'fold_<n>' | ...
    value   REAL,
    PRIMARY KEY (run_id, metric, period)
);
CREATE TRIGGER run_metrics_no_update BEFORE UPDATE ON run_metrics
BEGIN SELECT RAISE(ABORT, 'run_metrics is append-only'); END;
CREATE TRIGGER run_metrics_no_delete BEFORE DELETE ON run_metrics
BEGIN SELECT RAISE(ABORT, 'run_metrics is append-only'); END;

CREATE TABLE trades (
    trade_id     INTEGER PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    symbol       TEXT NOT NULL,
    entry_ts     TEXT NOT NULL,
    entry_px     REAL NOT NULL,
    exit_ts      TEXT NOT NULL,
    exit_px      REAL NOT NULL,
    qty          INTEGER NOT NULL CHECK (qty > 0),      -- long-only: positive share counts
    fees         REAL NOT NULL,
    slippage     REAL NOT NULL,
    pnl          REAL NOT NULL,
    mae          REAL NOT NULL,
    mfe          REAL NOT NULL,
    holding_bars INTEGER NOT NULL,
    exit_reason  TEXT NOT NULL CHECK (exit_reason IN ('signal','stop','end_of_data')),
    context_json TEXT NOT NULL DEFAULT '{}'             -- indicator values behind the entry signal
);
CREATE TRIGGER trades_no_update BEFORE UPDATE ON trades
BEGIN SELECT RAISE(ABORT, 'trades are append-only'); END;
CREATE TRIGGER trades_no_delete BEFORE DELETE ON trades
BEGIN SELECT RAISE(ABORT, 'trades are append-only'); END;

CREATE TABLE walkforward_folds (
    run_id             TEXT NOT NULL REFERENCES runs(run_id),
    fold_n             INTEGER NOT NULL,
    train_start        TEXT NOT NULL,
    train_end          TEXT NOT NULL,
    test_start         TEXT NOT NULL,
    test_end           TEXT NOT NULL,
    embargo_bars       INTEGER NOT NULL,
    chosen_params_json TEXT NOT NULL,
    train_sharpe       REAL,
    test_sharpe        REAL,
    test_return        REAL,
    test_trades        INTEGER,
    PRIMARY KEY (run_id, fold_n)
);
CREATE TRIGGER wf_no_update BEFORE UPDATE ON walkforward_folds
BEGIN SELECT RAISE(ABORT, 'walkforward_folds is append-only'); END;
CREATE TRIGGER wf_no_delete BEFORE DELETE ON walkforward_folds
BEGIN SELECT RAISE(ABORT, 'walkforward_folds is append-only'); END;

CREATE TABLE mc_simulations (
    id           INTEGER PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    method       TEXT NOT NULL CHECK (method IN
                 ('trade_bootstrap','block_bootstrap','cost_stress','delay_stress','param_perturbation')),
    n_paths      INTEGER NOT NULL,
    seed         INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TRIGGER mc_no_update BEFORE UPDATE ON mc_simulations
BEGIN SELECT RAISE(ABORT, 'mc_simulations is append-only'); END;
CREATE TRIGGER mc_no_delete BEFORE DELETE ON mc_simulations
BEGIN SELECT RAISE(ABORT, 'mc_simulations is append-only'); END;

CREATE INDEX idx_experiments_verdict ON experiments(verdict);
CREATE INDEX idx_runs_experiment ON runs(experiment_id);
CREATE INDEX idx_trades_run ON trades(run_id);
