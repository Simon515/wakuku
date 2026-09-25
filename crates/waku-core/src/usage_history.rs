//! Historical token and cost usage for the settings Usage page, scanned from
//! the provider CLIs' own on-disk session transcripts the way T3 Code and
//! `ccusage` do it, so usage covers turns driven outside Waku too: Claude
//! Code's `~/.claude/projects`, Codex's `~/.codex/sessions`, the Pi agent
//! session format shared by Pi's `~/.pi/agent/sessions` and Oh My Pi's
//! `~/.omp/agent/sessions`, Kimi Code's `~/.kimi-code/sessions` (plus the
//! pre-migration `~/.kimi/sessions`, which uses an older record shape), and
//! DeepSeek Harness's zstd-compressed `~/.dsh/sessions`. Costs are priced against
//! LiteLLM's model rate table, fetched at most daily and cached beside the
//! app database.
//!
//! Everything here blocks on the filesystem and (for the rate table) the
//! network, and must run on the background executor. Render reads only the
//! [`UsageHistory`] snapshot the app entity stores. Transcripts are
//! append-only, so parsed records are memoised per file by `(size, mtime)` in
//! a [`ScanCache`] the caller owns; warm rescans only reparse changed files.

use std::collections::{HashMap, HashSet};
use std::io::BufRead as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{Datelike as _, Local, NaiveDate, TimeZone as _, Utc};
use md5::Digest as _;
use serde_json::Value;

use crate::model::ProviderKind;

pub use waku_protocol::usage_history::{
    CostQuality, DaySlice, MONTHLY_WINDOW, ModelSlice, MonthSlice, PricingStatus, ProjectSlice,
    ProviderDay, ProviderSlice, TokenTotals, UsageHistory, UsageProvider, UsageWindow,
    WINDOW_CHOICES,
};

/// Files whose mtime predates the window start by more than this are skipped
/// without opening. The slack covers a session whose last write lands just
/// before local midnight on the window's first day.
const MTIME_SLACK: Duration = Duration::from_secs(36 * 3600);

const LITELLM_RATES_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
const RATES_CACHE_FILE: &str = "usage-model-rates.json";
/// Rates move rarely; a day-old table keeps the page working offline.
const RATES_TTL: Duration = Duration::from_secs(24 * 3600);

/// One usage event parsed out of a transcript line.
#[derive(Clone, Debug)]
pub struct UsageRecord {
    pub provider: UsageProvider,
    pub timestamp_ms: i64,
    pub model: String,
    pub session_id: String,
    /// The working directory the session ran in, as recorded by the CLI —
    /// the per-project view's grouping key. Empty when the transcript does
    /// not say.
    pub project: String,
    pub totals: TokenTotals,
    pub reported_cost_usd: Option<f64>,
    /// Key for cross-file de-duplication, or `None` when the record is
    /// inherently unique and needs no dedup.
    pub dedupe_key: Option<String>,
}

/// Cheap substring gate applied before JSON parsing. Transcripts are mostly
/// tool output; only a minority of lines carry usage, and skipping the rest
/// before `serde_json` sees them is worth an order of magnitude.
fn might_carry_usage(line: &str, provider: UsageProvider) -> bool {
    match provider {
        UsageProvider::Claude => line.contains("\"usage\""),
        UsageProvider::Codex => line.contains("\"token_count\""),
        // Both Kimi shapes carry usage on a line naming it; the modern
        // `usage.record` and the older `StatusUpdate`'s `token_usage`.
        UsageProvider::Kimi => {
            line.contains("\"token_usage\"") || line.contains("\"usage.record\"")
        }
        UsageProvider::DeepSeek => line.contains("\"inputTokens\""),
        // The Pi agent session format, shared by Pi and Oh My Pi.
        UsageProvider::OhMyPi | UsageProvider::Pi => line.contains("\"usage\""),
    }
}

/// Lines that carry no usage themselves but advance a parser's rolling
/// session state (identity, working directory, or model).
fn carries_scan_context(line: &str, provider: UsageProvider) -> bool {
    match provider {
        UsageProvider::Codex => {
            line.contains("\"turn_context\"") || line.contains("\"session_meta\"")
        }
        UsageProvider::OhMyPi | UsageProvider::Pi => {
            line.contains("\"type\":\"session\"") || line.contains("\"model_change\"")
        }
        UsageProvider::DeepSeek => {
            line.contains("\"type\":\"session\"") || line.contains("\"request/header\"")
        }
        UsageProvider::Claude | UsageProvider::Kimi => false,
    }
}

/// Positive finite number as a token count; anything else is zero.
fn int(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value.trunc() as u64)
        .unwrap_or(0)
}

fn parse_timestamp_ms(value: Option<&Value>) -> Option<i64> {
    let text = value?.as_str()?;
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|date| date.timestamp_millis())
}

/// Parses one line of a Claude Code transcript.
///
/// The CLI writes one record per assistant *content block*, and every one of
/// those records repeats the same complete `usage` object for the parent
/// message. Summing them overcounts severely, so callers must drop repeats by
/// `dedupe_key` and keep the first.
fn parse_claude_line(line: &str) -> Option<UsageRecord> {
    let record: Value = serde_json::from_str(line).ok()?;
    if record.get("type").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    let message = record.get("message")?.as_object()?;
    let usage = message.get("usage")?.as_object()?;
    let timestamp_ms = parse_timestamp_ms(record.get("timestamp"))?;
    let model = message.get("model").and_then(Value::as_str)?;
    if model.is_empty() {
        return None;
    }

    let message_id = message.get("id").and_then(Value::as_str);
    let request_id = record.get("requestId").and_then(Value::as_str);
    // Matches ccusage: prefer the message/request pair, fall back to whichever
    // half exists. Records with neither cannot be de-duplicated.
    let dedupe_key = (message_id.is_some() || request_id.is_some()).then(|| {
        format!(
            "{}:{}",
            message_id.unwrap_or_default(),
            request_id.unwrap_or_default()
        )
    });

    Some(UsageRecord {
        provider: UsageProvider::Claude,
        timestamp_ms,
        model: model.to_owned(),
        session_id: record
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        project: record
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        totals: TokenTotals {
            uncached_input: int(usage.get("input_tokens")),
            cached_input: int(usage.get("cache_read_input_tokens")),
            cache_creation: int(usage.get("cache_creation_input_tokens")),
            output: int(usage.get("output_tokens")),
            // Anthropic folds thinking tokens into output without a breakout.
            reasoning: 0,
        },
        reported_cost_usd: record
            .get("costUSD")
            .and_then(Value::as_f64)
            .filter(|cost| cost.is_finite()),
        dedupe_key,
    })
}

/// Rolling state for a single Codex rollout file. `token_count` events carry
/// no model, so the model is carried forward from the most recent
/// `turn_context`; sessions that switch models mid-run attribute correctly
/// from the switch onward.
#[derive(Clone, Copy)]
enum CodexForkPrefix {
    NotForked,
    /// The first meta marks a fork, but no copied ancestor meta has confirmed
    /// that this physical rollout actually contains a replay prefix.
    AwaitingReplay {
        fork_ms: Option<i64>,
    },
    /// Leading usage after a copied ancestor meta is re-stamped parent history.
    Suppressing {
        anchor_ms: i64,
    },
    Complete,
}

struct CodexScanState {
    model: String,
    session_id: String,
    /// Carried from `session_meta`/`turn_context`, which both record the
    /// working directory.
    cwd: String,
    last_usage_signature: Option<String>,
    saw_session_meta: bool,
    fork_prefix: CodexForkPrefix,
}

impl CodexScanState {
    fn new() -> Self {
        Self {
            model: String::new(),
            session_id: String::new(),
            cwd: String::new(),
            last_usage_signature: None,
            saw_session_meta: false,
            fork_prefix: CodexForkPrefix::NotForked,
        }
    }
}

/// Copied history is written synchronously, while the child's first usage
/// follows a real model turn. This is only a fallback for rollout formats that
/// do not repeat the child's own `session_meta` at the structural boundary.
const FORK_COPY_MAX_GAP_MS: i64 = 1_000;

fn codex_session_id(payload: &serde_json::Map<String, Value>) -> Option<&str> {
    payload
        .get("id")
        .or_else(|| payload.get("session_id"))
        .and_then(Value::as_str)
}

fn is_forked_codex_session(payload: &serde_json::Map<String, Value>) -> bool {
    payload
        .get("forked_from_id")
        .and_then(Value::as_str)
        .is_some()
        || payload
            .get("source")
            .and_then(|source| source.get("subagent"))
            .and_then(|subagent| subagent.get("thread_spawn"))
            .and_then(|spawn| spawn.get("parent_thread_id"))
            .and_then(Value::as_str)
            .is_some()
}

/// Feeds one line of a Codex rollout into `state`, returning a record when the
/// line was a usage event. Deltas come from `last_token_usage`; summing those
/// across a session reconciles with the session's final `total_token_usage`
/// provided consecutive duplicate events are dropped, which this does.
fn parse_codex_line(line: &str, state: &mut CodexScanState) -> Option<UsageRecord> {
    let record: Value = serde_json::from_str(line).ok()?;
    let payload = record.get("payload")?.as_object()?;

    match record.get("type").and_then(Value::as_str) {
        Some("session_meta") => {
            let id = codex_session_id(payload);
            if !state.saw_session_meta {
                // The first meta names the physical rollout. Forks replay
                // ancestor metas after it, and those must never replace the
                // child's identity or working directory.
                state.saw_session_meta = true;
                if let Some(id) = id {
                    state.session_id = id.to_owned();
                }
                if let Some(cwd) = payload.get("cwd").and_then(Value::as_str) {
                    state.cwd = cwd.to_owned();
                }
                state.fork_prefix = if is_forked_codex_session(payload) {
                    CodexForkPrefix::AwaitingReplay {
                        fork_ms: parse_timestamp_ms(record.get("timestamp")),
                    }
                } else {
                    CodexForkPrefix::NotForked
                };
                return None;
            }

            match state.fork_prefix {
                CodexForkPrefix::AwaitingReplay { fork_ms } => {
                    if id == Some(state.session_id.as_str()) {
                        // Some Codex writers repeat the child's meta when copied
                        // history ends. This is the authoritative boundary.
                        state.fork_prefix = CodexForkPrefix::Complete;
                    } else if id.is_some() {
                        // A different meta confirms that ancestor history really
                        // follows. A fork marker alone is not enough: subagents can
                        // start with no copied prefix and may report usage quickly.
                        if let Some(anchor_ms) =
                            parse_timestamp_ms(record.get("timestamp")).or(fork_ms)
                        {
                            state.fork_prefix = CodexForkPrefix::Suppressing { anchor_ms };
                        }
                    }
                }
                CodexForkPrefix::Suppressing { anchor_ms } => {
                    if id == Some(state.session_id.as_str()) {
                        state.fork_prefix = CodexForkPrefix::Complete;
                    } else if id.is_some() {
                        state.fork_prefix = CodexForkPrefix::Suppressing {
                            anchor_ms: parse_timestamp_ms(record.get("timestamp"))
                                .unwrap_or(anchor_ms),
                        };
                    }
                }
                CodexForkPrefix::NotForked | CodexForkPrefix::Complete => {}
            }
            return None;
        }
        Some("turn_context") => {
            if let Some(model) = payload.get("model").and_then(Value::as_str) {
                state.model = model.to_owned();
            }
            if let Some(cwd) = payload.get("cwd").and_then(Value::as_str) {
                state.cwd = cwd.to_owned();
            }
            return None;
        }
        _ => {}
    }

    if payload.get("type").and_then(Value::as_str) != Some("token_count") {
        return None;
    }
    let last = payload.get("info")?.get("last_token_usage")?.as_object()?;

    // Only an event that is otherwise eligible may consume the duplicate
    // signature. A token_count arriving before its turn_context (no model yet)
    // must not poison it, or the re-emitted copy after the model is known
    // would be skipped as a duplicate and those tokens never counted.
    let timestamp_ms = parse_timestamp_ms(record.get("timestamp"))?;
    if state.model.is_empty() {
        return None;
    }

    // Codex re-emits an unchanged token_count on some stream boundaries.
    // Summing those would double count, so identical consecutive payloads are
    // skipped.
    let signature = serde_json::to_string(last).ok()?;
    if state.last_usage_signature.as_deref() == Some(signature.as_str()) {
        return None;
    }
    state.last_usage_signature = Some(signature);

    if let CodexForkPrefix::Suppressing { anchor_ms } = state.fork_prefix {
        if timestamp_ms.saturating_sub(anchor_ms) < FORK_COPY_MAX_GAP_MS {
            state.fork_prefix = CodexForkPrefix::Suppressing {
                anchor_ms: timestamp_ms,
            };
            return None;
        }
        state.fork_prefix = CodexForkPrefix::Complete;
    }

    let input = int(last.get("input_tokens"));
    let cached_input = int(last.get("cached_input_tokens"));
    let cache_creation = int(last.get("cache_write_input_tokens"));
    let output = int(last.get("output_tokens"));
    let totals = TokenTotals {
        // Codex reports `input_tokens` inclusive of the cached portion.
        uncached_input: input.saturating_sub(cached_input + cache_creation),
        cached_input,
        cache_creation,
        output,
        // Reported inside output_tokens, surfaced separately for the mix.
        reasoning: int(last.get("reasoning_output_tokens")).min(output),
    };
    if totals.total() == 0 {
        return None;
    }
    if matches!(state.fork_prefix, CodexForkPrefix::AwaitingReplay { .. }) {
        // Once this physical rollout emits real usage, a replay prefix cannot
        // begin later in the file.
        state.fork_prefix = CodexForkPrefix::Complete;
    }

    Some(UsageRecord {
        provider: UsageProvider::Codex,
        timestamp_ms,
        model: state.model.clone(),
        session_id: state.session_id.clone(),
        project: state.cwd.clone(),
        totals,
        // Codex does not report cost in the rollout.
        reported_cost_usd: None,
        // Fork copies were suppressed while parsing the physical rollout, so
        // surviving events are unique and need no global deduplication.
        dedupe_key: None,
    })
}

/// Rolling state for a Pi-agent session transcript. Pi and Oh My Pi share the
/// format: a leading `session` line carries identity and the working
/// directory, `model_change` lines carry the active model forward for usage
/// records that lack one — the two CLIs disagree only on how `model_change`
/// spells that model.
struct PiScanState {
    session_id: String,
    cwd: String,
    fallback_model: String,
}

impl PiScanState {
    fn new() -> Self {
        Self {
            session_id: String::new(),
            cwd: String::new(),
            fallback_model: String::new(),
        }
    }
}

/// Parses one line of a Pi-agent transcript (Pi and Oh My Pi). Each assistant
/// message carries its own complete usage and cost, so no per-file repeats
/// need dropping; message ids still dedupe history copied into forks.
fn parse_pi_line(
    line: &str,
    provider: UsageProvider,
    state: &mut PiScanState,
) -> Option<UsageRecord> {
    let record: Value = serde_json::from_str(line).ok()?;
    match record.get("type").and_then(Value::as_str) {
        Some("session") => {
            if let Some(id) = record.get("id").and_then(Value::as_str) {
                state.session_id = id.to_owned();
            }
            if let Some(cwd) = record.get("cwd").and_then(Value::as_str) {
                state.cwd = cwd.to_owned();
            }
            return None;
        }
        Some("model_change") => {
            // The two CLIs do not agree on the field: Oh My Pi writes the
            // qualified `provider/model` as `model`, Pi splits it into
            // `provider` and `modelId`.
            if let Some(model) = record.get("model").and_then(Value::as_str) {
                state.fallback_model = model.to_owned();
            } else if let Some(model_id) = record.get("modelId").and_then(Value::as_str) {
                state.fallback_model = match record.get("provider").and_then(Value::as_str) {
                    Some(provider) if !provider.is_empty() => format!("{provider}/{model_id}"),
                    _ => model_id.to_owned(),
                };
            }
            return None;
        }
        _ => {}
    }
    if record.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }
    let message = record.get("message")?.as_object()?;
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    let usage = message.get("usage")?.as_object()?;
    let timestamp_ms = parse_timestamp_ms(record.get("timestamp"))?;
    let model = message
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .unwrap_or(&state.fallback_model);
    if model.is_empty() {
        return None;
    }

    let output = int(usage.get("output"));
    let totals = TokenTotals {
        uncached_input: int(usage.get("input")),
        cached_input: int(usage.get("cacheRead")),
        cache_creation: int(usage.get("cacheWrite")),
        output,
        // Reported inside output, surfaced separately for the mix.
        reasoning: int(usage.get("reasoning")).min(output),
    };
    if totals.total() == 0 {
        return None;
    }

    Some(UsageRecord {
        provider,
        timestamp_ms,
        model: model.to_owned(),
        session_id: state.session_id.clone(),
        project: state.cwd.clone(),
        totals,
        reported_cost_usd: usage
            .get("cost")
            .and_then(|cost| cost.get("total"))
            .and_then(Value::as_f64)
            .filter(|cost| cost.is_finite()),
        dedupe_key: record
            .get("id")
            .and_then(Value::as_str)
            .map(|id| format!("pi:{id}")),
    })
}

/// Session identity for a Kimi Code wire transcript. Neither record shape
/// carries a session id, so it comes from the directory the wire log sits in.
/// The modern layout nests `<workspace>/<session>/agents/<agent>/` and the
/// older one `<workspace>/<session>/`. The agent name is tracked separately
/// from the session because each subagent logs its own usage at the same wall
/// clock as its parent, so only session *and* agent make a record unique —
/// while the sessions the page counts remain one per session directory.
struct KimiScanState {
    /// The session the log belongs to: one per session directory.
    session_id: String,
    /// The agent whose log this is; empty for the single-log older layout.
    agent: String,
    /// The workspace directory key, resolved to a launch path by the scan.
    workspace: String,
}

/// Parses one line of a Kimi Code `wire.jsonl`, in either of the two shapes
/// Kimi has written.
///
/// The modern CLI (`~/.kimi-code`) records one `usage.record` per completed
/// step, already scoped to that step, with camelCase counters. The older one
/// (`~/.kimi`, pre-migration) emits a `StatusUpdate` per turn whose
/// `token_usage` uses snake_case. Both are per-request, so records sum
/// directly with no delta arithmetic.
fn parse_kimi_line(line: &str, state: &KimiScanState) -> Option<UsageRecord> {
    let record: Value = serde_json::from_str(line).ok()?;
    let (timestamp_ms, usage, model, dedupe_key) =
        if record.get("type").and_then(Value::as_str) == Some("usage.record") {
            // A session-scoped rollup repeats the whole session's usage, so
            // counting it would double every step already recorded.
            if record.get("usageScope").and_then(Value::as_str) != Some("turn") {
                return None;
            }
            let usage = record.get("usage")?.as_object()?;
            let timestamp_ms = record
                .get("time")
                .and_then(Value::as_f64)
                .filter(|time| time.is_finite())
                .map(|time| time.trunc() as i64)?;
            (
                timestamp_ms,
                TokenTotals {
                    uncached_input: int(usage.get("inputOther")),
                    cached_input: int(usage.get("inputCacheRead")),
                    cache_creation: int(usage.get("inputCacheCreation")),
                    output: int(usage.get("output")),
                    reasoning: 0,
                },
                record
                    .get("model")
                    .and_then(Value::as_str)
                    .filter(|model| !model.is_empty())
                    .map(str::to_owned),
                // One record per step. Two agents of the same session can
                // record the same wall clock, so the agent names the log.
                Some(format!(
                    "kimi:{}:{}:{}",
                    state.session_id, state.agent, timestamp_ms
                )),
            )
        } else {
            // Wire timestamps are fractional seconds since the epoch.
            let timestamp_ms = record
                .get("timestamp")
                .and_then(Value::as_f64)
                .filter(|seconds| seconds.is_finite())
                .map(|seconds| (seconds * 1000.0).trunc() as i64)?;
            let message = record.get("message")?.as_object()?;
            if message.get("type").and_then(Value::as_str) != Some("StatusUpdate") {
                return None;
            }
            let payload = message.get("payload")?.as_object()?;
            let usage = payload.get("token_usage")?.as_object()?;
            (
                timestamp_ms,
                TokenTotals {
                    uncached_input: int(usage.get("input_other")),
                    cached_input: int(usage.get("input_cache_read")),
                    cache_creation: int(usage.get("input_cache_creation")),
                    output: int(usage.get("output")),
                    reasoning: 0,
                },
                // The old shape writes no model anywhere in its transcripts; the
                // family label reports as unpriced rather than guessing one.
                None,
                payload
                    .get("message_id")
                    .and_then(Value::as_str)
                    .map(|id| format!("kimi:{}:{}", state.session_id, id)),
            )
        };
    if usage.total() == 0 {
        return None;
    }

    Some(UsageRecord {
        provider: UsageProvider::Kimi,
        timestamp_ms,
        model: model.unwrap_or_else(|| "kimi-code".to_owned()),
        session_id: state.session_id.clone(),
        project: state.workspace.clone(),
        totals: usage,
        reported_cost_usd: None,
        dedupe_key,
    })
}

/// Rolling state for a DeepSeek Harness session. Usage chunks carry no model,
/// so it is carried forward from the most recent `request/header`.
struct DshScanState {
    session_id: String,
    cwd: String,
    model: String,
}

impl DshScanState {
    fn new() -> Self {
        Self {
            session_id: String::new(),
            cwd: String::new(),
            model: String::new(),
        }
    }
}

/// Parses one line of a DeepSeek Harness session log. Each request's final
/// usage arrives as an `assistant/chunk` of type `usage`; `seq` numbers every
/// line, which keys cross-file deduplication.
fn parse_dsh_line(line: &str, state: &mut DshScanState) -> Option<UsageRecord> {
    let record: Value = serde_json::from_str(line).ok()?;
    match record.get("type").and_then(Value::as_str) {
        Some("session") => {
            if let Some(id) = record.get("id").and_then(Value::as_str) {
                state.session_id = id.to_owned();
            }
            if let Some(cwd) = record.get("cwd").and_then(Value::as_str) {
                state.cwd = cwd.to_owned();
            }
            return None;
        }
        Some("request/header") => {
            if let Some(model) = record
                .get("data")
                .and_then(|data| data.get("header"))
                .and_then(|header| header.get("config"))
                .and_then(|config| config.get("model"))
                .and_then(Value::as_str)
            {
                state.model = model.to_owned();
            }
            return None;
        }
        _ => {}
    }
    if record.get("type").and_then(Value::as_str) != Some("assistant/chunk") {
        return None;
    }
    let chunk = record.get("data")?.get("chunk")?.as_object()?;
    if chunk.get("type").and_then(Value::as_str) != Some("usage") {
        return None;
    }
    let usage = chunk.get("usage")?.as_object()?;
    let timestamp_ms = record
        .get("time")
        .and_then(Value::as_f64)
        .filter(|time| time.is_finite())
        .map(|time| time.trunc() as i64)?;
    // A usage chunk before any request/header has no model to attribute.
    if state.model.is_empty() {
        return None;
    }
    let totals = TokenTotals {
        uncached_input: int(usage.get("inputTokens")),
        output: int(usage.get("outputTokens")),
        ..TokenTotals::default()
    };
    if totals.total() == 0 {
        return None;
    }

    Some(UsageRecord {
        provider: UsageProvider::DeepSeek,
        timestamp_ms,
        model: state.model.clone(),
        session_id: state.session_id.clone(),
        project: state.cwd.clone(),
        totals,
        reported_cost_usd: None,
        dedupe_key: record
            .get("seq")
            .and_then(Value::as_i64)
            .map(|seq| format!("dsh:{}:{seq}", state.session_id)),
    })
}

/* ------------------------------------------------------------------------- */
/* Pricing                                                                   */
/* ------------------------------------------------------------------------- */

/// USD per token at the base tier. Tiered variants (long-context, flex,
/// batch) are deliberately ignored: transcripts don't record which tier
/// served a request, so anything else would be a guess dressed as precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelRate {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_creation: f64,
}

#[derive(Clone, Debug)]
pub struct RateTable {
    pub rates: HashMap<String, ModelRate>,
    pub status: PricingStatus,
}

impl RateTable {
    pub fn unavailable() -> Self {
        Self {
            rates: HashMap::new(),
            status: PricingStatus::Unavailable,
        }
    }
}

/// Models never priced regardless of the table: `<synthetic>` marks locally
/// generated messages that were never billed, and bare family names are
/// ambiguous across generations, so they report as unpriced instead of
/// guessing one.
const UNPRICEABLE_MODELS: [&str; 6] = [
    "<synthetic>",
    "synthetic",
    "opus",
    "sonnet",
    "haiku",
    "fable",
];

/// Canonicalises a model name for lookup: strips a `provider/` prefix
/// (LiteLLM publishes both `claude-opus-5` and `anthropic/claude-opus-5`) and
/// lowercases, since transcripts are inconsistent about casing.
fn normalize_model_name(model: &str) -> String {
    let trimmed = model.trim().to_ascii_lowercase();
    match trimmed.rfind('/') {
        Some(slash) => trimmed[slash + 1..].to_owned(),
        None => trimmed,
    }
}

fn lookup_rate<'a>(table: &'a RateTable, model: &str) -> Option<&'a ModelRate> {
    let normalized = normalize_model_name(model);
    if normalized.is_empty() || UNPRICEABLE_MODELS.contains(&normalized.as_str()) {
        return None;
    }
    table.rates.get(&normalized)
}

/// Projects the LiteLLM document into a rate table. Entries without both an
/// input and an output rate are dropped: a half-priced model would silently
/// under-report cost, which is worse than reporting it as unpriced.
fn parse_rate_table(document: &Value) -> HashMap<String, ModelRate> {
    let mut table = HashMap::new();
    let Some(entries) = document.as_object() else {
        return table;
    };
    let finite = |value: Option<&Value>| value.and_then(Value::as_f64).filter(|v| v.is_finite());
    for (name, entry) in entries {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let Some(input) = finite(entry.get("input_cost_per_token")) else {
            continue;
        };
        let Some(output) = finite(entry.get("output_cost_per_token")) else {
            continue;
        };
        table.insert(
            normalize_model_name(name),
            ModelRate {
                input,
                output,
                // Anthropic bills cache reads at a discount and writes at a
                // premium. When a model omits them, cached input is priced as
                // plain input rather than as free.
                cache_read: finite(entry.get("cache_read_input_token_cost")).unwrap_or(input),
                cache_creation: finite(entry.get("cache_creation_input_token_cost"))
                    .unwrap_or(input),
            },
        );
    }
    table
}

fn unix_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

fn read_rates_cache(path: &Path) -> Option<(i64, HashMap<String, ModelRate>)> {
    let document: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let fetched_at_ms = document.get("fetched_at_ms")?.as_i64()?;
    let mut rates = HashMap::new();
    for (name, entry) in document.get("rates")?.as_object()? {
        let values = entry.as_array()?;
        let field = |index: usize| values.get(index).and_then(Value::as_f64);
        rates.insert(
            name.clone(),
            ModelRate {
                input: field(0)?,
                output: field(1)?,
                cache_read: field(2)?,
                cache_creation: field(3)?,
            },
        );
    }
    Some((fetched_at_ms, rates))
}

fn write_rates_cache(path: &Path, fetched_at_ms: i64, rates: &HashMap<String, ModelRate>) {
    let entries: serde_json::Map<String, Value> = rates
        .iter()
        .map(|(name, rate)| {
            (
                name.clone(),
                serde_json::json!([
                    rate.input,
                    rate.output,
                    rate.cache_read,
                    rate.cache_creation
                ]),
            )
        })
        .collect();
    let document = serde_json::json!({ "fetched_at_ms": fetched_at_ms, "rates": entries });
    // Best effort: a cache that fails to write only costs the next scan a
    // refetch.
    let _ = std::fs::write(path, document.to_string());
}

/// Load the LiteLLM rate table, preferring the on-disk snapshot while it is
/// within TTL, refetching otherwise, and degrading to the stale snapshot or an
/// empty table rather than failing the scan. Blocking: disk plus up to one
/// HTTPS round trip. Never call from the UI thread.
pub fn load_rate_table(cache_dir: &Path) -> RateTable {
    let cache_path = cache_dir.join(RATES_CACHE_FILE);
    let disk = read_rates_cache(&cache_path).filter(|(_, rates)| !rates.is_empty());
    let now_ms = unix_time_ms();
    if let Some((fetched_at_ms, rates)) = &disk
        && now_ms.saturating_sub(*fetched_at_ms) < RATES_TTL.as_millis() as i64
    {
        return RateTable {
            rates: rates.clone(),
            status: PricingStatus::Cached,
        };
    }

    let fetched =
        crate::usage::http_get(LITELLM_RATES_URL, &["Accept: application/json".to_owned()])
            .ok()
            .filter(|(status, _)| *status == 200)
            .and_then(|(_, body)| serde_json::from_str::<Value>(&body).ok())
            .map(|document| parse_rate_table(&document))
            .filter(|rates| !rates.is_empty());

    match (fetched, disk) {
        (Some(rates), _) => {
            write_rates_cache(&cache_path, now_ms, &rates);
            RateTable {
                rates,
                status: PricingStatus::Fresh,
            }
        }
        (None, Some((_, rates))) => RateTable {
            rates,
            status: PricingStatus::Cached,
        },
        (None, None) => RateTable::unavailable(),
    }
}

/* ------------------------------------------------------------------------- */
/* Scanning                                                                  */
/* ------------------------------------------------------------------------- */

/// Parsed records for one transcript, memoised by `(size, mtime, provider)`.
/// Records are stored unfiltered by window; the aggregation applies the day
/// filter, so one cache serves every window length.
pub struct FileCacheEntry {
    size: u64,
    mtime_ms: i64,
    provider: UsageProvider,
    records: Vec<UsageRecord>,
}

pub type ScanCache = HashMap<PathBuf, FileCacheEntry>;

/// The transcript roots scanned for one provider.
///
/// Each arm reuses the resolver its own provider module already uses, so the
/// scan reads exactly the transcripts that module would write: env overrides,
/// a configured `sessionDir`, and Oh My Pi's per-profile directories all move
/// sessions off the default path. Kimi has two stores — the modern
/// `~/.kimi-code` the current CLI writes and the pre-migration `~/.kimi` an
/// upgrade left behind — and both are scanned, because that migration copied
/// the conversation rather than the usage records.
fn provider_roots(provider: UsageProvider) -> Vec<PathBuf> {
    let home = || dirs::home_dir().unwrap_or_default();
    let mut roots = match provider {
        UsageProvider::Claude => crate::claude_session::projects_directory()
            .map(|dir| vec![dir])
            .unwrap_or_default(),
        UsageProvider::Codex => match std::env::var_os("CODEX_HOME") {
            Some(dir) if !dir.is_empty() => vec![PathBuf::from(dir).join("sessions")],
            _ => vec![home().join(".codex/sessions")],
        },
        UsageProvider::DeepSeek => match std::env::var_os("DSH_HOME") {
            Some(dir) if !dir.is_empty() => vec![PathBuf::from(dir).join("sessions")],
            _ => vec![home().join(".dsh/sessions")],
        },
        UsageProvider::Kimi => crate::kimi_session::session_home()
            .map(|dir| vec![dir.join("sessions")])
            .unwrap_or_default(),
        UsageProvider::OhMyPi | UsageProvider::Pi => {
            let kind = if provider == UsageProvider::Pi {
                ProviderKind::Pi
            } else {
                ProviderKind::OhMyPi
            };
            crate::pi_session::session_roots(kind).unwrap_or_default()
        }
    };
    // The pre-migration Kimi store is only ever at the default location.
    if provider == UsageProvider::Kimi && std::env::var_os("KIMI_CODE_HOME").is_none() {
        roots.push(home().join(".kimi/sessions"));
    }
    roots
}

/// The transcript file extension one provider writes: `.jsonl` everywhere
/// except DeepSeek Harness, which compresses each session to `.jsonl.zstd`.
fn transcript_extension(provider: UsageProvider) -> &'static str {
    match provider {
        UsageProvider::DeepSeek => "zstd",
        _ => "jsonl",
    }
}

/// Lists a provider's transcripts under `root` modified at or after
/// `since_ms`. Errors on individual entries are swallowed: session files
/// rotate and get removed while the walk is in flight, and a partial listing
/// beats failing the page. Returns the number of files skipped by the mtime
/// prefilter.
fn list_transcript_files(
    root: &Path,
    since_ms: i64,
    extension: &str,
    found: &mut Vec<(PathBuf, u64, i64)>,
) -> usize {
    let mut skipped = 0;
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            skipped += list_transcript_files(&path, since_ms, extension, found);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some(extension) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let mtime_ms = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_millis() as i64)
            .unwrap_or(0);
        if mtime_ms < since_ms {
            skipped += 1;
            continue;
        }
        found.push((path, metadata.len(), mtime_ms));
    }
    skipped
}

/// The session identity a Kimi wire transcript records only implicitly. The
/// modern layout nests the log as `<workspace>/<session>/agents/<agent>/`, so
/// the session is the directory above `agents` and the workspace is the one
/// above that. The older layout nests `<workspace-hash>/<session>/` directly.
fn kimi_scan_state(path: &Path) -> KimiScanState {
    // The log sits in the agent's own directory: `<session>/agents/<agent>/`
    // in the modern layout, `<session>/` in the older one. Where the parent is
    // the `agents` level, the session is one level further up.
    let log_directory = path.parent().unwrap_or(Path::new(""));
    let nest = log_directory.parent().unwrap_or(Path::new(""));
    let modern = nest.file_name().and_then(|name| name.to_str()) == Some("agents");
    let session_directory = if modern {
        nest.parent().unwrap_or(Path::new(""))
    } else {
        log_directory
    };
    let name_of = |path: &Path| {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    KimiScanState {
        session_id: name_of(session_directory),
        agent: if modern {
            name_of(log_directory)
        } else {
            String::new()
        },
        workspace: name_of(session_directory.parent().unwrap_or(Path::new(""))),
    }
}

/// DeepSeek Harness appends to its session log, so its archive holds one zstd
/// frame per flush: `StreamingDecoder` stops at the first frame's end and
/// would silently drop the rest of the session. Walk the frames and hand back
/// the concatenated payload, which is byte-for-byte the original log. A
/// trailing partial frame — a write in flight while this reads — ends the
/// walk with everything decoded so far; the next write changes the file's
/// `(size, mtime)` and the cache rescans it.
fn read_zstd_frames(path: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    let mut content = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let Ok(mut decoder) = ruzstd::decoding::StreamingDecoder::new(&bytes[offset..]) else {
            break;
        };
        if std::io::Read::read_to_end(&mut decoder, &mut content).is_err() {
            break;
        }
        let consumed = decoder.decoder.bytes_read_from_source() as usize;
        if consumed == 0 {
            break;
        }
        offset += consumed;
    }
    Some(content)
}

/// Streams one transcript and returns the usage records it contains, already
/// de-duplicated within the file, or `None` when it could not be read. The
/// distinction matters to the cache: an empty transcript is a stable fact
/// worth memoising, while a transient read failure memoised under the same
/// `(size, mtime)` would silently drop the file's usage until it changes.
fn read_transcript_records(path: &Path, provider: UsageProvider) -> Option<Vec<UsageRecord>> {
    let mut reader: Box<dyn std::io::BufRead> = match provider {
        UsageProvider::DeepSeek => {
            let content = read_zstd_frames(path)?;
            Box::new(std::io::BufReader::new(std::io::Cursor::new(content)))
        }
        _ => Box::new(std::io::BufReader::new(std::fs::File::open(path).ok()?)),
    };
    let mut line = String::new();
    let mut records = Vec::new();
    let mut codex_state = CodexScanState::new();
    let mut pi_state = PiScanState::new();
    let mut dsh_state = DshScanState::new();
    let kimi_state = kimi_scan_state(path);
    let mut seen_in_file = HashSet::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        // Context lines (session headers, model switches) advance a parser's
        // rolling state without carrying usage themselves.
        if !might_carry_usage(&line, provider) && !carries_scan_context(&line, provider) {
            continue;
        }
        let record = match provider {
            UsageProvider::Codex => parse_codex_line(&line, &mut codex_state),
            UsageProvider::Claude => parse_claude_line(&line),
            UsageProvider::OhMyPi | UsageProvider::Pi => {
                parse_pi_line(&line, provider, &mut pi_state)
            }
            UsageProvider::Kimi => parse_kimi_line(&line, &kimi_state),
            UsageProvider::DeepSeek => parse_dsh_line(&line, &mut dsh_state),
        };
        if let Some(record) = record {
            // Every Claude assistant content block repeats the parent
            // message's usage; the first record wins. Cross-file repeats
            // (resumed or forked sessions) are handled by the aggregator's
            // global pass.
            if provider == UsageProvider::Claude
                && let Some(key) = &record.dedupe_key
                && !seen_in_file.insert(key.clone())
            {
                continue;
            }
            records.push(record);
        }
    }
    Some(records)
}

/* ------------------------------------------------------------------------- */
/* Aggregation                                                               */
/* ------------------------------------------------------------------------- */

#[derive(Clone, Copy, PartialEq)]
enum CostSource {
    Reported,
    Priced,
    Unpriced,
}

#[derive(Default)]
struct Bucket {
    totals: TokenTotals,
    cost_usd: f64,
    cache_savings_usd: f64,
    records: u64,
    unpriced_records: u64,
    reported_records: u64,
}

/// Per-project accumulation, keyed by the resolved project path.
#[derive(Default)]
struct ProjectAccumulator {
    cost_usd: f64,
    total_tokens: u64,
    by_provider: [ProviderDay; UsageProvider::COUNT],
    sessions: HashSet<(UsageProvider, String)>,
    /// Cost per model, for the row's "top models" caption.
    models: HashMap<String, f64>,
    last_day: Option<NaiveDate>,
}

/// Folds records into `(day, provider, model)` buckets with global cross-file
/// de-duplication, then derives the page's view.
struct Aggregator {
    since_day: NaiveDate,
    until_day: NaiveDate,
    /// Known project roots, longest path first. A record whose working
    /// directory sits under a root is attributed to that root, so launches
    /// from a repo's subdirectories and worktrees inside it merge into one
    /// project row instead of splintering by cwd.
    project_roots: Vec<PathBuf>,
    buckets: HashMap<(NaiveDate, UsageProvider, String), Bucket>,
    seen: HashSet<String>,
    sessions: HashSet<(UsageProvider, String)>,
    /// Sessions per calendar month (keyed by the month's first day). A
    /// session spanning two months counts in both, which is what "sessions
    /// that month" means.
    month_sessions: HashSet<(NaiveDate, UsageProvider, String)>,
    projects: HashMap<String, ProjectAccumulator>,
}

impl Aggregator {
    fn new(since_day: NaiveDate, until_day: NaiveDate, project_roots: &[PathBuf]) -> Self {
        let mut project_roots = project_roots.to_vec();
        project_roots.sort_by_key(|root| std::cmp::Reverse(root.as_os_str().len()));
        Self {
            since_day,
            until_day,
            project_roots,
            buckets: HashMap::new(),
            seen: HashSet::new(),
            sessions: HashSet::new(),
            month_sessions: HashSet::new(),
            projects: HashMap::new(),
        }
    }

    /// The project a working directory belongs to: the longest known root
    /// containing it, else the directory itself.
    fn resolve_project(&self, cwd: &str) -> String {
        if cwd.is_empty() {
            return String::new();
        }
        let path = Path::new(cwd);
        self.project_roots
            .iter()
            .find(|root| path.starts_with(root))
            .map(|root| root.display().to_string())
            .unwrap_or_else(|| cwd.to_owned())
    }

    fn add(&mut self, record: &UsageRecord, rates: &RateTable) {
        if let Some(key) = &record.dedupe_key
            && !self.seen.insert(key.clone())
        {
            return;
        }
        let Some(timestamp) = Utc.timestamp_millis_opt(record.timestamp_ms).single() else {
            return;
        };
        let day = timestamp.with_timezone(&Local).date_naive();
        if day < self.since_day || day > self.until_day {
            return;
        }

        let (cost_usd, source) = match record.reported_cost_usd {
            Some(cost) => (cost, CostSource::Reported),
            None => match lookup_rate(rates, &record.model) {
                Some(rate) => (
                    record.totals.uncached_input as f64 * rate.input
                        + record.totals.cached_input as f64 * rate.cache_read
                        + record.totals.cache_creation as f64 * rate.cache_creation
                        + record.totals.output as f64 * rate.output,
                    CostSource::Priced,
                ),
                None => (0.0, CostSource::Unpriced),
            },
        };
        // What the cached input would have cost at full input rates, minus
        // what it actually cost. Drives the "cache savings" figure.
        let cache_savings_usd = lookup_rate(rates, &record.model)
            .map(|rate| record.totals.cached_input as f64 * (rate.input - rate.cache_read))
            .unwrap_or(0.0);

        let bucket = self
            .buckets
            .entry((day, record.provider, record.model.clone()))
            .or_default();
        bucket.totals.add(&record.totals);
        bucket.cost_usd += cost_usd;
        bucket.cache_savings_usd += cache_savings_usd;
        bucket.records += 1;
        match source {
            CostSource::Unpriced => bucket.unpriced_records += 1,
            CostSource::Reported => bucket.reported_records += 1,
            CostSource::Priced => {}
        }
        if !record.session_id.is_empty() {
            self.sessions
                .insert((record.provider, record.session_id.clone()));
            self.month_sessions.insert((
                first_of_month(day),
                record.provider,
                record.session_id.clone(),
            ));
        }

        let tokens = record.totals.total();
        let project_key = self.resolve_project(&record.project);
        let project = self.projects.entry(project_key).or_default();
        project.cost_usd += cost_usd;
        project.total_tokens += tokens;
        project.by_provider[record.provider.index()].cost_usd += cost_usd;
        project.by_provider[record.provider.index()].total_tokens += tokens;
        if !record.session_id.is_empty() {
            project
                .sessions
                .insert((record.provider, record.session_id.clone()));
        }
        *project.models.entry(record.model.clone()).or_default() += cost_usd;
        project.last_day = Some(project.last_day.map_or(day, |last| last.max(day)));
    }
}

/// Kimi names each session directory after the workspace it ran in, and the
/// launch directory only survives elsewhere on disk. Recover it by mapping the
/// directory name back to a path, using the records Kimi itself keeps: the
/// modern `workspaces.json` (exact name → root), and the older
/// `session_index.jsonl` plus `kimi.json` work dirs (session id → work dir,
/// and the MD5-named workspace directories). Launches in neither list stay
/// unattributed rather than guessed.
///
/// The map is keyed by both the workspace directory name and the session name,
/// since the two layouts split the identity differently.
fn kimi_project_map(project_roots: &[PathBuf]) -> HashMap<String, String> {
    let home = dirs::home_dir();
    let kimi_code_home = home
        .as_ref()
        .map(|home| home.join(".kimi-code"))
        .unwrap_or_default();
    kimi_project_map_in(home.as_deref(), &kimi_code_home, project_roots)
}

fn kimi_project_map_in(
    home: Option<&Path>,
    kimi_code_home: &Path,
    project_roots: &[PathBuf],
) -> HashMap<String, String> {
    let mut map = HashMap::new();

    // The modern store: one entry per workspace, named exactly as its
    // directory is.
    if let Ok(document) = std::fs::read_to_string(kimi_code_home.join("workspaces.json"))
        && let Ok(root) = serde_json::from_str::<Value>(&document)
        && let Some(workspaces) = root.get("workspaces").and_then(Value::as_object)
    {
        for (name, workspace) in workspaces {
            if let Some(path) = workspace.get("root").and_then(Value::as_str) {
                map.insert(name.clone(), path.to_owned());
            }
        }
    }
    // Sessions migrated from the older store keep their id and record the
    // working directory explicitly.
    if let Ok(document) = std::fs::read_to_string(kimi_code_home.join("session_index.jsonl")) {
        for line in document.lines() {
            let Ok(entry) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if let (Some(session_id), Some(work_dir)) = (
                entry.get("sessionId").and_then(Value::as_str),
                entry.get("workDir").and_then(Value::as_str),
            ) {
                map.insert(session_id.to_owned(), work_dir.to_owned());
            }
        }
    }

    // The older store names its workspace directories after the MD5 of the
    // launch path, so its map is built from the forward digest.
    let mut legacy_paths: Vec<String> = Vec::new();
    if let Some(home) = home
        && let Ok(document) = std::fs::read_to_string(home.join(".kimi/kimi.json"))
        && let Ok(root) = serde_json::from_str::<Value>(&document)
        && let Some(work_dirs) = root.get("work_dirs").and_then(Value::as_array)
    {
        for entry in work_dirs {
            if let Some(path) = entry.get("path").and_then(Value::as_str) {
                legacy_paths.push(path.to_owned());
            }
        }
    }
    for root in project_roots {
        legacy_paths.push(root.display().to_string());
    }
    for path in legacy_paths {
        map.insert(md5_hex(&path), path);
    }
    map
}

/// The MD5 hex digest Kimi Code uses as a session directory's name — verified
/// against `md5("/Users/jiyuliang/工作/DSP-Modeling/Qi-Pure")` on a live
/// `~/.kimi/sessions` tree.
fn md5_hex(path: &str) -> String {
    use std::fmt::Write as _;
    let digest = md5::Md5::digest(path.as_bytes());
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hash, "{byte:02x}");
    }
    hash
}

/// Scan every provider's transcripts for `window` ending today and derive
/// the page snapshot. `project_roots` are the app's known project paths, used
/// to attribute working directories to projects. Blocking; run on the
/// background executor with the caller-owned `cache` and a loaded rate table.
pub fn scan(
    cache: &mut ScanCache,
    rates: &RateTable,
    window: UsageWindow,
    project_roots: &[PathBuf],
) -> UsageHistory {
    let started = Instant::now();
    let (since_day, until_day) = window.bounds(Local::now().date_naive());
    // Local midnight on the window's first day; a DST gap falls back to the
    // day boundary in UTC, which the mtime slack absorbs anyway.
    let window_start_ms = since_day
        .and_hms_opt(0, 0, 0)
        .and_then(|midnight| Local.from_local_datetime(&midnight).earliest())
        .map(|midnight| midnight.timestamp_millis())
        .unwrap_or_else(|| unix_time_ms() - (until_day - since_day).num_days().max(1) * 86_400_000);
    let mtime_cutoff_ms = window_start_ms - MTIME_SLACK.as_millis() as i64;
    // Kimi's project attribution is resolved at scan time, not parse time, so
    // newly known project roots reattribute cached transcripts.
    let kimi_projects = kimi_project_map(project_roots);

    let mut aggregator = Aggregator::new(since_day, until_day, project_roots);
    let mut scanned_files = 0;
    let mut skipped_files = 0;
    let mut errors = Vec::new();

    for provider in UsageProvider::ALL {
        for root in provider_roots(provider) {
            if !root.is_dir() {
                // Provider never used on this machine; zero usage, not an error.
                continue;
            }
            let mut files = Vec::new();
            skipped_files += list_transcript_files(
                &root,
                mtime_cutoff_ms,
                transcript_extension(provider),
                &mut files,
            );
            if files.is_empty() && std::fs::read_dir(&root).is_err() {
                errors.push(format!(
                    "{} transcripts at {} could not be read.",
                    provider.label(),
                    root.display()
                ));
                continue;
            }
            for (path, size, mtime_ms) in files {
                scanned_files += 1;
                let cached = cache.get(&path);
                // Provider is part of the identity: if two providers were ever
                // pointed at one directory, a hit parsed by the other parser must
                // not be reused.
                let records = match cached {
                    Some(entry)
                        if entry.size == size
                            && entry.mtime_ms == mtime_ms
                            && entry.provider == provider =>
                    {
                        &entry.records
                    }
                    _ => match read_transcript_records(&path, provider) {
                        Some(records) => {
                            &cache
                                .entry(path)
                                .insert_entry(FileCacheEntry {
                                    size,
                                    mtime_ms,
                                    provider,
                                    records,
                                })
                                .into_mut()
                                .records
                        }
                        // A read failure is not an empty transcript: caching it
                        // under this (size, mtime) would silently drop the file's
                        // usage until it changes.
                        None => continue,
                    },
                };
                for record in records {
                    // Kimi's transcripts name the working directory only as a
                    // workspace or session directory; resolve it against the
                    // paths Kimi itself remembers, leaving unknown names
                    // unattributed.
                    let resolved;
                    let record = if record.provider == UsageProvider::Kimi {
                        resolved = UsageRecord {
                            project: kimi_projects
                                .get(&record.project)
                                .cloned()
                                .unwrap_or_default(),
                            ..record.clone()
                        };
                        &resolved
                    } else {
                        record
                    };
                    aggregator.add(record, rates);
                }
            }
        }
    }

    derive_history(
        aggregator,
        window,
        since_day,
        until_day,
        rates.status,
        scanned_files,
        skipped_files,
        errors,
        started.elapsed(),
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_history(
    aggregator: Aggregator,
    window: UsageWindow,
    since_day: NaiveDate,
    until_day: NaiveDate,
    pricing: PricingStatus,
    scanned_files: usize,
    skipped_files: usize,
    errors: Vec<String>,
    scan_duration: Duration,
) -> UsageHistory {
    let mut totals = TokenTotals::default();
    let mut cost_usd = 0.0;
    let mut cache_savings_usd = 0.0;
    let mut records = 0;
    let mut unpriced_records = 0;
    let mut reported_records = 0;
    let mut providers: HashMap<UsageProvider, (f64, u64)> = HashMap::new();
    let mut models: HashMap<(UsageProvider, String), (f64, u64)> = HashMap::new();
    let mut daily: HashMap<NaiveDate, DaySlice> = HashMap::new();
    let mut month_models: HashMap<(NaiveDate, String), f64> = HashMap::new();

    for ((day, provider, model), bucket) in &aggregator.buckets {
        let tokens = bucket.totals.total();
        totals.add(&bucket.totals);
        cost_usd += bucket.cost_usd;
        cache_savings_usd += bucket.cache_savings_usd;
        records += bucket.records;
        unpriced_records += bucket.unpriced_records;
        reported_records += bucket.reported_records;

        let provider_entry = providers.entry(*provider).or_default();
        provider_entry.0 += bucket.cost_usd;
        provider_entry.1 += tokens;

        let model_entry = models.entry((*provider, model.clone())).or_default();
        model_entry.0 += bucket.cost_usd;
        model_entry.1 += tokens;

        let day_entry = daily.entry(*day).or_insert_with(|| DaySlice {
            day: *day,
            cost_usd: 0.0,
            total_tokens: 0,
            by_provider: [ProviderDay::default(); UsageProvider::COUNT],
        });
        day_entry.cost_usd += bucket.cost_usd;
        day_entry.total_tokens += tokens;
        day_entry.by_provider[provider.index()].cost_usd += bucket.cost_usd;
        day_entry.by_provider[provider.index()].total_tokens += tokens;

        *month_models
            .entry((first_of_month(*day), model.clone()))
            .or_default() += bucket.cost_usd;
    }

    let total_tokens = totals.total();
    let share = |part: f64, whole: f64| if whole == 0.0 { 0.0 } else { part / whole };

    let mut provider_slices: Vec<ProviderSlice> = providers
        .into_iter()
        .map(
            |(provider, (provider_cost, provider_tokens))| ProviderSlice {
                provider,
                cost_usd: provider_cost,
                total_tokens: provider_tokens,
                cost_share: share(provider_cost, cost_usd),
                token_share: share(provider_tokens as f64, total_tokens as f64),
            },
        )
        .collect();
    // Cost descends; ties break by lane order so an all-unpriced window still
    // renders the same chart legend and columns on every scan instead of
    // following hash-map iteration order.
    provider_slices.sort_by(|a, b| {
        b.cost_usd
            .total_cmp(&a.cost_usd)
            .then(a.provider.index().cmp(&b.provider.index()))
    });

    let mut model_slices: Vec<ModelSlice> = models
        .into_iter()
        .map(
            |((provider, model), (model_cost, model_tokens))| ModelSlice {
                provider,
                model,
                cost_usd: model_cost,
                total_tokens: model_tokens,
                cost_share: share(model_cost, cost_usd),
            },
        )
        .collect();
    model_slices.sort_by(|a, b| {
        b.cost_usd
            .total_cmp(&a.cost_usd)
            .then(b.total_tokens.cmp(&a.total_tokens))
    });

    let mut day_slices: Vec<DaySlice> = daily.into_values().collect();
    day_slices.sort_by_key(|slice| slice.day);

    // Months fold over the day slices; only sessions and the model caption
    // need their own record-level accumulators.
    let mut months: HashMap<NaiveDate, MonthSlice> = HashMap::new();
    for day in &day_slices {
        let month = months
            .entry(first_of_month(day.day))
            .or_insert_with(|| MonthSlice {
                first_day: first_of_month(day.day),
                cost_usd: 0.0,
                total_tokens: 0,
                by_provider: [ProviderDay::default(); UsageProvider::COUNT],
                sessions: 0,
                active_days: 0,
                top_models: Vec::new(),
            });
        month.cost_usd += day.cost_usd;
        month.total_tokens += day.total_tokens;
        for index in 0..month.by_provider.len() {
            month.by_provider[index].cost_usd += day.by_provider[index].cost_usd;
            month.by_provider[index].total_tokens += day.by_provider[index].total_tokens;
        }
        if day.total_tokens > 0 {
            month.active_days += 1;
        }
    }
    for (first_day, _, _) in &aggregator.month_sessions {
        if let Some(month) = months.get_mut(first_day) {
            month.sessions += 1;
        }
    }
    for ((first_day, model), model_cost) in month_models {
        if let Some(month) = months.get_mut(&first_day) {
            month.top_models.push((model, model_cost));
        }
    }
    let mut month_slices: Vec<MonthSlice> = months.into_values().collect();
    month_slices.sort_by_key(|slice| slice.first_day);
    for month in &mut month_slices {
        month.top_models.sort_by(|a, b| b.1.total_cmp(&a.1));
    }

    let mut project_slices: Vec<ProjectSlice> = aggregator
        .projects
        .into_iter()
        .map(|(path, project)| {
            let mut top_models: Vec<(String, f64)> = project.models.into_iter().collect();
            top_models.sort_by(|a, b| b.1.total_cmp(&a.1));
            ProjectSlice {
                path,
                cost_usd: project.cost_usd,
                total_tokens: project.total_tokens,
                by_provider: project.by_provider,
                sessions: project.sessions.len() as u64,
                cost_share: share(project.cost_usd, cost_usd),
                last_day: project.last_day,
                top_models,
            }
        })
        .collect();
    project_slices.sort_by(|a, b| {
        b.cost_usd
            .total_cmp(&a.cost_usd)
            .then(b.total_tokens.cmp(&a.total_tokens))
    });

    let record_share = |part: u64| share(part as f64, records as f64);
    UsageHistory {
        window,
        since_day,
        until_day,
        totals,
        total_tokens,
        cost_usd,
        records,
        sessions: aggregator.sessions.len() as u64,
        providers: provider_slices,
        models: model_slices,
        daily: day_slices,
        months: month_slices,
        projects: project_slices,
        quality: CostQuality {
            provider_reported_share: record_share(reported_records),
            model_priced_share: record_share(records - reported_records - unpriced_records),
            unpriced_share: record_share(unpriced_records),
            cache_savings_usd,
        },
        pricing,
        scanned_files,
        skipped_files,
        errors,
        scan_duration,
    }
}

/// Inclusive day list between the window bounds, oldest first — the chart's
/// x-axis, including days with no activity.
pub fn enumerate_days(since_day: NaiveDate, until_day: NaiveDate) -> Vec<NaiveDate> {
    let mut days = Vec::new();
    let mut cursor = since_day;
    while cursor <= until_day {
        days.push(cursor);
        cursor = cursor + chrono::Days::new(1);
    }
    days
}

/// The first day of `day`'s calendar month.
pub fn first_of_month(day: NaiveDate) -> NaiveDate {
    day.with_day(1).unwrap_or(day)
}

/// Inclusive first-of-month list between the bounds' months, oldest first —
/// the statement view's rows, including months with no activity.
pub fn enumerate_months(since_day: NaiveDate, until_day: NaiveDate) -> Vec<NaiveDate> {
    let mut months = Vec::new();
    let mut cursor = first_of_month(since_day);
    let last = first_of_month(until_day);
    while cursor <= last {
        months.push(cursor);
        let Some(next) = cursor.checked_add_months(chrono::Months::new(1)) else {
            break;
        };
        cursor = next;
    }
    months
}

/// Number of days in `first_day`'s month.
pub fn days_in_month(first_day: NaiveDate) -> u32 {
    first_day
        .checked_add_months(chrono::Months::new(1))
        .and_then(|next| next.pred_opt())
        .map(|last| last.day())
        .unwrap_or(31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate_table(entries: &[(&str, ModelRate)]) -> RateTable {
        RateTable {
            rates: entries
                .iter()
                .map(|(name, rate)| ((*name).to_owned(), *rate))
                .collect(),
            status: PricingStatus::Fresh,
        }
    }

    const FLAT_RATE: ModelRate = ModelRate {
        input: 1e-6,
        output: 2e-6,
        cache_read: 1e-7,
        cache_creation: 1.25e-6,
    };

    fn codex_meta(
        timestamp: &str,
        id: &str,
        cwd: &str,
        forked_from_id: Option<&str>,
        parent_thread_id: Option<&str>,
    ) -> String {
        let mut payload = serde_json::json!({ "id": id, "cwd": cwd });
        if let Some(parent) = forked_from_id {
            payload["forked_from_id"] = Value::String(parent.to_owned());
        }
        if let Some(parent) = parent_thread_id {
            payload["source"] = serde_json::json!({
                "subagent": { "thread_spawn": { "parent_thread_id": parent } }
            });
        }
        serde_json::json!({
            "timestamp": timestamp,
            "type": "session_meta",
            "payload": payload,
        })
        .to_string()
    }

    fn codex_context(timestamp: &str, cwd: &str) -> String {
        serde_json::json!({
            "timestamp": timestamp,
            "type": "turn_context",
            "payload": { "model": "gpt-5.3-codex", "cwd": cwd },
        })
        .to_string()
    }

    fn codex_count(timestamp: &str, input: u64, output: u64) -> String {
        serde_json::json!({
            "timestamp": timestamp,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": input,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": output,
                        "reasoning_output_tokens": 0,
                    }
                }
            },
        })
        .to_string()
    }

    #[test]
    fn claude_lines_parse_usage_and_dedupe_key() {
        // Shape captured from a live transcript on 2026-08-08.
        let line = r#"{"type":"assistant","timestamp":"2026-08-08T15:18:37.487Z",
            "requestId":"req_1","sessionId":"session-1","costUSD":null,
            "cwd":"/Users/me/dev/waku",
            "message":{"id":"msg_1","model":"claude-fable-5",
            "usage":{"input_tokens":2,"cache_creation_input_tokens":50700,
            "cache_read_input_tokens":0,"output_tokens":1238}}}"#
            .replace('\n', " ");
        let record = parse_claude_line(&line).expect("the line carries usage");
        assert_eq!(record.provider, UsageProvider::Claude);
        assert_eq!(record.model, "claude-fable-5");
        assert_eq!(record.session_id, "session-1");
        assert_eq!(record.project, "/Users/me/dev/waku");
        assert_eq!(record.dedupe_key.as_deref(), Some("msg_1:req_1"));
        assert_eq!(record.reported_cost_usd, None);
        assert_eq!(record.totals.uncached_input, 2);
        assert_eq!(record.totals.cache_creation, 50700);
        assert_eq!(record.totals.output, 1238);
        assert_eq!(record.totals.total(), 2 + 50700 + 1238);

        // Non-assistant lines and lines without usage parse to nothing.
        assert!(parse_claude_line(r#"{"type":"user","message":{}}"#).is_none());
    }

    #[test]
    fn codex_lines_carry_model_forward_and_skip_duplicates() {
        let mut state = CodexScanState::new();
        let meta = r#"{"timestamp":"2026-08-06T16:31:19.166Z","type":"session_meta",
            "payload":{"id":"codex-session","cwd":"/Users/me/dev/waku"}}"#
            .replace('\n', " ");
        let context = r#"{"timestamp":"2026-08-06T16:31:20.000Z","type":"turn_context",
            "payload":{"model":"gpt-5.3-codex"}}"#
            .replace('\n', " ");
        let count = r#"{"timestamp":"2026-08-06T16:31:25.000Z","type":"event_msg",
            "payload":{"type":"token_count","info":{"last_token_usage":{
            "input_tokens":21047,"cached_input_tokens":1000,"cache_write_input_tokens":47,
            "output_tokens":280,"reasoning_output_tokens":165}}}}"#
            .replace('\n', " ");

        // A token_count before any turn_context has no model and is dropped
        // without consuming the duplicate signature.
        assert!(parse_codex_line(&count, &mut state).is_none());
        assert!(parse_codex_line(&meta, &mut state).is_none());
        assert!(parse_codex_line(&context, &mut state).is_none());

        let record = parse_codex_line(&count, &mut state).expect("model is known now");
        assert_eq!(record.model, "gpt-5.3-codex");
        assert_eq!(record.session_id, "codex-session");
        assert_eq!(record.project, "/Users/me/dev/waku");
        // input_tokens includes the cached portion.
        assert_eq!(record.totals.uncached_input, 21047 - 1000 - 47);
        assert_eq!(record.totals.cached_input, 1000);
        assert_eq!(record.totals.cache_creation, 47);
        assert_eq!(record.totals.reasoning, 165);

        // The identical re-emitted event is a duplicate, not more usage.
        assert!(parse_codex_line(&count, &mut state).is_none());
    }

    #[test]
    fn codex_fork_uses_session_metas_as_the_copy_boundary() {
        let mut state = CodexScanState::new();
        parse_codex_line(
            &codex_meta(
                "2026-08-06T16:31:19.000Z",
                "child",
                "/child",
                Some("parent"),
                None,
            ),
            &mut state,
        );
        // Copied history is re-stamped to the fork instant. Its ancestor meta
        // confirms that the following usage is replay, while leaving the
        // physical child's identity and cwd intact.
        parse_codex_line(
            &codex_meta("2026-08-06T16:31:19.001Z", "parent", "/parent", None, None),
            &mut state,
        );
        parse_codex_line(
            &codex_context("2026-08-06T16:31:19.002Z", "/parent"),
            &mut state,
        );
        assert!(
            parse_codex_line(
                &codex_count("2026-08-06T16:31:19.003Z", 100, 10),
                &mut state,
            )
            .is_none()
        );

        // This writer repeats the child's meta at the exact structural
        // boundary. Genuine child usage must count even inside the fallback's
        // one-second window.
        parse_codex_line(
            &codex_meta(
                "2026-08-06T16:31:19.004Z",
                "child",
                "/child",
                Some("parent"),
                None,
            ),
            &mut state,
        );
        parse_codex_line(
            &codex_context("2026-08-06T16:31:19.050Z", "/child"),
            &mut state,
        );
        let child = parse_codex_line(
            &codex_count("2026-08-06T16:31:19.100Z", 300, 30),
            &mut state,
        )
        .expect("usage after the child's meta is genuine");
        assert_eq!(child.session_id, "child");
        assert_eq!(child.project, "/child");
        assert_eq!(child.totals.total(), 330);
    }

    #[test]
    fn codex_subagent_without_an_ancestor_meta_counts_fast_usage() {
        let mut state = CodexScanState::new();
        parse_codex_line(
            &codex_meta(
                "2026-08-06T16:31:19.000Z",
                "child",
                "/child",
                None,
                Some("parent"),
            ),
            &mut state,
        );
        parse_codex_line(
            &codex_context("2026-08-06T16:31:19.050Z", "/child"),
            &mut state,
        );
        let child = parse_codex_line(
            &codex_count("2026-08-06T16:31:19.100Z", 100, 10),
            &mut state,
        )
        .expect("a fork marker alone does not prove history was copied");
        assert_eq!(child.session_id, "child");
    }

    #[test]
    fn codex_fork_falls_back_to_the_gap_when_no_child_meta_repeats() {
        let mut state = CodexScanState::new();
        parse_codex_line(
            &codex_meta(
                "2026-08-06T16:31:19.000Z",
                "child",
                "/child",
                Some("parent"),
                None,
            ),
            &mut state,
        );
        parse_codex_line(
            &codex_meta("2026-08-06T16:31:19.001Z", "parent", "/parent", None, None),
            &mut state,
        );
        parse_codex_line(
            &codex_context("2026-08-06T16:31:19.001Z", "/parent"),
            &mut state,
        );
        assert!(
            parse_codex_line(
                &codex_count("2026-08-06T16:31:19.002Z", 100, 10),
                &mut state,
            )
            .is_none()
        );
        assert!(
            parse_codex_line(
                &codex_count("2026-08-06T16:31:19.040Z", 200, 20),
                &mut state,
            )
            .is_none()
        );

        let child = parse_codex_line(
            &codex_count("2026-08-06T16:31:20.040Z", 300, 30),
            &mut state,
        )
        .expect("a full-second gap ends copy suppression");
        assert_eq!(child.session_id, "child");
        assert_eq!(child.totals.total(), 330);
    }

    #[test]
    fn aggregation_dedupes_across_files_and_prices_records() {
        let rates = rate_table(&[("claude-fable-5", FLAT_RATE)]);
        let record = UsageRecord {
            provider: UsageProvider::Claude,
            timestamp_ms: Local::now().timestamp_millis(),
            model: "claude-fable-5".to_owned(),
            session_id: "session-1".to_owned(),
            project: "/Users/me/dev/waku/crates/ui".to_owned(),
            totals: TokenTotals {
                uncached_input: 1_000,
                cached_input: 10_000,
                cache_creation: 2_000,
                output: 500,
                reasoning: 0,
            },
            reported_cost_usd: None,
            dedupe_key: Some("msg:req".to_owned()),
        };
        let today = Local::now().date_naive();
        let roots = [PathBuf::from("/Users/me/dev/waku")];
        let mut aggregator = Aggregator::new(today - chrono::Days::new(29), today, &roots);
        aggregator.add(&record, &rates);
        // The same message copied into a forked session's transcript.
        aggregator.add(&record, &rates);

        let history = derive_history(
            aggregator,
            UsageWindow::TrailingDays(30),
            today - chrono::Days::new(29),
            today,
            PricingStatus::Fresh,
            1,
            0,
            Vec::new(),
            Duration::ZERO,
        );
        assert_eq!(history.records, 1);
        assert_eq!(history.sessions, 1);
        assert_eq!(history.total_tokens, 13_500);
        let expected_cost = 1_000.0 * 1e-6 + 10_000.0 * 1e-7 + 2_000.0 * 1.25e-6 + 500.0 * 2e-6;
        assert!((history.cost_usd - expected_cost).abs() < 1e-9);
        // Savings: cached input at full rate minus the cache-read rate.
        let expected_savings = 10_000.0 * (1e-6 - 1e-7);
        assert!((history.quality.cache_savings_usd - expected_savings).abs() < 1e-9);
        assert_eq!(history.daily.len(), 1);
        assert_eq!(history.providers.len(), 1);
        assert!((history.quality.model_priced_share - 1.0).abs() < f64::EPSILON);

        // The subdirectory cwd resolved to its containing project root, and
        // the month fold carries the same totals as the single active day.
        assert_eq!(history.projects.len(), 1);
        let project = &history.projects[0];
        assert_eq!(project.path, "/Users/me/dev/waku");
        assert_eq!(project.sessions, 1);
        assert_eq!(project.total_tokens, 13_500);
        assert!((project.cost_share - 1.0).abs() < f64::EPSILON);
        assert_eq!(project.last_day, Some(today));
        assert_eq!(project.top_models[0].0, "claude-fable-5");
        assert_eq!(history.months.len(), 1);
        let month = &history.months[0];
        assert_eq!(month.first_day, first_of_month(today));
        assert_eq!(month.total_tokens, 13_500);
        assert_eq!(month.sessions, 1);
        assert_eq!(month.active_days, 1);
        assert_eq!(month.top_models[0].0, "claude-fable-5");
    }

    #[test]
    fn months_and_projects_split_across_boundaries() {
        let rates = rate_table(&[("claude-fable-5", FLAT_RATE)]);
        let today = Local::now().date_naive();
        // Two months back so both records always land inside the window even
        // on the first of a month.
        let since = first_of_month(today)
            .checked_sub_months(chrono::Months::new(2))
            .unwrap();
        let roots: [PathBuf; 0] = [];
        let mut aggregator = Aggregator::new(since, today, &roots);
        let record = |days_ago: i64, session: &str, project: &str| UsageRecord {
            provider: UsageProvider::Claude,
            timestamp_ms: Local::now().timestamp_millis() - days_ago * 86_400_000,
            model: "claude-fable-5".to_owned(),
            session_id: session.to_owned(),
            project: project.to_owned(),
            totals: TokenTotals {
                uncached_input: 100,
                ..TokenTotals::default()
            },
            reported_cost_usd: None,
            dedupe_key: None,
        };
        aggregator.add(&record(0, "session-now", "/a"), &rates);
        aggregator.add(&record(45, "session-then", "/b"), &rates);

        let history = derive_history(
            aggregator,
            MONTHLY_WINDOW,
            since,
            today,
            PricingStatus::Fresh,
            0,
            0,
            Vec::new(),
            Duration::ZERO,
        );
        assert_eq!(history.months.len(), 2, "45 days apart spans two months");
        assert!(history.months[0].first_day < history.months[1].first_day);
        assert_eq!(history.months[1].sessions, 1);
        assert_eq!(history.projects.len(), 2);
        // Equal costs tie-break by tokens; both carry half the total each.
        assert!((history.projects[0].cost_share - 0.5).abs() < 1e-9);
        assert!(
            history
                .month(first_of_month(today))
                .is_some_and(|month| month.total_tokens == 100)
        );
    }

    #[test]
    fn out_of_window_and_unpriced_records_are_classified() {
        let rates = RateTable::unavailable();
        let today = Local::now().date_naive();
        let roots: [PathBuf; 0] = [];
        let mut aggregator = Aggregator::new(today, today, &roots);
        let mut record = UsageRecord {
            provider: UsageProvider::Codex,
            timestamp_ms: Local::now().timestamp_millis(),
            model: "gpt-5.3-codex".to_owned(),
            session_id: String::new(),
            project: String::new(),
            totals: TokenTotals {
                uncached_input: 100,
                ..TokenTotals::default()
            },
            reported_cost_usd: None,
            dedupe_key: None,
        };
        aggregator.add(&record, &rates);
        // Ten days ago falls outside a one-day window.
        record.timestamp_ms -= 10 * 86_400_000;
        aggregator.add(&record, &rates);

        let history = derive_history(
            aggregator,
            UsageWindow::TrailingDays(1),
            today,
            today,
            PricingStatus::Unavailable,
            0,
            0,
            Vec::new(),
            Duration::ZERO,
        );
        assert_eq!(history.records, 1);
        assert_eq!(history.sessions, 0, "an empty session id is not a session");
        assert!((history.quality.unpriced_share - 1.0).abs() < f64::EPSILON);
        assert_eq!(history.cost_usd, 0.0);
    }

    #[test]
    fn model_names_normalize_and_family_names_stay_unpriced() {
        assert_eq!(
            normalize_model_name("anthropic/Claude-Fable-5"),
            "claude-fable-5"
        );
        assert_eq!(normalize_model_name("  gpt-5.3-codex "), "gpt-5.3-codex");
        let rates = rate_table(&[("fable", FLAT_RATE), ("claude-fable-5", FLAT_RATE)]);
        assert!(
            lookup_rate(&rates, "Fable").is_none(),
            "bare family names are ambiguous"
        );
        assert!(lookup_rate(&rates, "<synthetic>").is_none());
        assert!(lookup_rate(&rates, "anthropic/claude-fable-5").is_some());
    }

    #[test]
    fn rate_tables_parse_and_round_trip_through_the_disk_cache() {
        let document: Value = serde_json::from_str(
            r#"{
                "claude-fable-5": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6,
                    "cache_read_input_token_cost": 1e-7, "cache_creation_input_token_cost": 1.25e-6},
                "no-output-rate": {"input_cost_per_token": 1e-6},
                "gpt-5.3-codex": {"input_cost_per_token": 3e-6, "output_cost_per_token": 9e-6}
            }"#,
        )
        .unwrap();
        let rates = parse_rate_table(&document);
        assert_eq!(rates.len(), 2, "half-priced models are dropped");
        // Missing cache rates fall back to the input rate, not to free.
        assert_eq!(rates["gpt-5.3-codex"].cache_read, 3e-6);

        let dir = std::env::temp_dir().join(format!("waku-usage-rates-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(RATES_CACHE_FILE);
        write_rates_cache(&path, 12345, &rates);
        let (fetched_at_ms, restored) = read_rates_cache(&path).expect("cache round-trips");
        assert_eq!(fetched_at_ms, 12345);
        assert_eq!(restored.len(), rates.len());
        assert_eq!(restored["claude-fable-5"], rates["claude-fable-5"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn windows_and_month_enumeration_cover_calendars() {
        let day = NaiveDate::from_ymd_opt(2026, 8, 9).unwrap();
        let ymd =
            |year, month, day_of_month| NaiveDate::from_ymd_opt(year, month, day_of_month).unwrap();
        assert_eq!(
            UsageWindow::TrailingDays(7).bounds(day),
            (ymd(2026, 8, 3), day)
        );
        // Twelve months = this month plus the eleven before it.
        assert_eq!(UsageWindow::Months(12).bounds(day), (ymd(2025, 9, 1), day));
        assert_eq!(UsageWindow::ThisMonth.bounds(day), (ymd(2026, 8, 1), day));
        // The whole previous month, ending before today — across a year
        // boundary too.
        assert_eq!(
            UsageWindow::LastMonth.bounds(day),
            (ymd(2026, 7, 1), ymd(2026, 7, 31))
        );
        assert_eq!(
            UsageWindow::LastMonth.bounds(ymd(2026, 1, 15)),
            (ymd(2025, 12, 1), ymd(2025, 12, 31))
        );
        let months = enumerate_months(
            NaiveDate::from_ymd_opt(2025, 11, 15).unwrap(),
            NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
        );
        assert_eq!(months.len(), 4);
        assert_eq!(months[0], NaiveDate::from_ymd_opt(2025, 11, 1).unwrap());
        assert_eq!(months[3], NaiveDate::from_ymd_opt(2026, 2, 1).unwrap());
        assert_eq!(
            days_in_month(NaiveDate::from_ymd_opt(2026, 2, 1).unwrap()),
            28
        );
        assert_eq!(
            days_in_month(NaiveDate::from_ymd_opt(2024, 2, 1).unwrap()),
            29
        );
    }

    #[test]
    fn day_enumeration_is_inclusive() {
        let since = NaiveDate::from_ymd_opt(2026, 8, 1).unwrap();
        let until = NaiveDate::from_ymd_opt(2026, 8, 3).unwrap();
        let days = enumerate_days(since, until);
        assert_eq!(days.len(), 3);
        assert_eq!(days[0], since);
        assert_eq!(days[2], until);
        assert_eq!(enumerate_days(until, since), Vec::<NaiveDate>::new());
    }

    #[test]
    fn pi_lines_parse_usage_model_and_reported_cost() {
        // Shapes captured from live ~/.omp and ~/.pi session transcripts: the
        // two CLIs share one message format but not the `model_change` fields.
        let session = r#"{"type":"session","version":3,"id":"019fe6b1-835b-7000-9fb4-252c89d32281","timestamp":"2026-08-09T13:23:41.019Z","cwd":"/Users/me/dev/waku"}"#;
        let message = r#"{"type":"message","id":"7f49a50f","parentId":"b8a3e260","timestamp":"2026-08-09T13:23:55.721Z","message":{"role":"assistant","content":[{"type":"text","text":"ok"}],"api":"ollama-chat","provider":"ollama-cloud","model":"glm-5.2","usage":{"input":8987,"output":198,"cacheRead":512,"cacheWrite":0,"reasoning":83,"totalTokens":9697,"cost":{"input":0.0026961,"output":0.0002376,"cacheRead":0.000003072,"cacheWrite":0,"total":0.002936772}}}}"#;
        let mut state = PiScanState::new();
        assert!(parse_pi_line(session, UsageProvider::OhMyPi, &mut state).is_none());

        let record =
            parse_pi_line(message, UsageProvider::OhMyPi, &mut state).expect("usage present");
        assert_eq!(record.provider, UsageProvider::OhMyPi);
        assert_eq!(record.model, "glm-5.2");
        assert_eq!(record.session_id, "019fe6b1-835b-7000-9fb4-252c89d32281");
        assert_eq!(record.project, "/Users/me/dev/waku");
        assert_eq!(record.timestamp_ms, 1_786_281_835_721);
        assert_eq!(record.totals.uncached_input, 8987);
        assert_eq!(record.totals.cached_input, 512);
        assert_eq!(record.totals.cache_creation, 0);
        assert_eq!(record.totals.output, 198);
        assert_eq!(record.totals.reasoning, 83);
        assert!((record.reported_cost_usd.unwrap() - 0.002936772).abs() < 1e-12);
        assert_eq!(record.dedupe_key.as_deref(), Some("pi:7f49a50f"));

        // A user line and a usage-less assistant line parse to nothing.
        assert!(parse_pi_line(
            r#"{"type":"message","timestamp":"2026-08-09T13:23:56.000Z","message":{"role":"user","content":[]}}"#,
            UsageProvider::OhMyPi,
            &mut state
        )
        .is_none());
        assert!(parse_pi_line(
            r#"{"type":"message","timestamp":"2026-08-09T13:23:56.000Z","message":{"role":"assistant","content":[]}}"#,
            UsageProvider::OhMyPi,
            &mut state
        )
        .is_none());

        // A record without its own model falls back to `model_change`, whose
        // shape the two CLIs spell differently: Oh My Pi writes the qualified
        // `model`, Pi splits it into `provider` and `modelId`.
        assert!(parse_pi_line(
            r#"{"type":"model_change","id":"108b8777","parentId":null,"timestamp":"2026-08-09T13:23:41.066Z","model":"ollama-cloud/glm-5.2","resolvedModelIsFallback":false}"#,
            UsageProvider::OhMyPi,
            &mut state
        )
        .is_none());
        let bare = parse_pi_line(
            r#"{"type":"message","id":"9faba389","timestamp":"2026-08-09T13:24:01.000Z","message":{"role":"assistant","usage":{"input":10,"output":5,"cacheRead":0,"cacheWrite":0,"reasoning":0,"totalTokens":15}}}"#,
            UsageProvider::OhMyPi,
            &mut state,
        )
        .expect("the fallback model attributes the record");
        assert_eq!(bare.provider, UsageProvider::OhMyPi);
        assert_eq!(bare.model, "ollama-cloud/glm-5.2");
        assert!(bare.reported_cost_usd.is_none());

        // Pi's own split shape, captured from a live ~/.pi transcript.
        let mut pi_state = PiScanState::new();
        assert!(parse_pi_line(
            r#"{"type":"model_change","id":"6418d4c7","parentId":null,"timestamp":"2026-08-10T13:21:32.477Z","provider":"kimi-coding","modelId":"k3-256k"}"#,
            UsageProvider::Pi,
            &mut pi_state
        )
        .is_none());
        let pi_bare = parse_pi_line(
            r#"{"type":"message","id":"5c1e2b04","timestamp":"2026-08-10T13:22:59.804Z","message":{"role":"assistant","usage":{"input":10,"output":5,"cacheRead":0,"cacheWrite":0,"reasoning":0,"totalTokens":15}}}"#,
            UsageProvider::Pi,
            &mut pi_state,
        )
        .expect("Pi's split model_change attributes the record");
        assert_eq!(pi_bare.provider, UsageProvider::Pi);
        assert_eq!(pi_bare.model, "kimi-coding/k3-256k");
    }

    #[test]
    fn kimi_wire_updates_parse_per_step_usage_in_both_formats() {
        // The older shape, captured from a live ~/.kimi/sessions wire.jsonl.
        let legacy = KimiScanState {
            session_id: "9ce64e58-5796-47fd-94f6-e52162446e90".to_owned(),
            agent: String::new(),
            workspace: "b7d87b9c610d01d676c088f03905388d".to_owned(),
        };
        let update = r#"{"timestamp": 1774854982.58214, "message": {"type": "StatusUpdate", "payload": {"context_usage": 0.037, "context_tokens": 9736, "max_context_tokens": 262144, "token_usage": {"input_other": 1800, "output": 90, "input_cache_read": 7936, "input_cache_creation": 0}, "message_id": "chatcmpl-b1dqq0rsBjfW2OPoJRr7eBKv", "plan_mode": false}}}"#;
        let record = parse_kimi_line(update, &legacy).expect("a completed turn carries usage");
        assert_eq!(record.provider, UsageProvider::Kimi);
        assert_eq!(record.model, "kimi-code");
        assert_eq!(record.session_id, "9ce64e58-5796-47fd-94f6-e52162446e90");
        assert_eq!(record.project, "b7d87b9c610d01d676c088f03905388d");
        assert_eq!(record.timestamp_ms, 1_774_854_982_582);
        assert_eq!(record.totals.uncached_input, 1800);
        assert_eq!(record.totals.cached_input, 7936);
        assert_eq!(record.totals.output, 90);
        assert_eq!(
            record.dedupe_key.as_deref(),
            Some("kimi:9ce64e58-5796-47fd-94f6-e52162446e90:chatcmpl-b1dqq0rsBjfW2OPoJRr7eBKv")
        );

        // Other wire traffic and zero-usage updates parse to nothing.
        assert!(
            parse_kimi_line(
                r#"{"timestamp": 1774854982.0, "message": {"type": "TurnBegin", "payload": {}}}"#,
                &legacy
            )
            .is_none()
        );
        assert!(parse_kimi_line(
            r#"{"timestamp": 1774854982.0, "message": {"type": "StatusUpdate", "payload": {"token_usage": {"input_other": 0, "output": 0, "input_cache_read": 0, "input_cache_creation": 0}, "message_id": "chatcmpl-zero"}}}"#,
            &legacy
        )
        .is_none());

        // The modern shape, captured from a live ~/.kimi-code
        // `<session>/agents/main/wire.jsonl` on 2026-09-25.
        let modern = KimiScanState {
            session_id: "session_90dbd78a-5176-4f37-88e8-43d27da01ed4".to_owned(),
            agent: "main".to_owned(),
            workspace: "wd_renamer_77d4205a98d1".to_owned(),
        };
        let step = r#"{"type":"usage.record","model":"kimi-code/kimi-for-coding","usage":{"inputOther":8839,"output":312,"inputCacheRead":14592,"inputCacheCreation":0},"usageScope":"turn","time":1782527245226}"#;
        let record = parse_kimi_line(step, &modern).expect("a completed step carries usage");
        assert_eq!(record.model, "kimi-code/kimi-for-coding");
        assert_eq!(
            record.session_id,
            "session_90dbd78a-5176-4f37-88e8-43d27da01ed4"
        );
        assert_eq!(record.project, "wd_renamer_77d4205a98d1");
        assert_eq!(record.timestamp_ms, 1_782_527_245_226);
        assert_eq!(record.totals.uncached_input, 8839);
        assert_eq!(record.totals.cached_input, 14592);
        assert_eq!(record.totals.output, 312);
        assert_eq!(
            record.dedupe_key.as_deref(),
            Some("kimi:session_90dbd78a-5176-4f37-88e8-43d27da01ed4:main:1782527245226")
        );

        // The session-scope rollup repeats steps already counted, so it must
        // never become a record of its own.
        assert!(parse_kimi_line(
            r#"{"type":"usage.record","model":"kimi-code/kimi-for-coding","usage":{"inputOther":644,"output":2018,"inputCacheRead":207616,"inputCacheCreation":0},"usageScope":"session","time":1782549291718}"#,
            &modern
        )
        .is_none());
        assert!(parse_kimi_line(
            r#"{"type":"usage.record","usage":{"inputOther":0,"output":0,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1782549291718}"#,
            &modern
        )
        .is_none());
    }

    /// The two Kimi layouts place the wire log at different depths, and the
    /// session and workspace names live at different levels.
    #[test]
    fn kimi_scan_state_reads_both_wire_log_layouts() {
        let modern = kimi_scan_state(Path::new(
            "/h/.kimi-code/sessions/wd_renamer_77d4205a98d1/session_90dbd78a/agents/main/wire.jsonl",
        ));
        assert_eq!(modern.session_id, "session_90dbd78a");
        assert_eq!(modern.agent, "main");
        assert_eq!(modern.workspace, "wd_renamer_77d4205a98d1");

        // A subagent's log belongs to the same session but is its own log: two
        // agents can record the same wall clock, and only the agent name keeps
        // their records apart.
        let modern_subagent = kimi_scan_state(Path::new(
            "/h/.kimi-code/sessions/wd_renamer_77d4205a98d1/session_90dbd78a/agents/agent-7/wire.jsonl",
        ));
        assert_eq!(modern_subagent.session_id, "session_90dbd78a");
        assert_eq!(modern_subagent.agent, "agent-7");
        assert_eq!(modern_subagent.workspace, "wd_renamer_77d4205a98d1");

        let legacy = kimi_scan_state(Path::new(
            "/h/.kimi/sessions/b7d87b9c610d01d676c088f03905388d/9ce64e58-5796/wire.jsonl",
        ));
        assert_eq!(legacy.session_id, "9ce64e58-5796");
        assert_eq!(legacy.agent, "");
        assert_eq!(legacy.workspace, "b7d87b9c610d01d676c088f03905388d");
    }

    #[test]
    fn dsh_usage_chunks_carry_the_request_header_model() {
        // Shapes captured from a live ~/.dsh session.jsonl (decompressed) on
        // 2026-09-17.
        let mut state = DshScanState::new();
        let session = r#"{"type":"session","version":0,"id":"session-cf88fef4-768c-490d-9dd9-6b72fa7e696c","createdAt":1787044214506,"cwd":"/Users/me/dev/waku","delegationDepth":0}"#;
        let header = r#"{"type":"request/header","seq":12,"time":1787044263136,"data":{"header":{"config":{"provider":"ollama-cloud","model":"gemma4:cloud","maxTokens":65536}}}}"#;
        let chunk = r#"{"type":"assistant/chunk","seq":15,"time":1787044263229,"data":{"turn":1,"step":1,"chunk":{"type":"usage","usage":{"inputTokens":11838,"outputTokens":80}}}}"#;

        // Usage before any request/header has no model to attribute.
        assert!(parse_dsh_line(chunk, &mut state).is_none());
        assert!(parse_dsh_line(session, &mut state).is_none());
        assert!(parse_dsh_line(header, &mut state).is_none());

        let record = parse_dsh_line(chunk, &mut state).expect("the model is known now");
        assert_eq!(record.provider, UsageProvider::DeepSeek);
        assert_eq!(record.model, "gemma4:cloud");
        assert_eq!(
            record.session_id,
            "session-cf88fef4-768c-490d-9dd9-6b72fa7e696c"
        );
        assert_eq!(record.project, "/Users/me/dev/waku");
        assert_eq!(record.timestamp_ms, 1_787_044_263_229);
        assert_eq!(record.totals.uncached_input, 11838);
        assert_eq!(record.totals.output, 80);
        assert_eq!(record.totals.total(), 11918);
        assert!(record.reported_cost_usd.is_none());
        assert_eq!(
            record.dedupe_key.as_deref(),
            Some("dsh:session-cf88fef4-768c-490d-9dd9-6b72fa7e696c:15")
        );

        // The zero-usage chunks title-generation requests emit parse to nothing.
        assert!(parse_dsh_line(
            r#"{"type":"assistant/chunk","seq":16,"time":1787044263300,"data":{"chunk":{"type":"usage","usage":{"inputTokens":0,"outputTokens":0}}}}"#,
            &mut state
        )
        .is_none());
    }

    #[test]
    fn kimi_project_map_resolves_workspace_and_session_names() {
        // The directory name the older store derives, verified against a live tree.
        assert_eq!(
            md5_hex("/Users/jiyuliang/工作/DSP-Modeling/Qi-Pure"),
            "b7d87b9c610d01d676c088f03905388d"
        );

        let dir = std::env::temp_dir().join(format!("waku-kimi-map-{}", std::process::id()));
        let code_home = dir.join(".kimi-code");
        std::fs::create_dir_all(dir.join(".kimi")).unwrap();
        std::fs::create_dir_all(&code_home).unwrap();
        std::fs::write(
            dir.join(".kimi/kimi.json"),
            r#"{"work_dirs": [{"path": "/Users/jiyuliang/工作/DSP-Modeling/Qi-Pure", "kaos": "local"}, {"path": "/", "kaos": "local"}]}"#,
        )
        .unwrap();
        std::fs::write(
            code_home.join("workspaces.json"),
            r#"{"version":1,"workspaces":{"wd_renamer_77d4205a98d1":{"root":"/Users/jiyuliang/个人/Renamer"}}}"#,
        )
        .unwrap();
        std::fs::write(
            code_home.join("session_index.jsonl"),
            concat!(
                r#"{"sessionId":"ses_3155979c-67f3-4b66-90ca-a0bb7c6d0f0f","sessionDir":"/x","workDir":"/Users/jiyuliang/工作/DSP-Modeling/qi-pure"}"#,
                "\n"
            ),
        )
        .unwrap();

        let map = kimi_project_map_in(
            Some(&dir),
            &code_home,
            &[PathBuf::from("/Users/me/dev/waku")],
        );
        // The modern store resolves the workspace directory name directly.
        assert_eq!(
            map.get("wd_renamer_77d4205a98d1").map(String::as_str),
            Some("/Users/jiyuliang/个人/Renamer")
        );
        // Migrated sessions resolve by their own name, not the workspace's.
        assert_eq!(
            map.get("ses_3155979c-67f3-4b66-90ca-a0bb7c6d0f0f")
                .map(String::as_str),
            Some("/Users/jiyuliang/工作/DSP-Modeling/qi-pure")
        );
        // The older store still resolves through its MD5 directory names, and
        // the app's own roots are included so a launch from one maps too.
        assert_eq!(
            map.get("b7d87b9c610d01d676c088f03905388d")
                .map(String::as_str),
            Some("/Users/jiyuliang/工作/DSP-Modeling/Qi-Pure")
        );
        assert_eq!(
            map.get(&md5_hex("/Users/me/dev/waku")).map(String::as_str),
            Some("/Users/me/dev/waku")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
