# Fork history

This repository is a fork of [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory),
based on upstream `v0.9.29`. Everything below is the delta carried on top of that tag.

The fork exists to answer two problems that stock agentmemory has. The first is
that the graph retrieval leg reads the entire graph on every entity-bearing query, which takes
`memory_smart_search` down once the graph grows past a few hundred megabytes. The second is the
iii engine itself, which the fork replaces with an in-process runtime on `node:sqlite`. The second
change is what the repository is named for and is the larger of the two.

## Reading the delta

Upstream `v0.9.29` is the base. To see the whole fork as one diff, add upstream as a remote and
compare against the tag:

```
git remote add upstream https://github.com/rohitg00/agentmemory.git
git fetch upstream --tags
git diff v0.9.29..HEAD
```

That diff is 167 files, roughly +27,000 / -1,200 lines, of which the in-process runtime and its
tests are about two thirds.

This repository holds a single squashed commit. The commit hashes quoted below come from the
private working fork the patches were developed in, so they will not resolve here; they are kept
because they are how each change is referred to in the upstream discussion threads and because
they preserve the order the work actually happened in.

## 1. Graph retrieval taken out of the hot path

Stock issues `kv.list(KV.graphNodes)` and `kv.list(KV.graphEdges)` per entity-bearing query,
enumerating the whole graph. On a corpus whose graph reached around 330 MB this overran the engine
and `smart_search` returned HTTP 500. Upstream `v0.9.29` fixed the same enumeration bug on the
write path, in `persistGraphDelta()`, and left the reader alone, so the fault survives in current
stock.

| Commit | Change |
|---|---|
| `4a5677a` | Fail-closed graph read path plus the `AGENTMEMORY_GRAPH_LEG` kill-switch |
| `88d9545` | Serve graph reads from bounded side-indexes instead of full enumeration (upstream PR #893) |
| `e7f73d0` `c0a80d2` `482dcac` `49602e5` | Persist and resolve a real `sessionId` on graph nodes, degrade gracefully when the session list is unavailable (upstream PR #937) |
| `53e9e9c` | `findObservation` lookups that silently returned nothing, which is what breaks `expandIds` |

Two modes result. With `AGENTMEMORY_GRAPH_LEG=off` there are no graph-scope reads at all: graph
writes and maintenance are skipped, `smart_search` runs on BM25 plus vector and reports
`graphOmitted: true`, and graph, temporal and export queries return a typed
`GRAPH_INDEX_NOT_READY` rather than hanging. With the flag unset the graph leg runs through the
#893 side-indexes. The off mode is the one proven in production; arming the side-indexes
on an already-large corpus was not feasible, because the initial raw-graph enumeration needed to
build them exceeds the call timeout.

`53e9e9c` is a plain bug fix in `findObservation`, unrelated to the graph leg, and stock is
affected identically. Without it `expandIds` returns empty results with no error. It is a clean
upstream candidate, and it is distinct from upstream issue #440, which is the MCP client dropping
`expandIds` in its proxy.

Upstream PRs #893 and #937 were both still open at the time of the `v0.9.29` merge. If upstream
lands them, resolve toward upstream and drop the copies here.

## 2. Retrieval correctness

Saved memories were reachable through `memory_recall` but did not surface through
`memory_smart_search`. The cause was not an indexing gap, since the documents were present in both
indexes. Two defects in ranking were responsible, and a set of indexing fixes followed from
investigating them.

The primary defect was that the hybrid search legs were fetched at `limit * 2`. A document strong
in BM25 but deep in the vector leg fell outside the vector fetch at small page sizes and forfeited
its vector term entirely. Under reciprocal rank fusion that is unrecoverable, because the vector
term is worth up to `0.6/61` while the whole BM25 term tops out at `0.4/61`, so any document
present in both legs beats any document present in one. Curated memories are exactly the long,
diffuse documents that sit deep in the vector leg. The symptom was a monotonicity violation rather
than a preference: a memory absent from the top 15 ranked fifth at limit 60.

The secondary defect was that prefix expansion summed without bound, so every distinct index term
sharing a query token's prefix added its own full BM25 contribution. On a 200-document corpus the
query `v1` scored 3.63 with one sibling term and 110.01 with eighty, against a ceiling of about 22
for any single term. Combining siblings with `max` instead, the usual dis_max treatment, is
bounded by construction and leaves exact-term saturation untouched.

| Commit | Change |
|---|---|
| `3412b2f` | Fetch each leg to a depth floor rather than a multiple of `limit` |
| `513696b` | Bound prefix expansion by best sibling instead of the sum |
| `8fe5cb0` | Remove before re-add in the search index so `avgDocLen` stays correct |
| `a169b3c` | Chunk-aware remove in the vector index, ungated parent collapse |
| `75455da` | Make vector-index `add()` idempotent against stale chunk vectors |
| `cbb17eb` | Section-aware chunker and memory indexing projections |
| `82d6016` | Tunable fusion, session-cap exemption, layer boost in hybrid search |
| `4ffe1b9` | Persist `toolName`, honour index exclusion at the add sites |
| `84864c8` | Bounded-RAM index rebuild, echo exclusion, rebuild token |
| `84fd940` | Rebuild at boot when the vector index is empty but a provider exists |
| `a738a2f` | De-index superseded versions, batch chunked vector writes |
| `190e9d7` | Flush the live index periodically, not only at boot |
| `eabad8a` | Non-latest suppression at both read paths, flag-disabled |
| `c1d87db` `837b066` `3f5be07` | Baseline reconcile with a retained-predecessor checkpoint, catch-up pass, duplicate accounting |
| `08d5d47` | Never supersede or demote across distinct dates |
| `0f513c7` `046a4fd` | Wire query expansion into smart-search behind a flag, capture the merged result rather than the legs |
| `a9d29d3` | Gate the replace-import graph wipe the same way the restore is gated |
| `fc79097` | Probe capture points and an audit-noise rule |

`8fe5cb0`, `84fd940` and `a738a2f` are plain bug fixes with no flag at all, and are clean upstream
candidates alongside `53e9e9c`.

Two flags exist that production deliberately leaves off, because measurement contradicted the
design note that proposed them. `AGENTMEMORY_MEMORY_DOC_SLIM` was meant to raise memory scores by
dropping duplicated content, on the theory that the inflated `docLen` was halving the score
through length normalisation. It does the opposite: duplicating content scales `tf` and `docLen`
by the same factor and BM25 saturates in `tf`, so the fat document scores higher. Probed on a
22,600-document synthetic corpus, the memory-versus-echo score ratio was 0.258 fat against 0.205
slim. `test/remember-bm25-index.test.ts` encodes the measured direction so nobody enables the flag
expecting a win. `AGENTMEMORY_MAX_PER_SESSION=5` was meant to give memories more slots, but
`AGENTMEMORY_DIVERSITY_EXEMPT_MEMORY_SESSION` does that directly, and raising the cap on top of it
only lets one chatty session take more of the page.

All flags are documented in `.env.example`.

## 3. In-process runtime on `node:sqlite`, replacing the iii engine

This is the largest change in the fork, and the one the repository is named after. The iii
engine's responsibilities move in-process behind the interfaces the rest of the codebase already
used, so call sites are unchanged.

State lives in SQLite through five functions registered on an in-process shim, `state::get`,
`state::set`, `state::update`, `state::delete` and `state::list`, backed by `DatabaseSync` from
`node:sqlite`. The existing `StateKV` class in `src/state/kv.ts` is untouched. The schema keeps a
monotonic `seq` per row with a unique constraint on `(scope, key)` and an index on `(scope, seq)`.
`set` on an existing key has to update in place and keep its position, because the supersession
scan in `remember.ts` and BM25 tie ordering both depend on `list(scope)` returning insertion
order. `update` runs read-modify-write inside one transaction and rejects op shapes it does not
implement rather than silently dropping them.

Vectors move into the same database behind `VectorIndex`. Nothing persists the BM25 index. The
daemon rebuilds it from content at boot and fills the vector index on a schedule afterwards, and a
readiness gate on the shim stops anything serving against a half-built index.

| Commit | Change |
|---|---|
| `7878e32` | SQLite-backed `state::` functions behind `StateKV` |
| `f11bb28` | In-process SDK shim replacing the iii engine |
| `099ee58` | Vector rows in SQLite behind `VectorIndex` |
| `1752426` | BM25 rebuilt from content at boot, vector fill afterwards |
| `51d6234` | Readiness gate on the shim |
| `67c3bfc` | Rebuild survives bad rows |
| `49de938` `d6094bf` `e4a650d` `ca71b15` | State-store importer: migrate a legacy iii store into SQLite, verify before publishing, keep the replaced database until the new one is proven |
| `d2a7138` `77e6353` `0a2a2f0` `c422bbc` `7291cfa` | Cutover support: maintenance routes, acceptance legs, rollback on every exit, fail-closed gates |
| `467dfe8` through `4cf9b61` | Acceptance harness: judge each result list against its own raw leg and explain differences by leg membership, rather than by a score tolerance |
| `c55e2bb` | Memoize the stemmer; `AGENTMEMORY_LOG_LEVEL` and `AGENTMEMORY_AUDIT_LOG` |
| `1153558` | Stop the consolidation decay tier from blocking the event loop |

The importer in `src/ops/import-state-store.ts` reads a legacy iii state store and writes the
SQLite database, cross-checking row counts and structure before publishing the result. It is the
migration path for an existing iii installation.

Tokenize and stem accounted for 16.2 s of a 20.4 s boot rebuild over 44,631 documents and 8.77M
tokens; memoizing `stem()` cut that to 2.55 s of 4.65 s. The memo is bounded: words over 64
characters are stemmed uncached, cached keys are copied so a token split out of a large query
cannot pin the query text, and the map is cleared at 500,000 entries.

The consolidation pipeline was the one real regression the move introduced, and `1153558` is the
fix. Its decay tier looped `await kv.set(...)` over every semantic row. Under iii those were RPCs
that yielded; in-process they became 45,000 synchronous transactions with an fsync each, on the
event loop with no yield, blocking every request for 95 to 100 s per run. The fix restricts decay
writes to changed rows in bounded transactions, adds a yield guard in the state wrappers, batches
the daily sweeps' fan-outs, and puts a single-flight guard on the pipeline.

## 4. Compression and health

A compression failure could orphan an observation, a defect that went unnoticed for about nine
months. The compression prompt was also destroying its own input. It computed budgets wrongly and
cut text from the end rather than the middle.

| Commit | Change |
|---|---|
| `53e9ef5` | Never orphan an observation when compression fails |
| `0ad646a` | Observations integrity check in diagnostics |
| `83fcfba` `de63baa` | `POST /agentmemory/recompress-orphans`, with a `requeueQuarantined` flag |
| `34e4a2f` | Stop destroying prompt text: fence the input, real budgets, middle-out cuts |
| `dfea325` `1d656b6` | Retry failed validations with a JSON-neutralized payload, reclaiming the last-instruction position |
| `e667b72` `dce0bca` | Alert on `heapUsed` against `heap_size_limit` rather than `heapTotal`, in the monitor and the viewer gauge alike |

## 5. Consolidation on session stop

`d9a953c` backports upstream PR #1136. `event::session::stopped` fires `mem::consolidate-pipeline`
and `mem::auto-crystallize`, debounced to at most once per `AGENTMEMORY_CONSOLIDATION_COOLDOWN_MS`
(default 300000). It removes the client-side `CONSOLIDATION_ENABLED` blocks from
`src/hooks/session-end.ts` and `plugin/scripts/session-end.mjs`, so the stop lifecycle is the
single driver and client and server cannot double-fire. Eviction's stale-session recovery passes
`skipConsolidation: true` and runs one corpus-wide pass afterwards, so N recovered sessions no
longer launch N full-corpus consolidations.

This is live behaviour rather than an inert flag, and the session-stop path can fire far more
often than the 120-minute timer it supplements. Read it together with `1153558` above before
enabling it on a large corpus.

## 6. Provider

`45c5762` adds `ANTHROPIC_EFFORT`, mapped to the API's `output_config.effort`, mirroring the shape
`src/providers/openai.ts` has always had for `OPENAI_REASONING_EFFORT`. Unset changes nothing, so
this is a clean upstream candidate.

It exists because adaptive thinking is on by default on newer Anthropic models, and omitting
`thinking` is not the same as disabling it. That interacts badly with two things already in the
codebase. Consolidation races `provider.compress()` against a hard 30 s timeout and silently
continues past the whole concept group on a timeout. And `AnthropicProvider.call()` returns an
empty string when a response carries no text block, which would write an empty memory with no
throw and no log.

Measured against real prompts, the default effort took a hard consolidation from a 4.5x margin
against that timeout down to 1.7x. `medium` restored it to 2.7x, cut output tokens by 31%, and was
also the fastest option on the query-expansion hot path.

Effort levels are model-specific, so check `capabilities.effort` on `GET /v1/models/{id}` rather
than assuming. Some models accept every level, some reject the highest, and some report no support
at all. On a model that reports no support, setting the variable will fail every compress call.

## 7. Repository and build changes

`2ab496c` migrates to `@huggingface/transformers` v4.

`d3960e0` tracks `package-lock.json`, overriding upstream's ignore rule. The build has to be
reproducible from this repository alone; without the lockfile `git archive` drops it and `npm ci`
either fails or silently resolves a different dependency tree than the one that produced the
shipped `dist`.

`0946a83`, `ab8ed11` and `bfc515c` add `.gitattributes`, pinning shell scripts, systemd units, the
generated skill references and the tracked plugin build outputs to LF. Without it a Windows
checkout with `core.autocrlf=true` hands CRLF to bash, which fails with
`$'\r': command not found`, and makes the byte-comparing `skills:check` report drift on every
file.

`ce2760e` and `889d615` remove the GitHub Actions workflows (`ci.yml`, `publish.yml`) and
`dependabot.yml`. Publishing to npm is not something a fork should do, and CI moved to a local
pre-push hook to avoid Actions minutes on a private repository. Actions are also disabled at the
repository level, so nothing can consume minutes even if a workflow file reappears. All checks are
local. `.githooks/pre-push` runs `npm run build` and `npm run skills:check`. Activate it per clone
with:

```
git config core.hooksPath .githooks
```
