# Mutation testing localstub

## Reproducing the scan

Use `uv sync --all-extras --dev`, then:

```sh
uv run poe test
uv run poe mutate
uv run poe mutate-decorated
```

The full source scan includes every module under `src/localstub/`, with the
entire `tests/` suite available for test selection. It does not filter to
coverage.py-covered lines or impose a maximum stack depth. Four worker
processes bound concurrency. Coverage collection is disabled only inside
mutation runs; `poe test` is unchanged. These tasks are deliberately not CI
gates: surviving mutations require interpretation.

### Dataclasses

Released mutmut 3.7.0 skips decorated classes. The uv source/lock pin uses
[the upstream dataclass fix](https://github.com/boxed/mutmut/commit/dc58270d5234752e22247f814431750eafcf6e5f),
not a moving branch or a locally edited dependency. The package still labels
itself 3.7.0. Replace the pin with a released version once that release
contains the fix. Installing without uv's source overrides would lose it.

### Properties and other decorators

Even with the upstream fix, mutmut excludes properties and generally
decorated functions. `scripts/mutate_decorated.py` makes a disposable copy of
source and tests, gives decorated bodies unique undecorated helper names,
then applies the original descriptors/decorators to those helpers. Getter
and setter bindings are preserved. Only the helper mutations are selected;
the ordinary scan's mutants are not counted a second time.

The script supports this repository's single-decorator forms. It retains
classmethods, staticmethods, and overload declarations for mutmut to handle.
It is not a general-purpose Python decorator rewriter: generated names and
line numbers differ, and decorator expressions, generated dataclass methods,
and class/module-level defaults are not themselves mutation targets.

The copied source passes the full suite without mutations. Dedicated tests
also verify property getter/setter binding, classmethods, cached functions,
async context managers, source isolation, and cache invalidation.

Two cache pitfalls matter:

- **Inherited runtime caches:** mutation workers fork after baseline tests
  have populated `lru_cache`. The copied module registers an after-fork
  cache clear, so workers execute changed bodies rather than reuse baseline
  results. Cache behavior is otherwise retained, including in the parent.
- **Stale test relevance:** renamed helpers can otherwise reuse old
  test-to-function mappings and appear to have no tests. The supplemental
  run recreates its mutation cache each time. Its latest results remain in
  `mutants-decorated/mutants/` until the next run.

### Inspecting and refreshing results

```sh
# Primary scan: cached results, individual diffs, interactive browser.
uv run mutmut results
uv run mutmut show 'localstub.http.responsespec.xǁHTTPResponseǁraw__mutmut_1'
uv run mutmut browse

# Supplemental scan: separate workspace and result cache.
uv run python scripts/mutate_decorated.py results
uv run python scripts/mutate_decorated.py show \
  'localstub.router.xǁRouterǁmutation_body_has_routes_28__mutmut_2'
uv run python scripts/mutate_decorated.py browse

# Rerun all mutations in changed modules, including cached survivors.
uv run poe mutate 'localstub.http.responsespec.*' 'localstub.router.*'
```

Helper names include original source line numbers. Use `results` to find
current IDs; `mutants-decorated/decorated-bodies.json` maps them back to
original callables. Supplemental diffs refer to the generated copy, not the
distributed library. Never apply them to `src/`.

New test node IDs are discovered automatically. If existing tests change
which functions they call, or source functions are renamed, remove only
`mutants/mutmut-stats.json` to rebuild relevance, then explicitly rerun the
affected mutants/modules. This preserves unrelated `.meta` verdicts. An
unfiltered cached run alone does not recheck all old survivors or "no tests"
results after test changes.

## Results (2026-09-07)

Executed on Linux with CPython 3.14.7. The full primary scan completed all
5,622 targets. After adding regression assertions, all 210 mutations in
`http.headers`, `http.responsespec`, and `router` were explicitly rerun with
fresh test-relevance data. Unrelated primary verdicts below retain the
original full run's results. The supplemental scan was rerun from scratch
against the final tests.

| Run | Total | Killed | Survived | Timeout | No tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial full primary scan | 5,622 | 3,939 | 1,526 | 79 | 78 |
| Primary after targeted reruns | 5,622 | 3,942 | 1,523 | 79 | 78 |
| Final supplemental scan | 109 | 91 | 18 | 0 | 0 |
| Combined latest verdicts | **5,731** | **4,033** | **1,541** | **79** | **78** |

There are **zero unchecked targets**, skipped mutants, suspicious outcomes,
or reported segmentation faults. "No tests" targets were accounted for,
not executed. Timeouts are not counted as assertion-based kills.

Verification:

- `uv run poe auto-check`: passed (ruff, pyright, ast-grep, import contracts).
- `uv run poe test`: **704 passed**, 93% overall coverage.
- Full pytest suite against the non-mutated transformed copy: **704 passed**.
- `uv run pyright scripts/mutate_decorated.py`: zero errors or warnings.
- `uv lock --check` and `git diff --check`: passed.
- The three newly caught primary mutants report `killed`, as does the
  supplemental `Router.has_routes` mutant. All seven `endpoint_url` mutants
  now have tests; three are killed and four survive.

## Coverage inventory

The inventory covers **87 classes and 447 module-level function/class-method
declarations**. Overload declarations are counted as declarations, not as
duplicate mutations of their implementation. Nested closures are attributed
to their enclosing callable by mutmut.

The primary pass generates **5,622 mutants**; the supplemental pass adds
**109**, for **5,731 distinct targets**. Representative class counts:

| Class | Primary | Supplemental |
| --- | ---: | ---: |
| HTTPResponse | 68 | 0 |
| Headers | 84 | 1 |
| Router | 14 | 4 |
| HTTPRequest | 12 | 24 |
| RecordedHTTPRequest | 67 | 11 |
| ConnectionPool | 240 | 0 |
| RawForwarder | 546 | 0 |
| AsyncHTTPTestServer | 713 | 9 |
| AsyncTLSInterceptProxy | 740 | 14 |
| TrafficRecorder | 103 | 0 |
| TokenBucket | 50 | 0 |

343 declarations have at least one mutant across the two passes. The other
104 are explicitly accounted for: 34 protocol/stub/abstract declarations,
6 typing overload declarations, and 64 bodies with no applicable mutmut
operator. Those 64 include simple forwarding accessors and wrappers.
For example, `RecordedHTTPResponse` has three forwarding properties and
still generates zero mutants; it is **not** evidence of perfect testing.
Data-only classes and generated dataclass methods similarly cannot be
claimed as mutation-tested. No production-code exclusions or suppression
comments were added to improve the numbers.

## Confirmed test gaps

The initial small experiment caught missing assertions for iterable response
headers (including duplicates) and case-insensitive default-header overrides.
The class-inclusive scan additionally identified:

- `HTTPResponse.raw()` changing its default status from 200 to 201.
- `Headers.__getitem__()` losing the missing key in `KeyError.args`.
- `Router.handle()` passing `None` instead of the caller's context.
- `Router.has_routes` reporting true for a fresh router with no routes.
- `AsyncTLSInterceptProxy.endpoint_url` having no tests at all: it now has
  coverage for the pre-start error and the URL of its bound listener.

Regression assertions cover these public behaviors. Runtime library source
is unchanged. Other inspected survivors should not be chased blindly:
codec capitalization and `typing.cast` type arguments are runtime-equivalent;
a constant hash is legal under Python's hash contract; header-name casing
and exact diagnostic strings are not automatically meaningful regressions.

### Remaining test-relevance gaps

The primary scan's 78 "no tests" results are concentrated in ten callables:

| Callable | Mutants without relevant tests |
| --- | ---: |
| `cli.main` | 3 |
| `cli.run_proxy` | 22 |
| `AsyncRequestParser._body_parse_error` | 20 |
| `AsyncMultiResponseParser._body_parse_error` | 15 |
| `server._default_throttle_key` | 2 |
| `AsyncHTTPTestServer.set_request_headers_handler` | 1 |
| `AsyncHTTPTestServer.use_headers` | 1 |
| `RecordingStreamWriter.get_extra_info` | 4 |
| `RecordingStreamWriter.write_eof` | 9 |
| `RecordingStreamWriter.writelines` | 1 |

These are follow-up candidates, not suppressed results. Before adding tests,
distinguish genuinely unexercised behavior from a test-tracking limitation;
test the behavior through supported public APIs rather than importing private
helpers just to improve a score.

## Interpreting the results

- **Killed:** selected tests failed under the mutation.
- **Survived:** selected tests passed; may be equivalent, an unasserted
  behavior, or a test/tool interaction. Not automatically a real bug.
- **No tests:** mutmut found no relevant test; no mutant test execution
  happened. This is a coverage gap, not a passing result.
- **Timeout:** the mutation exceeded the tool's budget. Keep this separate
  from assertion-based kills; timeout alone does not diagnose a library bug.
- **Not checked:** outstanding work, not a final scan result.

The exhaustive inventory/result exports accompany the scan. They distinguish
complete enumeration from manual triage: selected core-API survivors were
reviewed, not every surviving mutation. A high ordinary line/branch coverage
percentage does not imply a correspondingly high mutation score.
