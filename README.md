# LayaKit

A Swift package that runs one Laya typed decision per call on-device via Core ML — no Python at runtime. Ported from [`mizorewww/laya-coreml`](https://github.com/mizorewww/laya-coreml).

## Requirements

- macOS 15+
- Apple silicon
- Xcode 26+

## Install

Add LayaKit as a SwiftPM dependency:

```swift
.package(url: "https://github.com/<org>/LayaKit", from: "1.0.0")
```

```swift
.target(
    name: "YourTarget",
    dependencies: [.product(name: "LayaKit", package: "LayaKit")]
)
```

## Download bundles

```
pip install -U huggingface_hub
hf download aac6fef/laya-multilingual-coreml --local-dir models/general
hf download aac6fef/laya-multilingual-coreml-ane --local-dir models/ane
```

## Usage

```swift
let agent = try await LayaAgent(bundle: URL(fileURLWithPath: "models/ane"))
let answer = try agent.predict(state: "I was billed twice. Refund the duplicate.",
                               question: .choice(instructions: "Who should handle this?",
                                                 options: ["billing", "technical", "sales"]))
```

## Question types

| Case | Parameters | Description |
| --- | --- | --- |
| `.choice` | `instructions: String`, `options: [String]`, `descriptions: [String: String]?` | Pick one label out of a set of options, optionally with a description per option. |
| `.score` | `instructions: String`, `levels: [String]` | Rate along an ordered scale; each level is described by a string. |
| `.noul` | `instructions: String`, `falseText: String?`, `trueText: String?` | A yes/no judgment, optionally with custom text for the false/true cases. |

## CLI usage

`laya-cli` runs a single question against a bundle and prints the answer as JSON.

```
laya-cli --bundle <path> --state <text> --question <path-to-json> [--bench N]
```

- `--bundle`: path to a model bundle directory (e.g. `models/general` or `models/ane`).
- `--state`: the input text describing the situation.
- `--question`: path to a JSON file describing one question (see below).
- `--bench N`: after printing the answer, run `N` additional `predict` calls and print p50/p90/mean latency (ms) to stderr.

Question JSON shape:

```json
{"type": "choice", "instructions": "…", "criteria": ["billing", "technical", "sales"]}
{"type": "choice", "instructions": "…", "criteria": {"billing": "handles invoices", "technical": "handles bugs"}}
{"type": "score", "instructions": "…", "criteria": ["not urgent", "urgent", "critical"]}
{"type": "noul", "instructions": "…", "criteria": {"false": "no refund needed", "true": "refund required"}}
```

Example invocation:

```
laya-cli --bundle models/ane --state "I was billed twice. Refund the duplicate." --question examples/choice.json
```

See `examples/choice.json`, `examples/score.json`, and `examples/noul.json` for ready-to-use question files.

## Performance

Measured with `laya-cli --bench 100` on an Apple M5 Pro, macOS 27.0:

| Bundle | p50 latency | p90 latency | mean latency | Cold init |
| --- | --- | --- | --- | --- |
| `models/general` | 4.83 ms | 5.45 ms | 4.97 ms | 1.53 s |
| `models/ane` | 5.19 ms | 6.86 ms | 5.56 ms | 1.00 s |

The first `predict` call is excluded from these numbers because `LayaAgent.init` performs a warm-up call. `.mlmodelc` compilation is cached next to each bundle's `.mlpackage`, so cold-init time reflects a warm mlmodelc cache after the first launch.

## Limits

- The ANE bundle accepts at most 96 tokens; exceeding this throws `LayaError.tooManyTokens`.
- Choice questions accept at most 32 options; exceeding this throws `LayaError.tooManyOptions`.

## Testing

```
.venv/bin/python scripts/make_fixtures.py
swift test
```

Tests read model bundles from `LAYA_GENERAL_BUNDLE` and `LAYA_ANE_BUNDLE` environment variables, falling back to `models/general` and `models/ane`.

## License

Adapted from [`mizorewww/laya-coreml`](https://github.com/mizorewww/laya-coreml), licensed under Apache-2.0. See `reference/LICENSE` and `reference/NOTICE` for upstream attribution.
