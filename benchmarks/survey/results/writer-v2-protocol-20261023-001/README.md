# Full-input writer protocol observation

This is a protocol diagnostic, not a Survey quality pass.

- Source: `bcc12ed`, `grounded-writer-v1`, full public few-shot inputs.
- Model: `doubao-seed-2-1-pro-260915`; request messages, schema and sampling parameters match the preceding failed request.
- HTTP 200 / stop; valid JSON and local writer schema; five body sections, 66 native units.
- The provider HTTP content and SDK content match exactly.
- New usage: one request, 14,578 known tokens; no new unknown usage or retries.
- Reviewer, Evaluation, Judge and Jev were not run. Scientific score is unavailable and publishable is false.
- The earlier identical request remains a failure (one request / 12,477 tokens). These two observations do not establish a cause or reliable pass rate.

The bounded allowance was 120,000 tokens / one request / 20,000 output tokens. Full inputs, HTTP bodies and private records remain local. [Structured result](result.json).
