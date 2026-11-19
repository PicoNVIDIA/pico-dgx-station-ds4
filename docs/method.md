# Method

## The benchmark

Speed and accuracy come from `bench_full.py` in Garner McCloud's repo, the benchmark behind his published two-GPU numbers. One server session runs every stage.

| Stage | What it sends | Are the prompts unique? |
|---|---|---|
| Fixed code prompt | One user. One 2.6K-token prompt, 2,048 tokens out, greedy, thinking off. One warm-up run, then 5 timed; the median is reported | No. The same prompt each run, with a random first line so the prefix cache cannot reuse it |
| Coding prompt | One user. One source file with a request for tests, exactly 2,048 tokens out, greedy. One warm-up run, then 5 timed | No. The same file each run, with a random run id first |
| GSM8K windows | 1, 2, 4, 8, 16 and 32 users for 60 seconds after a 20-second warm-up. Sampled at T=1.0, top-p 0.95, thinking on, up to 4,096 tokens out | 200 distinct questions. Each user starts at a random one and walks the list, so questions recur within a window and across runs |
| Prefill | Prompts of 14K, 57K and 114K tokens, one token out | Yes. Random filler and a cache salt per request |
| Capacity | 16 prompts of 95K tokens decoding 2,500 tokens at once, plus one more request. The benchmark labels them 110K | Yes, 16 distinct contexts |
| Quality | GSM8K-200 at T=0.3, three needles in a 174K-token prompt (labelled 200K), one tool call | The same 200 questions as the windows |

James Meadlock's two prompt sets (20 distinct prompts each, greedy, thinking off, 256 tokens out) give the mixed-prompt speed and the step time for one user. The same 40 prompts are used for the output-identity check.

All speeds are output tokens per second. Thinking tokens count as output.

## What the numbers are, and are not

- **Settled, not first-contact.** The engine counts expert picks and moves experts toward the traffic. Every boot is warmed with two rounds of both prompt sets, and the same sets are then timed. On the code prompt the five timed runs climb (489, 565, 522, 565, 611 on the default profile). A new prompt reads lower until the placement has adapted. One test measures that (`ho1m`: a boot with no warm-up rounds, set A three times, then set B). Set B, which the server had not seen, ran at 317 tok/s on first contact and 333 to 335 once settled. Set A, on the freshly booted server, ran at 284 and then 334 to 340.
- **Ranges are few samples.** A range in a table is the lowest and highest value across boots or passes of one config, usually two or three. Single runs spread wider than that.
- **Repeatability we saw.** Stream windows: of 85 pairs of passes taken in one boot, 70 agree within 3%; the largest gap is 8.7% and the large ones are all at 8 users, where the first pass after a boot reads low. In the three profile boots the largest gap is 2.5%. The mixed prompt sets: of 60 pairs, 57 agree within 1%; the largest gaps are 4.5% (the default profile's second set), 3.1% and 2.4%. The code prompt's session median ranged from 545.6 to 580.9 across boots of the 262K config.
- **Order matters.** The code prompt reads higher as the first stage of a session than after the prompt sets. The headline comparison with the public recipe uses the same order for both.

## Rules we held to

- One boot at a time. There is one GPU, so every config is its own boot, in series.
- Every boot is sent one real request before anything is timed.
- A candidate is compared with a baseline taken the same way where we had one. The README says where a comparison is not like for like.
- Other engines get the same prompts and stages. SGLang's token counter only moves when a request finishes, so its windows are counted at the client (`harness/sglang/bench_sgl.py`). The same client count on this recipe's engine matched the server counter to 0.1 tok/s (`results/logs/gtq15-run.log`).
- Cells that failed to boot stay in the logs.

## Output identity

`harness/identity/ident_capture.py` sends the 40 prompts greedy, one at a time, and stores the tokens with their log-probabilities. `ident_cmp.py` compares two captures and `ident_diag.py` describes where they split. Captures are in `results/identity/captures/`.

- At one user the engine repeats itself exactly: across runs, across hot-expert counts and after the manager has moved experts. So every one-user change in this repo is held to 40 of 40 identical against the reference capture (the public recipe at 275 experts).
- With several requests in a batch the unmodified engine does not match its own one-at-a-time output: 5 of 40 when the prompts are sent 8 at a time, 3 of 40 at 32. Many-user configs are therefore judged on the quality stage, not on token identity.
- Against stock vLLM v0.32.0, 3 of 40 answers have the same text. See `results/identity/stock-vs-engine.txt`. Stock vLLM matches itself on 40 of 40.

## Kernel patches

What each patch got, in the order it was done:

| Step | Stage skip (adopted) | Packed scale banks (experimental) |
|---|---|---|
| Patch applied and switched off: same output | Yes (`sk0_1m`): 40 of 40 | Yes, as its own control boot (`k7ctlA`), with the same kernel build as before |
| Edited kernel on the old data path | not applicable | Yes (`k7ctlB`) |
| Audit boot: every use checked byte for byte | Yes: 217,623 skips, 0 differing words (`k6a`) | Yes, on three layers: 4.07 billion output words, 0 differing (`k7audit`) |
| A control that must fail | No | Yes: with the packing order flipped, 94% of words differ (`k7auditx`) |
| Speed runs only after a gate script read the log | Yes (`gtq7-gate.log`) | Yes (`gtq19-gate.log`) |

No speed is read from an audit boot.

## What "measured once" means here

One machine, two nights, one operator, mostly one session per cell. Nothing has been re-run on a second Station.
