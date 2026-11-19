import os, sys, json, random, types
os.environ["GRACETIER_STAGE_LOG"] = os.path.abspath("out/k4-test.jsonl")
os.environ["GRACETIER_STAGE_LOG_PICKS"] = "1"
os.makedirs("out", exist_ok=True)
if os.path.exists(os.environ["GRACETIER_STAGE_LOG"]): os.remove(os.environ["GRACETIER_STAGE_LOG"])
sys.path.insert(0, "c/tree/fastpath"); sys.path.insert(0, "faketorch")
try:
    import torch
except ImportError:
    print("NO_TORCH"); sys.exit(3)
torch.cuda.is_current_stream_capturing = lambda: False
import fastpath_gracetier as G
random.seed(1)
L, E, H = 4, 384, 275
C = E - H
class St: pass
G._S.res_cnt = torch.zeros(L, E, dtype=torch.int32)
truth = []
for l in range(L):
    st = St(); st.idx = l; st.H = H; st.C = C; st.cold = list(range(H, E)); st.cnt = torch.zeros(C, dtype=torch.int32)
    G._S.states[l] = st
class Sched:
    total_num_scheduled_tokens = 6
    num_scheduled_tokens = {"r0": 6}
# residency mirror stub: slot c of layer 1 holds a different expert after "swap"
m = types.SimpleNamespace(stats={"swaps": 0}, state=types.SimpleNamespace(cold={l: list(range(H, E)) for l in range(L)}))
G._S.res = m
N = 300
for s in range(N + 1):
    G._stage_log_tick(Sched())          # logs step s-1
    if s == N: break
    if s == 100:
        m.state.cold[1][0] = 7; m.stats["swaps"] = 1     # a swap at barrier 100: slot 0 of layer 1 now holds expert 7
    step = {}
    for l in range(L):
        k = random.choice([0, 1, 1, 2, 3])
        base = [0, 1, 2] if random.random() < 0.6 else random.sample(range(C), 3)
        slots = sorted(set(base[:k]))
        st = G._S.states[l]
        tr = []
        for c in slots:
            n = random.randint(1, 3)
            st.cnt[c] += n
            e = m.state.cold[l][c]
            G._S.res_cnt[l, e] += n
            tr += [e, c, n]
        G._S.res_cnt[l, 5] += 36 - sum(tr[2::3])        # hot picks
        if tr: step[str(l)] = tr
    truth.append(step)
G._S.slog["f"].flush()
lines = [json.loads(x) for x in open(os.environ["GRACETIER_STAGE_LOG"])]
assert lines[0]["event"] == "open" and lines[0]["picks"] is True, lines[0]
recs = lines[1:]
assert len(recs) == N, len(recs)     # the baseline read precedes step 0; every step is logged at the next tick
for r, want in zip(recs, truth):
    assert r["cold"] == want, (r["s"], r["cold"], want)
    for l, flat in r["picks"].items():
        assert sum(flat[1::2]) == 36, flat
assert recs[99]["swaps"] == 0 and recs[100]["swaps"] == 1, (recs[99]["swaps"], recs[100]["swaps"])
assert any(7 in r["cold"].get("1", [])[0::3] for r in recs[100:]), "swapped expert id never logged"
print("TICK_OK", len(recs), "records; first:", json.dumps(recs[0])[:200])
