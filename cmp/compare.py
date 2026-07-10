#!/usr/bin/env python3
"""3-way diarization comparison: sherpa vs pyannote 3.x vs pyannote 4.x (community-1).
speaking_time_seconds = diarizer-turn time (whisper-independent, fair).
segments[]           = whisper segments w/ speaker label (per-run), used for agreement.
Agreement = optimal-matched co-speech / total overlap (DER-complement), per pair."""
import json
from itertools import permutations


def load(p):
    return json.load(open(p))


def st(d):
    return {int(k): float(v) for k, v in d.get("speaking_time_seconds", {}).items()}


def tl(d):
    return [(s["start"], s["end"], s["speaker"]) for s in d["segments"]]


def sw(t):
    n = 0
    last = None
    for _, _, s in t:
        if s != last:
            n += 1
            last = s
    return n


def overlap(tl_a, tl_b):
    sa = sorted({s for _, _, s in tl_a})
    sb = sorted({s for _, _, s in tl_b})
    M = {i: {j: 0.0 for j in sb} for i in sa}
    tot = 0.0
    for as_, ae, asp in tl_a:
        for bs, be, bsp in tl_b:
            ov = max(0.0, min(ae, be) - max(as_, bs))
            if ov > 0:
                M[asp][bsp] += ov
                tot += ov
    return M, tot, sa, sb


def best_match(M, sa, sb):
    best, bp = -1.0, None
    for p in permutations(sb, len(sa)):
        s = sum(M[a][p[i]] for i, a in enumerate(sa))
        if s > best:
            best, bp = s, p
    return dict(zip(sa, bp)), best


def agree(pa, pb):
    da, db = load(pa), load(pb)
    M, tot, sa, sb = overlap(tl(da), tl(db))
    m, matched = best_match(M, sa, sb)
    return (100.0 * matched / tot) if tot else 0.0, m


def hms(x):
    return f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"


def file_report(name, paths):
    # paths: dict label->path  (sherpa, pyannote3, pyannote4)
    data = {k: load(v) for k, v in paths.items()}
    sts = {k: st(d) for k, d in data.items()}
    tls = {k: tl(d) for k, d in data.items()}
    order = ["sherpa", "pyannote3", "pyannote4"]

    print(f"\n{'#'*74}\n# {name}\n{'#'*74}")

    # granularity: speaker-switches on the whisper segments (more = finer turns)
    print("  speaker-switches on whisper segments (granularity proxy):")
    print("     " + "   ".join(f"{k}={sw(tls[k])}" for k in order))

    # speaking time, speakers sorted by time desc (so 'rank 1' = most talkative)
    print("\n  speaking time (sec), each diarizer's speakers ranked most->least talkative:")
    print(f"     {'rank':>5}  " + "  ".join(f"{k:>20}" for k in order))
    for rank in range(3):
        cells = []
        for k in order:
            ranked = sorted(sts[k].items(), key=lambda x: -x[1])
            if rank < len(ranked):
                sp, secs = ranked[rank]
                cells.append(f"SPK_{sp:02d}={secs:>4.0f}s")
            else:
                cells.append("-")
        print(f"     #{rank+1:<4}  " + "  ".join(f"{c:>20}" for c in cells))
    for k in order:
        print(f"     {k:>5} total = {sum(sts[k].values()):.0f}s")

    # pairwise agreement
    print("\n  pairwise agreement (optimal speaker match, % of co-speech):")
    pairs = [("sherpa", "pyannote3"), ("sherpa", "pyannote4"), ("pyannote3", "pyannote4")]
    for a, b in pairs:
        ag, _ = agree(paths[a], paths[b])
        print(f"     {a:>10} vs {b:<10}:  {ag:5.1f}%")

    # cross-map onto pyannote4 (reference) — shows where engines diverge
    ref = "pyannote4"
    print(f"\n  how each engine maps onto {ref} (community-1) speakers:")
    print(f"     {'p4 speaker':>11} {'p4 time':>9}   {'sherpa matched':>20}  {'pyannote3 matched':>20}")
    _, m_ref_sh = _match_dirs(tls[ref], tls["sherpa"])
    _, m_ref_p3 = _match_dirs(tls[ref], tls["pyannote3"])
    for sp in sorted(sts[ref]):
        sh_sp = m_ref_sh.get(sp, sp)
        p3_sp = m_ref_p3.get(sp, sp)
        print(f"     SPK_{sp:02d}        {sts[ref][sp]:>6.0f}s   "
              f"sherpa SPK_{sh_sp:02d}={sts['sherpa'].get(sh_sp,0):>4.0f}s   "
              f"p3 SPK_{p3_sp:02d}={sts['pyannote3'].get(p3_sp,0):>4.0f}s")


def _match_dirs(tl_ref, tl_other):
    M, tot, sa, sb = overlap(tl_ref, tl_other)
    m, matched = best_match(M, sa, sb)  # m: ref_speaker -> other_speaker
    return (100.0 * matched / tot) if tot else 0.0, m


if __name__ == "__main__":
    for label, base in [("LONG FILE (25 min)", "long"), ("SHORT FILE (3.5 min)", "short")]:
        file_report(label, {
            "sherpa": f"cmp/sherpa/{base}.json",
            "pyannote3": f"cmp/pyannote3/{base}.json",
            "pyannote4": f"cmp/pyannote4/{base}.json",
        })
