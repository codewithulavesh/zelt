"""Synthetic recorded trajectories, used by `earlystop selftest` to verify the whole
analysis pipeline without a GPU. NOT real results - never report these numbers."""

import random


def make_synthetic(n=600, seed=0, chunk=48):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        gold = str(rng.randint(1, 500))
        solvable = rng.random() < 0.8
        solve_t = int(rng.lognormvariate(5.2, 0.6))
        total = solve_t + int(rng.lognormvariate(5.6, 0.7))  # overthinking tail
        wrong_fixed = str(rng.randint(501, 999))
        probes, t = [], chunk
        while t <= total:
            right = solvable and t >= solve_t
            if right and rng.random() < 0.93:
                ans, good = gold, True
            elif (not right) and rng.random() < 0.12:
                ans, good = gold, True  # lucky early guess
            else:
                ans = (
                    wrong_fixed
                    if (not solvable and rng.random() < 0.7)
                    else str(rng.randint(501, 999))
                )
                good = False
            a, b = (8, 2) if good else (3, 3)
            gm = rng.betavariate(a, b)
            probes.append(
                dict(
                    t=t,
                    ans=ans,
                    gm=gm,
                    mean=min(1.0, gm + 0.03),
                    minp=max(0.0, gm - 0.15 * rng.random()),
                    ent=max(0.0, (1 - gm) * 3 + rng.gauss(0, 0.2)),
                    ptok=6,
                )
            )
            t += chunk
        full_ans = probes[-1]["ans"] if probes else None
        rows.append(
            dict(
                idx=i,
                gold=gold,
                full_tokens=total,
                full_ans=full_ans,
                full_correct=(full_ans == gold),
                probes=probes,
                hit_max=False,
            )
        )
    return rows
