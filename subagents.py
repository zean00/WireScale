"""S-series: sub-agent routing on raw-embedding centroids (protocol.md,
S-series — registered pre-data; LAB-ONLY curiosity).

Shape: 8 domains x 8 agents = 64 sub-agents. Agent utterances carry the
agent's specialty pair with P(m), else domain-flavored generic phrases
(domain visible, agent NOT nameable). Readout is the production
mechanism: slot-7 raw embeddings, L2'd, centroid cosine — no trained
heads. Arms: A_flat (64-way argmax), B_walk (greedy domain->agent),
B_oracle (gold domain), E_beam2 (domain top-2, mean path cosine picks
— mean, never sum: the router-walk length-bias lesson).

    python subagents.py --smoke   # tiny counts, machinery check
    python subagents.py           # registered run, m in {1.0, 0.5, 0.0}
"""
import argparse
import json
import os

import numpy as np

import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PREFIXES = ["hey", "ok", "please", "could you", "i need you to", "can you"]

DOMAINS = {
    "calendar": {
        "dom": ("tidy up", "the schedule", "look over", "my week",
                "check", "the calendar", "sort out", "the agenda",
                "go over", "the events", "sort", "the invites"),
        "agents": {
            "meeting-scheduler": [("book", "a standup"), ("set up", "the sync"),
                                  ("schedule", "the review"), ("arrange", "a call"),
                                  ("lock in", "the 1:1"), ("plan", "the kickoff")],
            "reminder-setter": [("remind me", "about the standup"),
                                ("ping me", "before the sync"),
                                ("nudge me", "at meeting time"),
                                ("alert me", "about the review"),
                                ("warn me", "before the call"),
                                ("flag", "the upcoming 1:1")],
            "availability-checker": [("find", "a free slot"),
                                     ("look for", "an opening"),
                                     ("check", "my free time"),
                                     ("spot", "a conflict"),
                                     ("hunt for", "an opening"),
                                     ("see", "when I'm free")],
            "decliner": [("cancel", "the standup"), ("drop", "the sync"),
                         ("scrap", "the review call"), ("pull me out of", "the 1:1"),
                         ("free me from", "the call"), ("scuttle", "the kickoff")],
            "rescheduler": [("move", "the standup"), ("shift", "the sync"),
                            ("push", "the review"), ("bump", "the 1:1"),
                            ("postpone", "the call"), ("juggle", "the kickoff")],
            "minutes-taker": [("write up", "the standup notes"),
                              ("summarize", "the sync"),
                              ("transcribe", "the review"),
                              ("log", "the 1:1 decisions"),
                              ("record", "the call notes"),
                              ("capture", "the kickoff minutes")],
            "invite-manager": [("add", "ana to the standup"),
                               ("invite", "the team to the sync"),
                               ("cc", "raul on the review"),
                               ("loop in", "mia for the 1:1"),
                               ("add", "the intern to the call"),
                               ("send", "the kickoff invite")],
            "focus-blocker": [("block", "deep-work time"),
                              ("carve out", "a no-meeting slot"),
                              ("reserve", "focus hours"),
                              ("hold", "a quiet block"),
                              ("guard", "my maker time"),
                              ("wall off", "the afternoon")],
        },
    },
    "code": {
        "dom": ("fix", "the codebase", "work on", "the repo",
                "clean up", "the project", "look at", "the build",
                "look over", "the tests", "check on", "the pipeline"),
        "agents": {
            "bug-hunter": [("debug", "the login crash"), ("trace", "the null error"),
                           ("chase", "the race condition"), ("diagnose", "the 500s"),
                           ("reproduce", "the hang"), ("isolate", "the leak")],
            "reviewer": [("review", "my pull request"), ("look over", "the diff"),
                         ("critique", "the patch"), ("check", "the merge"),
                         ("vet", "the commit"), ("audit", "the branch")],
            "test-writer": [("write", "unit tests"), ("add", "test coverage"),
                            ("draft", "the test suite"), ("mock", "the api calls"),
                            ("cover", "the edge cases"), ("assert", "the outputs")],
            "refactorer": [("refactor", "the auth module"),
                           ("untangle", "the helpers"),
                           ("split", "the god class"),
                           ("rename", "the confusives"),
                           ("extract", "the duplicated logic"),
                           ("flatten", "the callbacks")],
            "documenter": [("document", "the api"), ("write", "the readme"),
                           ("explain", "the setup steps"),
                           ("annotate", "the config"),
                           ("describe", "the deploy process"),
                           ("draft", "the changelog")],
            "dependency-updater": [("bump", "the libraries"),
                                   ("upgrade", "the frameworks"),
                                   ("update", "the packages"),
                                   ("pin", "the versions"),
                                   ("patch", "the vulns"),
                                   ("refresh", "the lockfile")],
            "perf-tuner": [("profile", "the hot loop"), ("speed up", "the queries"),
                           ("cut", "the startup time"), ("cache", "the responses"),
                           ("trim", "the allocations"), ("bench", "the endpoint")],
            "migrator": [("port", "to the new framework"),
                          ("migrate", "the database"),
                          ("convert", "the old schemas"),
                          ("translate", "the legacy code"),
                          ("shift", "to python 3"),
                          ("move", "off the monolith")],
        },
    },
    "home": {
        "dom": ("around", "the house", "for", "the flat",
                "check on", "the place", "handle", "the chores",
                "walk through", "the rooms", "sort out", "the storage"),
        "agents": {
            "light-controller": [("dim", "the living room lamps"),
                                 ("brighten", "the kitchen lights"),
                                 ("switch off", "the hallway bulb"),
                                 ("warm up", "the bedroom lights"),
                                 ("toggle", "the porch light"),
                                 ("flood", "the garden lights")],
            "thermostat-agent": [("warm", "the bedroom"), ("cool", "the office"),
                                 ("set", "the temperature"), ("preheat", "the flat"),
                                 ("ease off", "the heating"),
                                 ("hold", "the climate")],
            "shopping-list-agent": [("add", "milk to the list"),
                                    ("note", "we need coffee"),
                                    ("put", "bread on the list"),
                                    ("add", "rice and beans"),
                                    ("list", "the veggies"),
                                    ("remember", "the cat food")],
            "cleaning-planner": [("plan", "the deep clean"),
                                 ("rota", "the chores"),
                                 ("schedule", "the vacuum run"),
                                 ("split", "the housework"),
                                 ("queue", "the laundry"),
                                 ("book", "the cleaner")],
            "security-monitor": [("arm", "the alarm"), ("check", "the door sensor"),
                                 ("review", "the camera clips"),
                                 ("watch", "the front door"),
                                 ("test", "the siren"),
                                 ("lock", "everything down")],
            "energy-tracker": [("track", "the power use"),
                               ("cut", "the standby drain"),
                               ("watch", "the meter"),
                               ("compare", "this month's kwh"),
                               ("audit", "the appliances"),
                               ("shave", "the peak usage")],
            "plant-butler": [("water", "the ferns"), ("check", "the orchid"),
                             ("feed", "the monstera"), ("mist", "the succulents"),
                             ("rotate", "the pots"), ("repot", "the ivy")],
            "repair-coordinator": [("book", "the plumber"),
                                   ("call", "the electrician"),
                                   ("schedule", "the boiler service"),
                                   ("chase", "the handyman"),
                                   ("get", "a roofing quote"),
                                   ("arrange", "the paint job")],
        },
    },
    "finance": {
        "dom": ("look at", "my money", "go over", "the budget",
                "check", "the accounts", "handle", "the bills",
                "go over", "the spending", "check on", "the savings"),
        "agents": {
            "budget-planner": [("set", "a food budget"), ("plan", "the monthly spend"),
                               ("cap", "the fun money"), ("split", "the paycheck"),
                               ("trim", "the subscriptions"),
                               ("map", "the categories")],
            "bill-negotiator": [("call", "the internet provider"),
                                ("haggle", "the insurance rate"),
                                ("dispute", "the gym charge"),
                                ("push back on", "the fees"),
                                ("renegotiate", "the phone plan"),
                                ("cancel", "the unused plan")],
            "savings-automator": [("auto-move", "50 to savings"),
                                  ("round up", "the purchases"),
                                  ("stash", "the bonus"),
                                  ("schedule", "the transfer"),
                                  ("fill", "the emergency fund"),
                                  ("skim", "the spare cash")],
            "expense-categorizer": [("sort", "the receipts"),
                                    ("tag", "the takeout spend"),
                                    ("split", "the groceries"),
                                    ("file", "the work expenses"),
                                    ("label", "the subscriptions"),
                                    ("bucket", "the shopping")],
            "subscription-auditor": [("list", "my recurring charges"),
                                     ("hunt", "the zombie subscriptions"),
                                     ("price", "the streaming stack"),
                                     ("compare", "the plans"),
                                     ("dump", "the dupes"),
                                     ("tally", "the memberships")],
            "invoice-chaser": [("chase", "the unpaid invoice"),
                               ("nudge", "the client on payment"),
                               ("bill", "the consultancy"),
                               ("re-send", "the overdue invoice"),
                               ("track", "the receivables"),
                               ("remind", "the debtor")],
            "tax-prepper": [("gather", "the deductible receipts"),
                            ("estimate", "what i owe"),
                            ("organize", "the tax folder"),
                            ("check", "the deadline"),
                            ("find", "my tax credits"),
                            ("prep", "the filings")],
            "portfolio-watcher": [("check", "my index funds"),
                                  ("rebalance", "the portfolio"),
                                  ("peek", "at the etfs"),
                                  ("flag", "the drift"),
                                  ("compare", "the fund fees"),
                                  ("eyeball", "the allocation")],
        },
    },
    "health": {
        "dom": ("with", "my health", "about", "my routine",
                "keep", "my habits", "watch", "my wellbeing",
                "look at", "my sleep", "check on", "my steps"),
        "agents": {
            "habit-coach": [("track", "my water intake"),
                            ("streak", "my meditation"),
                            ("log", "the daily walk"),
                            ("nudge", "my bedtime routine"),
                            ("count", "my smoke-free days"),
                            ("grade", "my week")],
            "workout-planner": [("plan", "a push day"), ("schedule", "leg day"),
                                ("cycle", "the 5k plan"), ("set", "my intervals"),
                                ("cycle", "the hypertrophy block"),
                                ("warm up", "the mobility flow")],
            "meal-logger": [("log", "my lunch"), ("count", "the protein"),
                            ("track", "the calories"), ("note", "the snacks"),
                            ("sum up", "today's carbs"),
                            ("record", "the water")],
            "sleep-auditor": [("analyze", "my sleep stages"),
                              ("fix", "my wind-down"),
                              ("shift", "my bed time"),
                              ("cut", "the late caffeine"),
                              ("score", "last night"),
                              ("tune", "the alarm window")],
            "med-reminder": [("remind me", "about the pills"),
                             ("schedule", "the vitamin dose"),
                             ("time", "the second dose"),
                             ("watch", "the refill date"),
                             ("prompt", "the inhaler"),
                             ("track", "the prescription")],
            "symptom-tracker": [("log", "the headaches"),
                                ("correlate", "the flare ups"),
                                ("chart", "the energy dips"),
                                ("note", "the allergy days"),
                                ("watch", "the recovery"),
                                ("trend", "the joint pain")],
            "appointment-booker": [("book", "a dentist slot"),
                                   ("schedule", "the checkup"),
                                   ("find", "a physio opening"),
                                   ("renew", "the prescription appt"),
                                   ("arrange", "the eye test"),
                                   ("move", "the annual physical")],
            "hydration-nudge": [("ping me", "to drink water"),
                                ("pace", "my glasses a day"),
                                ("remind", "the refill"),
                                ("track", "the bottles"),
                                ("nudge", "the electrolytes"),
                                ("watch", "my intake")],
        },
    },
    "media": {
        "dom": ("with", "the music", "sort", "the playlists",
                "find", "something to watch", "play", "the album",
                "go over", "the queue", "look for", "a film"),
        "agents": {
            "playlist-curator": [("build", "a running playlist"),
                                 ("mix", "the focus queue"),
                                 ("line up", "the road trip songs"),
                                 ("refresh", "the gym list"),
                                 ("queue", "the chill set"),
                                 ("theme", "the dinner music")],
            "discoverer": [("find", "new indie releases"),
                           ("dig up", "similar artists"),
                           ("surface", "hidden gems"),
                           ("scout", "this week's drops"),
                           ("recommend", "an album"),
                           ("expand", "my taste")],
            "watchlist-manager": [("add", "the series to my list"),
                                  ("pick", "tonight's film"),
                                  ("sort", "the watchlist"),
                                  ("flag", "the new episodes"),
                                  ("rank", "the queue"),
                                  ("prune", "the abandoned shows")],
            "photo-organizer": [("sort", "the camera roll"),
                                ("album", "the holiday pics"),
                                ("dedupe", "the screenshots"),
                                ("tag", "the family photos"),
                                ("archive", "the old shots"),
                                ("curate", "the best fifty")],
            "podcast-sifter": [("find", "a history podcast"),
                               ("queue", "the new episodes"),
                               ("trim", "the backlog"),
                               ("suggest", "a commute listen"),
                               ("speed up", "the queue"),
                               ("chapter", "the long ones")],
            "volume-automator": [("duck", "the volume at night"),
                                 ("normalize", "the speakers"),
                                 ("set", "the tv level"),
                                 ("quiet", "the ads"),
                                 ("boost", "the dialogue"),
                                 ("schedule", "the quiet hours")],
            "lyric-finder": [("find", "that song from the ad"),
                             ("identify", "the hummed tune"),
                             ("look up", "the lyrics"),
                             ("name", "the sample"),
                             ("trace", "the earworm"),
                             ("credit", "the vocalist")],
            "device-dj": [("cast", "the kitchen speaker"),
                          ("sync", "the rooms"),
                          ("hand off", "to the speaker"),
                          ("group", "the players"),
                          ("move", "the music upstairs"),
                          ("queue", "on the soundbar")],
        },
    },
    "travel": {
        "dom": ("plan", "the trip", "sort", "the journey",
                "look at", "the getaway", "prep", "the travel",
                "look over", "the options", "check on", "the itinerary"),
        "agents": {
            "flight-scout": [("find", "cheap flights"), ("watch", "the fares"),
                             ("compare", "the connections"),
                             ("sniff out", "the deal"),
                             ("track", "the price drop"),
                             ("book", "the legroom seat")],
            "hotel-hunter": [("find", "a hotel near the venue"),
                             ("compare", "the rates"),
                             ("vet", "the reviews"),
                             ("hunt", "a quiet room"),
                             ("shortlist", "the stays"),
                             ("snag", "the free breakfast")],
            "itinerary-builder": [("plan", "the rome days"),
                                  ("pace", "the sightseeing"),
                                  ("map", "the walking route"),
                                  ("order", "the museum days"),
                                  ("slot", "the food stops"),
                                  ("pad", "the rest time")],
            "packlist-agent": [("list", "the carry-on"),
                               ("check", "the adapters"),
                               ("pack", "for the rain"),
                               ("weigh", "the bag"),
                               ("prep", "the toiletries"),
                               ("remind", "the passport")],
            "transit-router": [("route", "the airport run"),
                               ("compare", "the train times"),
                               ("find", "the night bus"),
                               ("time", "the connection"),
                               ("book", "the transfer"),
                               ("plan", "the metro passes")],
            "currency-advisor": [("check", "the exchange rate"),
                                 ("tip", "the cash vs card"),
                                 ("warn", "about the fees"),
                                 ("budget", "the daily spend"),
                                 ("find", "the atm advice"),
                                 ("convert", "the prices")],
            "visa-checker": [("check", "the entry rules"),
                             ("verify", "the passport validity"),
                             ("list", "the required forms"),
                             ("flag", "the vaccine rules"),
                             ("confirm", "the transit visa"),
                             ("sum up", "the customs limits")],
            "loyalty-optimizer": [("use", "the points"),
                                  ("compare", "the seat upgrades"),
                                  ("track", "the miles"),
                                  ("find", "the lounge access"),
                                  ("stack", "the status credits"),
                                  ("spend", "the voucher")],
        },
    },
    "writing": {
        "dom": ("with", "my writing", "work on", "the draft",
                "look at", "the text", "help with", "the words",
                "go over", "the notes", "sort out", "the outline"),
        "agents": {
            "draft-kicker": [("start", "the blog post"),
                             ("outline", "the newsletter"),
                             ("rough out", "the talk"),
                             ("sketch", "the intro"),
                             ("prime", "the first paragraph"),
                             ("frame", "the essay")],
            "editor": [("tighten", "the draft"), ("cut", "the fluff"),
                       ("fix", "the clunky sentences"),
                       ("smooth", "the transitions"),
                       ("sharpen", "the ending"),
                       ("prune", "the adverbs")],
            "proofreader": [("catch", "the typos"), ("check", "the grammar"),
                            ("fix", "the punctuation"),
                            ("hunt", "the double spaces"),
                            ("verify", "the names"),
                            ("sweep", "the homophones")],
            "translator": [("translate", "the email to spanish"),
                           ("render", "the doc in french"),
                           ("localize", "the landing page"),
                           ("convert", "the subtitles"),
                           ("port", "the manual"),
                           ("switch", "the captions to german")],
            "tone-shifter": [("make it", "more formal"),
                             ("soften", "the rejection"),
                             ("punch up", "the announcement"),
                             ("calm", "the angry reply"),
                             ("lighten", "the status update"),
                             ("professionalize", "the complaint")],
            "summarizer": [("condense", "the report"),
                           ("bullet", "the meeting notes"),
                           ("tldr", "the article"),
                           ("digest", "the thread"),
                           ("boil down", "the proposal"),
                           ("brief", "the boss")],
            "name-stormer": [("brainstorm", "podcast names"),
                             ("name", "the newsletter"),
                             ("suggest", "a title"),
                             ("mint", "a product name"),
                             ("spin", "the slogan"),
                             ("shortlist", "the domain ideas")],
            "story-continuer": [("continue", "my scene"),
                                ("unstick", "the plot"),
                                ("flesh out", "the side character"),
                                ("bridge", "the chapter gap"),
                                ("escalate", "the standoff"),
                                ("land", "the ending")],
        },
    },
}


def build(m, seed=0, n_refs=32, n_eval=64):
    """Return refs/evals: dicts with 'agent' (text lists, per agent),
    'domain' (text lists, per domain). Eval items carry gold labels."""
    import random
    rng = random.Random(seed)
    dom_names = list(DOMAINS)
    refs, evals = {"agent": {}, "domain": {}}, []
    for dname in dom_names:
        d = DOMAINS[dname]
        # Amendment 1: domain pools draw from the domain's FULL vocabulary
        # (dom pairs + every member agent's specialty pairs) — production
        # coarse nodes own subclass utterances; generic-only pools collapsed
        # (domain_acc 0.39 @ m=1.0, pass 1 archived). Amendment 2: texts are
        # prefix + TWO pairs (probe series: shared filler swamped the signal;
        # two content pairs lifted A_flat 0.35 -> 0.91 @ m=1.0).
        gen = [(d["dom"][2 * k], d["dom"][2 * k + 1])
               for k in range(len(d["dom"]) // 2)]
        vocab = list(gen)
        for apairs in d["agents"].values():
            vocab.extend(apairs)
        dpool, dseen = [], set()
        while len(dpool) < n_refs:
            p1 = vocab[rng.randrange(len(vocab))]
            p2 = vocab[rng.randrange(len(vocab))]
            t = (f"{rng.choice(PREFIXES)} {p1[0]} {p1[1]} "
                 f"{p2[0]} {p2[1]}")
            if t not in dseen:
                dseen.add(t)
                dpool.append(t)
        refs["domain"][dname] = dpool
        for aname, pairs in d["agents"].items():
            def marked():
                p1 = pairs[rng.randrange(len(pairs))]
                p2 = pairs[rng.randrange(len(pairs))]
                return (f"{rng.choice(PREFIXES)} {p1[0]} {p1[1]} "
                        f"{p2[0]} {p2[1]}")

            def unmarked():
                p1 = gen[rng.randrange(len(gen))]
                p2 = gen[rng.randrange(len(gen))]
                return (f"{rng.choice(PREFIXES)} {p1[0]} {p1[1]} "
                        f"{p2[0]} {p2[1]}")

            # draw with dedup. Amendment 2 universe re-derivation: per agent
            # 6 prefixes x 6^2 combos = 216 >= 2x the 96-draw pool (the old
            # 448 clause assumed the retired single-pair template)
            pool, seen = [], set()
            while len(pool) < n_refs + n_eval:
                t = marked() if rng.random() < m else unmarked()
                if t not in seen:
                    seen.add(t)
                    pool.append(t)
            refs["agent"][f"{dname}.{aname}"] = pool[:n_refs]
            for text in pool[n_refs:]:
                evals.append({"text": text, "gold_domain": dname,
                              "gold_agent": f"{dname}.{aname}"})
    return refs, evals


def l2(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def centroids(vecs_by_key, keys):
    return {k: l2(np.stack([l2(vecs_by_key[t].astype(np.float32))
                            for t in texts]).mean(0))
            for k, texts in keys.items()}


def run_m(m, cache, n_refs=32, n_eval=64, seed=0):
    refs, evals = build(m, seed=seed, n_refs=n_refs, n_eval=n_eval)
    akeys = refs["agent"]
    EC.embed([t for ts in akeys.values() for t in ts], cache)
    EC.embed([t for ts in refs["domain"].values() for t in ts], cache)
    EC.embed([e["text"] for e in evals], cache)
    EC.save_cache(cache)

    acents = centroids(cache, akeys)
    dcents = centroids(cache, refs["domain"])
    anames = list(acents)
    dnames = list(dcents)
    Amat = np.stack([acents[k] for k in anames])
    Dmat = np.stack([dcents[k] for k in dnames])
    members = {d: [i for i, k in enumerate(anames) if k.startswith(d + ".")]
               for d in dnames}

    stat = {a: {"n": 0, "ok": 0} for a in
            ("A_flat", "B_walk", "B_oracle", "E_beam2")}
    dom = {"n": 0, "ok": 0}
    for e in evals:
        v = l2(cache[e["text"]].astype(np.float32)[None, :])[0]
        dsims = Dmat @ v
        d_order = np.argsort(-dsims)
        asims = Amat @ v
        # A_flat
        stat["A_flat"]["n"] += 1
        stat["A_flat"]["ok"] += int(anames[int(np.argmax(asims))]
                                    == e["gold_agent"])
        # domain readout (every arm's coarse level, incl. oracle ceiling)
        dom["n"] += 1
        dom["ok"] += int(dnames[d_order[0]] == e["gold_domain"])
        gold_d = e["gold_domain"]
        # B_walk
        d0 = dnames[d_order[0]]
        pick = anames[members[d0][int(np.argmax(asims[members[d0]]))]]
        stat["B_walk"]["n"] += 1
        stat["B_walk"]["ok"] += int(pick == e["gold_agent"])
        # B_oracle
        pick = anames[members[gold_d][int(np.argmax(asims[members[gold_d]]))]]
        stat["B_oracle"]["n"] += 1
        stat["B_oracle"]["ok"] += int(pick == e["gold_agent"])
        # E_beam2: top-2 domains, agent top-1 per path, mean path picks
        best = None
        for ci in d_order[:2]:
            dname = dnames[ci]
            mem = members[dname]
            j = int(np.argmax(asims[mem]))
            pick = anames[mem[j]]
            score = 0.5 * (float(dsims[ci]) + float(asims[mem[j]]))
            if best is None or score > best[0]:
                best = (score, pick)
        stat["E_beam2"]["n"] += 1
        stat["E_beam2"]["ok"] += int(best[1] == e["gold_agent"])

    out = {"m": m, "n_agents": len(anames), "n_eval": len(evals),
           "arms": {a: round(s["ok"] / max(s["n"], 1), 4)
                    for a, s in stat.items()},
           "domain_acc": round(dom["ok"] / max(dom["n"], 1), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        cache = EC.load_cache()
        r = run_m(1.0, cache, n_refs=4, n_eval=8)
        print(json.dumps(r, indent=1))
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    cache = EC.load_cache()
    print(f"embed cache: {len(cache)} vectors, fp={EC.fingerprint()}",
          flush=True)
    all_out = {}
    for m in (1.0, 0.5, 0.0):
        print(f"--- m={m}", flush=True)
        r = run_m(m, cache)
        all_out[f"m{int(m*10):02d}"] = r
        print(json.dumps(r, indent=1), flush=True)
        json.dump(all_out, open(os.path.join(RESULTS, "subagents.json"), "w"),
                  indent=1)
    # registered verdict lines (S-series)
    for tag, m in (("m10", 1.0), ("m05", 0.5)):
        r = all_out[tag]
        d = r["arms"]["E_beam2"] - r["arms"]["B_walk"]
        print(f"S1 @{tag}: beam2-greedy = {d:+.4f} "
              f"({'PASS' if d >= 0.01 else 'FAIL'} bar +0.01)")
    r = all_out["m00"]
    floors = [r["arms"][a] for a in ("A_flat", "B_walk", "B_oracle")]
    print(f"S3 @m00: arms {[round(x, 4) for x in floors]} vs 1/A=0.125; "
          f"domain_acc={r['domain_acc']} "
          f"({'PASS' if r['domain_acc'] >= 0.90 else 'FAIL'} domain >= 0.90)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
