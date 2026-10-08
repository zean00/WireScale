"""Synthetic utterance-marked hierarchy for tree-lab (protocol.md machinery).

Deterministic: numpy default_rng, data seed 0. An utterance belongs to a
leaf path [L1, L2, L3, L4]; depth-d classes are prefixes. Sibling leaves
share objs/verbs; ONLY the modifier carries leaf information. Markedness
m = P(leaf modifier) else the L3 parent's generic pool (leaf-free), so
m=0 makes siblings identically distributed (50% conditional floor).
All EVAL text uses held-out verbs and held-out slot values.
"""
import numpy as np

VERBS_TRAIN = ["set", "turn", "adjust"]
VERBS_HELD = ["make", "put"]

SLOTS = {
    "room": (["living room", "bedroom", "kitchen", "study", "hallway", "garage"],
             ["nursery", "loft"]),
    "pct": ([str(v) for v in range(10, 91, 5)], ["3", "7", "95", "97"]),
    "tmp": ([str(v) for v in range(18, 27)], ["16", "28"]),
    "time": (["9 am", "7 pm", "noon", "sunset", "midnight", "6 am"],
             ["5:30 pm", "sunrise"]),
    "day": (["monday", "tuesday", "wednesday", "thursday", "friday"],
            ["saturday", "sunday"]),
}

# L3 node: objs (may hold {room}); generic mods carry no leaf info.
# Leaves: mods (may hold slots); sibling mods are mutually exclusive topics.
TREE = {
    "smart_home": {
        "lighting": {
            "fixtures": {
                "objs": ["the {room} lamps", "the {room} ceiling lights"],
                "generic": ["a bit", "for tonight", "gently"],
                "leaves": {
                    "brightness": ["to {pct} percent", "down to {pct} percent",
                                   "up to {pct} percent"],
                    "color": ["to warm white", "to a soft blue",
                              "to daylight white"],
                },
            },
            "scenes": {
                "objs": ["the movie scene", "the evening scene"],
                "generic": ["the usual way", "please", "again"],
                "leaves": {
                    "activate": ["now", "right away", "without delay"],
                    "schedule": ["for {time}", "starting at {time}",
                                 "every night at {time}"],
                },
            },
        },
        "climate": {
            "temperature": {
                "objs": ["the {room} thermostat",
                         "the temperature in the {room}"],
                "generic": ["a little", "for comfort", "please"],
                "leaves": {
                    "setpoint": ["to {tmp} degrees", "down to {tmp} degrees",
                                 "up to {tmp} degrees"],
                    "schedule": ["to weekday hours", "to the daytime plan",
                                 "to the night plan"],
                },
            },
            "air": {
                "objs": ["the {room} fan", "the air in the {room}"],
                "generic": ["a bit", "for a while", "gently"],
                "leaves": {
                    "fan": ["to medium speed", "to high speed",
                            "to low speed"],
                    "purifier": ["to auto mode", "to quiet mode",
                                 "to turbo mode"],
                },
            },
        },
    },
    "media": {
        "audio": {
            "playback": {
                "objs": ["my playlist", "the morning playlist"],
                "generic": ["now", "please", "again"],
                "leaves": {
                    "start": ["from the top", "with something upbeat",
                              "with something calm"],
                    "queue": ["with more like this", "with the new album",
                              "with ten more songs"],
                },
            },
            "output": {
                "objs": ["the music", "the audio"],
                "generic": ["for now", "please", "for the whole house"],
                "leaves": {
                    "speakers": ["on the {room} speakers",
                                 "on the speakers downstairs",
                                 "on every speaker"],
                    "headphones": ["to my headphones",
                                   "to the bluetooth headphones",
                                   "to the earbuds"],
                },
            },
        },
        "video": {
            "playback": {
                "objs": ["the movie", "the show"],
                "generic": ["now", "please", "from where I left off"],
                "leaves": {
                    "start": ["from the beginning", "from season one",
                              "with the newest episode"],
                    "skip": ["forward ten minutes", "to the next episode",
                             "past the intro"],
                },
            },
            "display": {
                "objs": ["the screen", "the picture"],
                "generic": ["a bit", "for now", "please"],
                "leaves": {
                    "cast": ["to the {room} tv", "to the big screen",
                             "to the projector"],
                    "brightness": ["down", "up", "to full"],
                },
            },
        },
    },
    "productivity": {
        "capture": {
            "notes": {
                "objs": ["a note", "my notes"],
                "generic": ["quickly", "please", "for later"],
                "leaves": {
                    "create": ["about the grocery list",
                               "about tomorrow's errands",
                               "about the meeting"],
                    "find": ["from last week", "about the trip",
                             "with the receipts"],
                },
            },
            "tasks": {
                "objs": ["a task", "my task list"],
                "generic": ["quickly", "please", "for today"],
                "leaves": {
                    "add": ["to buy stamps", "to call the dentist",
                            "to water the plants"],
                    "complete": ["the first one", "the errand one",
                                 "all the done ones"],
                },
            },
        },
        "time": {
            "scheduling": {
                "objs": ["a meeting", "my calendar"],
                "generic": ["next week", "please", "some day"],
                "leaves": {
                    "create": ["on {day} morning", "on {day} afternoon",
                               "for {day} evening"],
                    "move": ["to {day}", "to the same time on {day}",
                             "after lunch on {day}"],
                },
            },
            "reminders": {
                "objs": ["a reminder", "an alarm"],
                "generic": ["please", "for later", "quietly"],
                "leaves": {
                    "set": ["for {time}", "at {time} sharp",
                            "every day at {time}"],
                    "snooze": ["by ten minutes", "for five more minutes",
                               "until {time}"],
                },
            },
        },
    },
}

DATA_SEED = 0
POOL_PER_LEAF = 1024
EVAL_PER_LEAF = 40


def leaf_paths():
    """24 leaf paths in tree order, e.g. ['smart_home','lighting','fixtures','brightness']."""
    out = []
    for l1, subs in TREE.items():
        for l2, l3s in subs.items():
            for l3, node in l3s.items():
                for leaf in node["leaves"]:
                    out.append([l1, l2, l3, leaf])
    return out


def node_table():
    """Ordered node registry (tree order): {name: (depth, parent, children)}.
    Depth-1..4 names are tuples; index = position in this list (one-hot basis)."""
    nodes, order = {}, []

    def rec(name, depth, parent, children_map):
        order.append(name)
        nodes[name] = {"depth": depth, "parent": parent, "children": []}
        for ch_name, ch_kids in children_map:
            ch_path = name + (ch_name,) if name != ("root",) else (ch_name,)
            nodes[name]["children"].append(ch_path)
            rec(ch_path, depth + 1, name, ch_kids)

    top = [(l1, [(l2, [(l3, [(lf, []) for lf in TREE[l1][l2][l3]["leaves"]])
                     for l3 in TREE[l1][l2]])
             for l2 in TREE[l1]]) for l1 in TREE]
    rec(("root",), 0, None, top)
    return nodes


def _fill(template, rng, heldout):
    out = template
    for slot, (train_vals, held_vals) in SLOTS.items():
        tok = "{" + slot + "}"
        while tok in out:
            out = out.replace(tok, rng.choice(held_vals if heldout
                                              else train_vals), 1)
    return out


def make_utterance(path, rng, m, heldout):
    l1, l2, l3, leaf = path
    node = TREE[l1][l2][l3]
    verb = rng.choice(VERBS_HELD if heldout else VERBS_TRAIN)
    obj = _fill(rng.choice(node["objs"]), rng, heldout)
    if rng.random() < m:
        mod = rng.choice(node["leaves"][leaf])
    else:
        mod = rng.choice(node["generic"])
    return f"{verb} {obj} {_fill(mod, rng, heldout)}"


def build_masters(m):
    """{'train': [(text, leaf_idx)], 'eval': [(text, leaf_idx)]} for one m.
    Pool/depth variants are subsamples/relabelings of these."""
    paths = leaf_paths()
    train, ev = [], []
    for i, p in enumerate(paths):
        rng_t = np.random.default_rng(DATA_SEED * 100003 + 7 * i + int(m * 100))
        rng_e = np.random.default_rng(DATA_SEED * 100003 + 7 * i + int(m * 100) + 1)
        train += [(make_utterance(p, rng_t, m, False), i)
                  for _ in range(POOL_PER_LEAF)]
        ev += [(make_utterance(p, rng_e, m, True), i)
               for _ in range(EVAL_PER_LEAF)]
    return {"train": train, "eval": ev}


def class_at(depth, leaf_idx):
    """Depth-d class tuple for a leaf index (d in 1..4)."""
    return tuple(leaf_paths()[leaf_idx][:depth])
