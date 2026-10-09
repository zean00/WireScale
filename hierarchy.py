"""Synthetic utterance-marked hierarchy for WireScale (protocol.md machinery).

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

# Uniform non-informative prefix (8 train / 2 held-out): multiplied into
# every utterance so even slotless+roomless leaves clear pool 1024 uniques.
PREFIXES_TRAIN = ["please", "can you", "could you", "now", "hey", "ok then",
                  "right away", "for me"]
PREFIXES_HELD = ["one more thing", "also"]

SLOTS = {
    "room": (["living room", "bedroom", "kitchen", "study", "hallway",
              "garage", "basement", "attic"],
             ["nursery", "loft"]),
    "pct": ([str(v) for v in range(10, 91)], ["3", "7", "93", "95", "97"]),
    "tmp": ([str(v) for v in range(15, 31)], ["13", "14", "32", "33"]),
    "time": (["9 am", "7 pm", "noon", "sunset", "midnight", "6 am", "3 pm",
              "10 pm", "8 am", "5 pm", "11 am", "2 pm"],
             ["5:30 pm", "sunrise", "4:45 am", "11:15 pm"]),
    "day": (["monday", "tuesday", "wednesday", "thursday", "friday"],
            ["saturday", "sunday"]),
}

# L3 node: objs (each holds {room}); generic mods carry no leaf info.
# Leaves: mods (may hold slots); sibling mods are mutually exclusive topics.
TREE = {
    "smart_home": {
        "lighting": {
            "fixtures": {
                "objs": ["the {room} lamps", "the {room} ceiling lights",
                         "the {room} desk lamp", "the lights in the {room}",
                         "the {room} light strip", "the {room} floor lamp"],
                "generic": ["a bit", "for tonight", "gently", "for a moment",
                            "in a little while", "quietly"],
                "leaves": {
                    "brightness": ["to {pct} percent", "down to {pct} percent",
                                   "up to {pct} percent",
                                   "all the way to {pct} percent",
                                   "to around {pct} percent",
                                   "from {pct} percent down",
                                   "to {pct} percent brightness",
                                   "back to {pct} percent"],
                    "color": ["to warm white", "to a soft blue",
                              "to daylight white", "to a deep red",
                              "to a gentle amber", "to cool white",
                              "to a pale green", "to a warm orange"],
                },
            },
            "scenes": {
                "objs": ["the movie scene", "the evening scene",
                         "the reading scene", "the relax scene",
                         "the focus scene", "the party scene"],
                "generic": ["the usual way", "please", "again", "in a bit",
                            "for me", "one more time"],
                "leaves": {
                    "activate": ["now", "right away", "without delay",
                                 "this instant", "immediately please",
                                 "as soon as possible", "on", "right now"],
                    "schedule": ["for {time}", "starting at {time}",
                                 "every night at {time}",
                                 "to begin at {time}", "for {time} tonight",
                                 "to run at {time}",
                                 "so it starts at {time}",
                                 "for around {time}"],
                },
            },
        },
        "climate": {
            "temperature": {
                "objs": ["the {room} thermostat",
                         "the temperature in the {room}",
                         "the {room} climate control", "the heat in the {room}",
                         "the {room} ac unit", "the thermostat in the {room}"],
                "generic": ["a little", "for comfort", "please", "a touch",
                            "for now", "if you would"],
                "leaves": {
                    "setpoint": ["to {tmp} degrees", "down to {tmp} degrees",
                                 "up to {tmp} degrees",
                                 "to about {tmp} degrees",
                                 "all the way to {tmp} degrees",
                                 "to {tmp} degrees exactly",
                                 "from {tmp} degrees up",
                                 "to a steady {tmp} degrees"],
                    "schedule": ["to weekday hours", "to the daytime plan",
                                 "to the night plan", "to the weekend plan",
                                 "to the workday schedule",
                                 "to the away schedule", "to the morning plan",
                                 "to the eco plan"],
                },
            },
            "air": {
                "objs": ["the {room} fan", "the air in the {room}",
                         "the {room} ceiling fan",
                         "the ventilation in the {room}",
                         "the {room} air purifier",
                         "the airflow in the {room}"],
                "generic": ["a bit", "for a while", "gently", "for now",
                            "please", "for the next hour"],
                "leaves": {
                    "fan": ["to medium speed", "to high speed",
                            "to low speed", "to full speed",
                            "to the lowest speed", "to gentle mode",
                            "to strong mode", "to quiet speed"],
                    "purifier": ["to auto mode", "to quiet mode",
                                 "to turbo mode", "to sleep mode",
                                 "to max mode", "to eco mode",
                                 "to boost mode", "to standard mode"],
                },
            },
        },
    },
    "media": {
        "audio": {
            "playback": {
                "objs": ["my playlist", "the morning playlist",
                         "my workout playlist", "the focus playlist",
                         "my reading playlist", "my favorite album"],
                "generic": ["now", "please", "again", "in a moment", "for me",
                            "once more"],
                "leaves": {
                    "start": ["from the top", "with something upbeat",
                              "with something calm", "with the first track",
                              "with something new", "from a random spot",
                              "with my favorites", "with something classic"],
                    "queue": ["with more like this", "with the new album",
                              "with ten more songs", "with similar tracks",
                              "with more from this artist",
                              "with the rest of the list",
                              "with something similar",
                              "with twenty more songs"],
                },
            },
            "output": {
                "objs": ["the music in the {room}", "the audio",
                         "the sound in the {room}", "the podcast",
                         "the radio in the {room}", "the audiobook"],
                "generic": ["for now", "please", "for the whole house",
                            "for a while", "for everyone", "for the evening"],
                "leaves": {
                    "speakers": ["on the {room} speakers",
                                 "on the speakers downstairs",
                                 "on every speaker",
                                 "on the {room} speaker pair",
                                 "on the outdoor speakers",
                                 "on all the {room} speakers",
                                 "on the bookshelf speakers",
                                 "on the {room} soundbar"],
                    "headphones": ["to my headphones",
                                   "to the bluetooth headphones",
                                   "to the earbuds",
                                   "to my wired headphones",
                                   "to the office headphones",
                                   "to the spare earbuds",
                                   "to my gaming headset",
                                   "to the new headphones"],
                },
            },
        },
        "video": {
            "playback": {
                "objs": ["the movie", "the show", "the documentary",
                         "the series", "the film", "the episode"],
                "generic": ["now", "please", "from where I left off",
                            "in a moment", "for me", "later"],
                "leaves": {
                    "start": ["from the beginning", "from season one",
                              "with the newest episode",
                              "from the first minute", "with episode one",
                              "from the start", "with the pilot",
                              "with the latest one"],
                    "skip": ["forward ten minutes", "to the next episode",
                             "past the intro", "ahead five minutes",
                             "to the final scene", "past the recap",
                             "forward half an hour", "to the next chapter"],
                },
            },
            "display": {
                "objs": ["the screen", "the picture in the {room}",
                         "the display", "the stream",
                         "the video in the {room}", "the tv screen"],
                "generic": ["a bit", "for now", "please", "a little",
                            "for the moment", "slowly"],
                "leaves": {
                    "cast": ["to the {room} tv", "to the big screen",
                             "to the projector", "to the {room} display",
                             "to the television in the {room}",
                             "to the {room} monitor", "to the wall screen",
                             "to the {room} frame"],
                    "brightness": ["down", "up", "to full", "halfway down",
                                   "to the minimum", "up a notch",
                                   "down a notch", "to the maximum"],
                },
            },
        },
    },
    "productivity": {
        "capture": {
            "notes": {
                "objs": ["a note", "my notes", "a quick note", "my notebook",
                         "a memo", "an entry"],
                "generic": ["quickly", "please", "for later", "when you can",
                            "for me", "sometime today"],
                "leaves": {
                    "create": ["about the grocery list",
                               "about tomorrow's errands", "about the meeting",
                               "about the book idea", "about the trip plan",
                               "about the gift list", "about the recipe",
                               "about the packing list"],
                    "find": ["from last week", "about the trip",
                             "with the receipts", "about the meeting",
                             "from yesterday", "with the address",
                             "about the budget", "from last month"],
                },
            },
            "tasks": {
                "objs": ["a task", "my task list", "a to-do", "my to-do list",
                         "a chore", "a checklist item"],
                "generic": ["quickly", "please", "for today", "when possible",
                            "for this week", "for me"],
                "leaves": {
                    "add": ["to buy stamps", "to call the dentist",
                            "to water the plants", "to fix the shelf",
                            "to order filters", "to renew the pass",
                            "to book the car service", "to email the landlord"],
                    "complete": ["the first one", "the errand one",
                                 "all the done ones", "the top item",
                                 "the oldest one", "the quick ones",
                                 "the shopping ones", "today's items"],
                },
            },
        },
        "time": {
            "scheduling": {
                "objs": ["a meeting", "my calendar", "an appointment",
                         "my schedule", "a call", "a block"],
                "generic": ["next week", "please", "some day", "soon",
                            "if it fits", "this month"],
                "leaves": {
                    "create": ["on {day} morning", "on {day} afternoon",
                               "for {day} evening", "early on {day}",
                               "late on {day}", "around noon on {day}",
                               "for {day} at ten", "on {day} if free"],
                    "move": ["to {day}", "to the same time on {day}",
                             "after lunch on {day}", "to {day} morning",
                             "an hour later on {day}", "to early {day}",
                             "to the afternoon of {day}",
                             "to the end of {day}"],
                },
            },
            "reminders": {
                "objs": ["a reminder", "an alarm", "a timer", "an alert",
                         "a notification", "a wake-up call"],
                "generic": ["please", "for later", "quietly", "in advance",
                            "twice", "gently"],
                "leaves": {
                    "set": ["for {time}", "at {time} sharp",
                            "every day at {time}", "for {time} tomorrow",
                            "to ring at {time}", "for about {time}",
                            "each morning at {time}", "to go off at {time}"],
                    "snooze": ["by ten minutes", "for five more minutes",
                               "until {time}", "by fifteen minutes",
                               "for another nine minutes",
                               "until {time} again", "by half an hour",
                               "for a few more minutes"],
                },
            },
        },
    },
}

DATA_SEED = 0
POOL_PER_LEAF = 1024
EVAL_PER_LEAF = 40

# Amendment 2 (depth extension): every depth-4 leaf splits on two further
# binary axes, appended as trailing modifier tokens. Axis words are shared
# across all branches (lexicalized questions, best-case marking) and are
# train-only (no held-out variant). Depth-6 tree = 96 leaves.
SUB_AXES = {
    5: (("full", "with full detail"), ("brief", "in brief")),
    6: (("urgent", "urgently"), ("eventual", "eventually")),
}
GENERIC_DEEP = "as usual"  # shared m-dial fallback beyond L4


def leaf_paths(depth=4):
    """Leaf paths of length `depth` (4..6), tree order."""
    out = []
    for l1, subs in TREE.items():
        for l2, l3s in subs.items():
            for l3, node in l3s.items():
                for leaf in node["leaves"]:
                    base = [l1, l2, l3, leaf]
                    if depth == 4:
                        out.append(base)
                    elif depth == 6:
                        for (a5, _), (a6, _) in ((SUB_AXES[5][0], SUB_AXES[6][0]),
                                                 (SUB_AXES[5][0], SUB_AXES[6][1]),
                                                 (SUB_AXES[5][1], SUB_AXES[6][0]),
                                                 (SUB_AXES[5][1], SUB_AXES[6][1])):
                            out.append(base + [a5, a6])
                    else:
                        raise ValueError(depth)
    return out


def node_table(depth=4):
    """Ordered node registry (tree order): {name: (depth, parent, children)}.
    Depth-d names are path tuples; index = position in this list."""
    nodes, order = {}, []

    def rec(name, depth_, parent, children_map):
        order.append(name)
        nodes[name] = {"depth": depth_, "parent": parent, "children": []}
        for ch_name, ch_kids in children_map:
            ch_path = name + (ch_name,) if name != ("root",) else (ch_name,)
            nodes[name]["children"].append(ch_path)
            rec(ch_path, depth_ + 1, name, ch_kids)

    top = [(l1, [(l2, [(l3, [(lf, []) for lf in TREE[l1][l2][l3]["leaves"]])
                     for l3 in TREE[l1][l2]])
             for l2 in TREE[l1]]) for l1 in TREE]
    if depth >= 6:
        (n5a, _), (n5b, _) = SUB_AXES[5]
        (n6a, _), (n6b, _) = SUB_AXES[6]
        for l1, subs in top:
            for l2, l3s in subs:
                for l3, leaves in l3s:
                    leaves[:] = [(lf, [(n5a, [(n6a, []), (n6b, [])]),
                                       (n5b, [(n6a, []), (n6b, [])])])
                                 for lf, _ in leaves]
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
    l1, l2, l3, leaf = path[0], path[1], path[2], path[3]
    node = TREE[l1][l2][l3]
    verb = rng.choice(VERBS_HELD if heldout else VERBS_TRAIN)
    pre = rng.choice(PREFIXES_HELD if heldout else PREFIXES_TRAIN)
    obj = _fill(rng.choice(node["objs"]), rng, heldout)
    if rng.random() < m:
        mod = rng.choice(node["leaves"][leaf])
    else:
        mod = rng.choice(node["generic"])
    tail = []
    for d in (5, 6):
        if len(path) >= d:
            if rng.random() < m:
                tail.append(_word_of(SUB_AXES[d], path[d - 1]))
            else:
                tail.append(GENERIC_DEEP)
    return f"{pre} {verb} {obj} {_fill(mod, rng, heldout)} " + " ".join(tail) \
        if tail else f"{pre} {verb} {obj} {_fill(mod, rng, heldout)}"


def _word_of(axis, name):
    for word, mod in axis:
        if word == name:
            return mod
    raise KeyError(name)


def build_masters(m, depth=4):
    """{'train': [(text, leaf_idx)], 'eval': [(text, leaf_idx)]} for one m.
    Pool/depth variants are subsamples/relabelings of these."""
    paths = leaf_paths(depth)
    train, ev = [], []
    for i, p in enumerate(paths):
        rng_t = np.random.default_rng(DATA_SEED * 100003 + 7 * i + int(m * 100))
        rng_e = np.random.default_rng(DATA_SEED * 100003 + 7 * i + int(m * 100) + 1)
        train += [(make_utterance(p, rng_t, m, False), i)
                  for _ in range(POOL_PER_LEAF)]
        ev += [(make_utterance(p, rng_e, m, True), i)
               for _ in range(EVAL_PER_LEAF)]
    return {"train": train, "eval": ev}


_REG = None  # active leaf-path registry (set via set_depth)


def set_depth(depth):
    global _REG
    _REG = leaf_paths(depth)


def class_at(depth, leaf_idx):
    """Depth-d class tuple for a leaf index, in the ACTIVE registry
    (default shallow 24-leaf; set_depth(6) switches to 96-leaf)."""
    reg = _REG if _REG is not None else leaf_paths(4)
    return tuple(reg[leaf_idx][:depth])
