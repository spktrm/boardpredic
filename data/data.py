import json


with open("src/data/en_au.json", "r") as f:
    datum = json.load(f)

SET = "8"
CHAMPIONS = [None] + [i["apiName"] for i in datum["sets"][SET]["champions"]]
ITEMS_AUGMENTS = [None] + [i["apiName"] for i in datum["items"]]

CHAMPION_STOI = {k: i for i, k in enumerate(CHAMPIONS)}
CHAMPION_ITOS = {v: k for k, v in CHAMPION_STOI.items()}

ITEMS_AUGMENTS_STOI = {k: i for i, k in enumerate(ITEMS_AUGMENTS)}
ITEMS_AUGMENTS_ITOS = {v: k for k, v in ITEMS_AUGMENTS_STOI.items()}
