import random

import torch
import torch.nn as nn

from tqdm.auto import tqdm

from src.train.main import MyModel
from src.preproc.main import process_participant

from src.data.data import (
    CHAMPION_ITOS,
    CHAMPION_STOI,
    ITEMS_AUGMENTS_ITOS,
    ITEMS_AUGMENTS_STOI,
)


def main():
    # Define the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and optimizer
    model = MyModel()
    model.load_state_dict(torch.load("weights.pt"))
    model.to(device)
    model.eval()

    player1 = {
        "units": [
            {
                "character_id": "TFT8_Ezreal",
                "itemNames": [],
                "name": "",
                "rarity": 1,
                "tier": 1,
            },
            {
                "character_id": "TFT8_MissFortune",
                "itemNames": [
                    "TFT_Item_SpearOfShojin",
                    "TFT_Item_RabadonsDeathcap",
                    "TFT_Item_StatikkShiv",
                ],
                "name": "",
                "rarity": 4,
                "tier": 2,
            },
            {
                "character_id": "TFT8_EzrealFuture",
                "itemNames": [
                    "TFT_Item_GuardianAngel",
                    "TFT_Item_InfinityEdge",
                    "TFT_Item_HextechGunblade",
                ],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Janna",
                "itemNames": [],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Janna",
                "itemNames": [],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Urgot",
                "itemNames": ["TFT_Item_RabadonsDeathcap"],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Urgot",
                "itemNames": [],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Leona",
                "itemNames": [
                    "TFT_Item_WarmogsArmor",
                    "TFT_Item_BrambleVest",
                    "TFT_Item_TitansResolve",
                ],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Fiddlesticks",
                "itemNames": ["TFT_Item_Zephyr"],
                "name": "",
                "rarity": 6,
                "tier": 2,
            },
        ],
        "augments": [
            "TFT7_Augment_ClutteredMind",
            "TFT8_Augment_EzrealSupport",
            "TFT6_Augment_JeweledLotus",
        ],
    }
    board1, augment1, _ = process_participant(player1)

    player2 = {
        "units": [
            {
                "character_id": "TFT8_Nasus",
                "itemNames": [],
                "name": "",
                "rarity": 0,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Jinx",
                "itemNames": ["TFT_Item_Zephyr"],
                "name": "",
                "rarity": 1,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Vayne",
                "itemNames": [
                    "TFT_Item_GuinsoosRageblade",
                    "TFT_Item_MadredsBloodrazor",
                    "TFT_Item_PowerGauntlet",
                ],
                "name": "",
                "rarity": 1,
                "tier": 3,
            },
            {
                "character_id": "TFT8_Nilah",
                "itemNames": [],
                "name": "",
                "rarity": 2,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Riven",
                "itemNames": [
                    "TFT_Item_RedBuff",
                    "TFT_Item_WarmogsArmor",
                    "TFT_Item_DragonsClaw",
                ],
                "name": "",
                "rarity": 2,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Alistar",
                "itemNames": [],
                "name": "",
                "rarity": 2,
                "tier": 2,
            },
            {
                "character_id": "TFT8_MissFortune",
                "itemNames": [
                    "TFT_Item_Morellonomicon",
                    "TFT_Item_MadredsBloodrazor",
                    "TFT4_Item_OrnnZhonyasParadox",
                ],
                "name": "",
                "rarity": 4,
                "tier": 2,
            },
            {
                "character_id": "TFT8_Ekko",
                "itemNames": ["TFT_Item_ChainVest"],
                "name": "",
                "rarity": 4,
                "tier": 2,
            },
        ],
        "augments": [
            "TFT6_Augment_PortableForge",
            "TFT8_Augment_VayneCarry",
            "TFT6_Augment_TomeOfTraits1",
        ],
    }
    board2, augment2, _ = process_participant(player2)

    prediction = model(
        torch.stack((board1, board2)).unsqueeze(0).to(device),
        torch.stack((augment1, augment2)).unsqueeze(0).to(device),
    )

    def get_unit_str(unit):
        return (
            unit["character_id"].replace("TFT8_", "")
            + str(unit["tier"])
            + " "
            + " ".join([name for name in unit["itemNames"] if name])
        )

    for unit in player1["units"]:
        unit_str = get_unit_str(unit)
        print(unit_str)
    print()
    for unit in player2["units"]:
        unit_str = get_unit_str(unit)
        print(unit_str)
    print()
    print(prediction.cpu().squeeze().softmax(-1).tolist())


if __name__ == "__main__":
    main()
