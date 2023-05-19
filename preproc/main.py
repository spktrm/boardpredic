import json
import torch

from tqdm.auto import tqdm

from typing import List, Tuple

from src.data.data import (
    CHAMPION_ITOS,
    CHAMPION_STOI,
    ITEMS_AUGMENTS_ITOS,
    ITEMS_AUGMENTS_STOI,
)


def process_unit(unit: dict) -> torch.Tensor:
    character_id = unit["character_id"]
    item_names = unit["itemNames"]
    item_names += (3 - len(item_names)) * [None]
    tier = unit["tier"]

    unit_tensor = torch.tensor(
        [
            CHAMPION_STOI[character_id],
            tier,
            ITEMS_AUGMENTS_STOI[item_names[0]],
            ITEMS_AUGMENTS_STOI[item_names[1]],
            ITEMS_AUGMENTS_STOI[item_names[2]],
        ],
        dtype=torch.long,
    )

    return unit_tensor


def process_participant(participant: dict) -> Tuple[torch.Tensor, torch.Tensor, int]:
    placement = participant.get("placement")
    augments = participant["augments"] 
    augments += (3 - len(augments)) * [None]
    units = participant["units"]

    augment_tensor = torch.tensor(
        [ITEMS_AUGMENTS_STOI[augment] for augment in augments], dtype=torch.long
    )

    unit_tensors = []
    for unit in units:
        unit_tensors.append(process_unit(unit))

    unit_tensors = torch.stack(unit_tensors)
    unit_padding = torch.zeros((15 - len(unit_tensors), 5), dtype=torch.long)
    unit_tensors = torch.cat((unit_tensors, unit_padding))

    return unit_tensors, augment_tensor, placement


def process_match(match: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match_board_tensors = []
    match_augment_tensors = []
    match_placement_tensors = []

    info = match["info"]
    participants = info["participants"]

    for participant in participants:
        if participant["units"]:
            unit_tensors, augment_tensor, placement = process_participant(participant)

        match_board_tensors.append(unit_tensors)
        match_augment_tensors.append(augment_tensor)
        match_placement_tensors.append(placement)

    match_board_tensors = torch.stack(match_board_tensors)
    match_augment_tensors = torch.stack(match_augment_tensors)
    match_placement_tensors = torch.tensor(match_placement_tensors)

    return match_board_tensors, match_augment_tensors, match_placement_tensors


def main():
    with open("src/data/matches.json", "r") as f:
        matches = json.load(f)

    data_board_tensors = []
    data_augment_tensors = []
    data_placement_tensors = []

    for match in tqdm(matches):
        (
            match_board_tensors,
            match_augment_tensors,
            match_placement_tensors,
        ) = process_match(match)

        data_board_tensors.append(match_board_tensors)
        data_augment_tensors.append(match_augment_tensors)
        data_placement_tensors.append(match_placement_tensors)

    data_board_tensors = torch.stack(data_board_tensors)
    data_augment_tensors = torch.stack(data_augment_tensors)
    data_placement_tensors = torch.stack(data_placement_tensors)

    ratio = 0.85
    cutoff = int(data_board_tensors.shape[0] * ratio)

    torch.save(
        [
            data_board_tensors[:cutoff],
            data_augment_tensors[:cutoff],
            data_placement_tensors[:cutoff],
        ],
        "src/training_data.pt",
    )
    torch.save(
        [
            data_board_tensors[cutoff:],
            data_augment_tensors[cutoff:],
            data_placement_tensors[cutoff:],
        ],
        "src/validation_data.pt",
    )


if __name__ == "__main__":
    main()
