import random

import torch
import torch.nn as nn

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from src.data.data import (
    CHAMPION_ITOS,
    CHAMPION_STOI,
    ITEMS_AUGMENTS_ITOS,
    ITEMS_AUGMENTS_STOI,
)


class CompDataset(Dataset):
    def __init__(self, file_path: str, transform: bool = True):
        self.transform = transform
        self.boards, self.augments, self.placements = torch.load(file_path)

    def __len__(self):
        return len(self.boards)

    def _augment(
        self,
        boards: torch.Tensor,
        lower_board: torch.Tensor,
        lower_augment: torch.Tensor,
    ):
        def board_mask():
            mask = torch.randint(0, 2, (boards.shape[-2],), dtype=torch.bool)
            mask &= torch.randint(0, 2, (boards.shape[-2],), dtype=torch.bool)
            return ~mask

        while True:
            champion_mask = board_mask().unsqueeze(-1)
            if (lower_board * champion_mask).sum():
                break

        if random.random() < 0.5:
            lower_board *= champion_mask

            tier_mask = board_mask()
            lower_board[..., 1] = (
                (lower_board[..., 1] > 0)
                * tier_mask
                * (lower_board[..., 1] - 1).clamp(min=1)
            ) + (~tier_mask * lower_board[..., 1])

            item_mask = torch.stack((board_mask(), board_mask(), board_mask()), dim=-1)
            lower_board[..., 2:] = (
                lower_board[..., 2:] * (lower_board[..., 2:] > 0) * item_mask
            ) + (~item_mask * lower_board[..., 2:])

            augment_dropout = torch.randint(0, 2, (3,))
            lower_augment *= augment_dropout

        return lower_board, lower_augment

    def __getitem__(self, index):
        boards, augments, placements = (
            self.boards[index].clone(),
            self.augments[index].clone(),
            self.placements[index].clone(),
        )

        idx1, idx2 = random.choices(range(boards.shape[0]), k=2)

        board1 = boards[idx1]
        augment1 = augments[idx1]
        placement1 = placements[idx1]

        if idx1 == idx2:
            board2 = boards[idx2].clone()
            augment2 = augments[idx2].clone()
            placement2 = placements[idx2].clone()
        else:
            board2 = boards[idx2]
            augment2 = augments[idx2]
            placement2 = placements[idx2]

        if placement1 > placement2:
            reward = 0

            lower_board = board2
            lower_augment = augment2

        elif placement1 < placement2:
            reward = 1

            lower_board = board1
            lower_augment = augment1

        else:
            reward = 2

            if reward == 0:
                lower_board = board2
                lower_augment = augment2

            elif reward == 1:
                lower_board = board1
                lower_augment = augment1

        if self.transform and reward < 2:
            lower_board, lower_augment = self._augment(
                boards, lower_board, lower_augment
            )

        reward = torch.tensor(reward)

        return board1, board2, augment1, augment2, reward

    def get_batch(self, batch_size: int):
        batch = [self[i] for i in random.choices(range(len(self)), k=batch_size)]
        return [torch.stack([batch[i][j] for i in range(batch_size)]) for j in range(5)]


class MyModel(torch.nn.Module):
    def __init__(self, model_dim: int = 256):
        super().__init__()

        self.character_emb = nn.Embedding(
            len(CHAMPION_STOI),
            model_dim,
            padding_idx=0,
        )
        self.tier_emb = nn.Embedding(
            4,
            model_dim,
            padding_idx=0,
        )
        self.item_augment_emb = nn.Embedding(
            len(ITEMS_AUGMENTS_STOI),
            model_dim,
            padding_idx=0,
        )

        self.cls_emb = nn.Embedding(1, model_dim)
        self.entity_augment_emb = nn.Embedding(3, model_dim, padding_idx=0)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=2,
                dropout=0,
                dim_feedforward=(2 * model_dim),
                norm_first=True,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.pred = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim),
            # nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 3),
        )

    def forward(self, boards: torch.Tensor, augments: torch.Tensor):
        B, T, *_ = boards.shape

        cls = torch.zeros_like(boards[..., 0, 0])

        boards = boards
        augments = augments

        character = boards[..., 0]
        tier = boards[..., 1]
        items = boards[..., 2:]

        character_emb = self.character_emb(character)
        tier_emb = self.tier_emb(tier)
        items_emb = self.item_augment_emb(items)

        entities_emb = character_emb + tier_emb + items_emb.sum(-2)

        entities_emb = torch.cat(
            (entities_emb, self.item_augment_emb(augments)), dim=-2
        )

        entity_augment = torch.zeros_like(entities_emb[..., 0], dtype=torch.long)
        entity_augment[..., -3:] = 1
        entity_augment_emb = self.entity_augment_emb(entity_augment)

        entities_emb = entities_emb + entity_augment_emb
        entities_emb = torch.cat(
            (self.cls_emb(cls).unsqueeze(-2), entities_emb), dim=-2
        )
        entities_emb = entities_emb.flatten(0, 1)

        mask = torch.cat(
            (
                torch.zeros_like(character[..., 0, None]).bool(),
                (character == 0).bool(),
                (augments == 0).bool(),
            ),
            dim=-1,
        ).flatten(0, 1)

        entities_emb = self.encoder(entities_emb, src_key_padding_mask=mask)
        board_emb = torch.mean(entities_emb, 1)

        board1_emb, board2_emb = board_emb.view(B, 2, -1).chunk(2, 1)
        board_emb = torch.cat(
            (board1_emb, board2_emb, board1_emb - board2_emb), -1
        ).squeeze(1)

        return self.pred(board_emb)


def main():
    # Define hyperparameters
    batch_size = 32
    learning_rate = 2e-5
    num_steps = 5000

    training_data_path = "src/training_data.pt"
    validation_data_path = "src/training_data.pt"

    # Define the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset and dataloader
    training_dataset = CompDataset(training_data_path)
    validation_dataset = CompDataset(validation_data_path, transform=False)

    # Define the model and optimizer
    model = MyModel()
    # model.load_state_dict(torch.load("weights.pt"))
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate  # , weight_decay=1e-2
    )

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    train_buffer = []
    val_buffer = []
    progress = tqdm()

    # Train the model
    for step in range(1, num_steps + 1):
        (board1, board2, augment1, augment2, reward) = training_dataset.get_batch(
            batch_size
        )

        # Backward pass and optimize
        optimizer.zero_grad(set_to_none=True)

        board1 = board1.to(device)
        board2 = board2.to(device)
        augment1 = augment1.to(device)
        augment2 = augment2.to(device)
        reward = reward.to(device)

        # Forward pass
        prediction = model(
            torch.stack((board1, board2), dim=1),
            torch.stack((augment1, augment2), dim=1),
        )

        loss = criterion(prediction.squeeze(), reward.long())

        train_buffer.append(loss.item())
        if len(train_buffer) >= 100:
            train_buffer.pop(0)

        avg_training_loss = sum(train_buffer) / len(train_buffer)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            (board1, board2, augment1, augment2, reward) = validation_dataset.get_batch(
                batch_size
            )

            model.eval()

            board1 = board1.to(device)
            board2 = board2.to(device)
            augment1 = augment1.to(device)
            augment2 = augment2.to(device)
            reward = reward.to(device)

            # Forward pass
            prediction = model(
                torch.stack((board1, board2), dim=1),
                torch.stack((augment1, augment2), dim=1),
            )
            val_loss = criterion(prediction.squeeze(), reward.long())

            model.train()

        val_buffer.append(val_loss.item())
        if len(val_buffer) >= 100:
            val_buffer.pop(0)

        avg_val_loss = sum(val_buffer) / len(val_buffer)

        # Print training progress
        progress.set_description(
            "Step [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}".format(
                step,
                num_steps,
                avg_training_loss,
                avg_val_loss,
            )
        )
        progress.update(1)

        if step % 100 == 0:
            print("saving model...")
            torch.save(model.state_dict(), "weights.pt")
            print("model saved")


if __name__ == "__main__":
    main()
