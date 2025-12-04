# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, random_split
from lightning_utilities.core.imports import RequirementCache

from litgpt.prompts import PromptStyle
from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.tokenizer import Tokenizer


@dataclass
class LIMA(DataModule):
    """LIMA data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.1
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/lima")
    """The directory in which the downloaded dataset gets saved."""
    include_multiturn_conversations: bool = False
    """Whether to include multi-turn conversations in the dataset."""
    repo_id: str = "GAIR/lima"
    """The Hugging Face dataset repository ID from where to download the data."""
    access_token: Optional[str] = "..."
    """The Hugging Face API token to use for authentication. Can also be set through the
    `HF_TOKEN` environment variable."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if self.access_token is None:
            raise ValueError(
                "LIMA requires authentication, please set the `HF_TOKEN=your_token` environment"
                " variable or pass --access_token=your_token. You can find your token by visiting"
                " https://huggingface.co/settings/tokens"
            )
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length


    def prepare_data(self) -> None:
        from datasets import load_dataset
        load_dataset(self.repo_id, token=self.access_token)


    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset
        dataset = load_dataset(self.repo_id, token=self.access_token)
        data = format_dataset(dataset["train"], self.include_multiturn_conversations)


        # Partition the dataset into train and test
        train_data, test_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )


def download_if_missing(file_path: Path, file_url: str, mode: str = "w", stream: bool = False) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    requests_available = RequirementCache("requests")
    if not requests_available:
        raise ModuleNotFoundError(str(requests_available))
    import requests

    response = requests.get(file_url, stream=stream)
    with open(file_path, mode, encoding=None if mode == "wb" else "utf-8") as f:
        if stream:
            # credit: https://github.com/karpathy/llama2.c/blob/b3c4b6/tinystories.py#L25-L38
            from tqdm import tqdm

            pbar = tqdm(
                desc=str(file_path),
                total=int(response.headers.get("content-length", 0)),
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
            pbar.close()
        else:
            f.write(response.text)


def format_dataset(dataset_partition: dict, include_multi_turn_conversations: bool) -> List[dict]:
    formatted_ds = []

    for entry in dataset_partition:
        convo = entry["conversations"]
        if include_multi_turn_conversations:
            for i in range(0, len(convo) - 1, 2):
                formatted_ds.append({"instruction": convo[i], "input": "", "output": convo[i + 1]})
        else:
            formatted_ds.append({"instruction": convo[0], "input": "", "output": convo[1]})

    return formatted_ds
