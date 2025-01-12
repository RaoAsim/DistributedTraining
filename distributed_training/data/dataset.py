# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
import time
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
import bittensor as bt
import requests
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# Modified version of https://github.com/RaoFoundation/pretraining/blob/main/pretrain/dataset.py
class DataLoader(IterableDataset):
    max_rows: int = 10_800_000

    def __init__(
        self, batch_size, sequence_length, rows: typing.List[int], tokenizer=tokenizer
    ):
        bt.logging.info("Initializing DataLoader")
        start_time = time.time()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "airtrain-ai/fineweb-edu-fortified",
            "config": "CC-MAIN-2013-20",
            "split": "train",
        }
        self.rows = rows
        self.buffer = []
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries
        self.fetch_data_for_page(min(self.rows), len(self.rows))

        self.total_batches = len(self.buffer) // (
            self.sequence_length * self.batch_size
        )
        bt.logging.info(f"DataLoader initialized in {time.time() - start_time:.2f} seconds")

    def fetch_data_for_page(self, offset, length):
    
        iterations = math.ceil(length / 100)
        tasks = []  # List to hold tasks for parallel execution
    
        # ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=3) as executor:  # Two parallel tasks
            for iteration in range(iterations):
                iter_offset = offset + (iteration * 100)
                iter_length = min(100, length - (iteration * 100))
                
                # Submit tasks for parallel execution
                tasks.append(executor.submit(self._fetch_and_process, iter_offset, iter_length))
    
            # Process results as they complete
            for future in as_completed(tasks):
                try:
                    future.result()  # Raises any exception encountered during the task
                except Exception as e:
                    bt.logging.error(f"Error during data fetch: {e}")
        
    
    def _fetch_and_process(self, offset, length):
        """Helper method to handle individual fetch and processing."""
        attempt = 0
        while attempt < self.retry_limit:
            try:
                # Measure HTTP request time
                params = self.params.copy()
                params.update({"offset": offset, "length": length})
                response = requests.get(self.base_url, params=params)
    
                response.raise_for_status()
    
                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    self.buffer += self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += [self.tokenizer.eos_token_id]
                break  # Exit retry loop if successful
    
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(f"Failed to fetch data. Attempt {attempt}/{self.retry_limit}. Error: {e}")
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    bt.logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise
    def __len__(self):
        return self.total_batches

    def __iter__(self):
    
        while len(self.buffer) >= self.sequence_length * self.batch_size:
            batch = []
            label = []
    
            for _ in range(self.batch_size):
                # Tokenization and padding
                if len(self.buffer[: self.sequence_length]) != self.sequence_length:
                    batch.append(
                        torch.tensor(
                            self.buffer[: self.sequence_length]
                            + [self.tokenizer.eos_token_id] * (
                                self.sequence_length - len(self.buffer[: self.sequence_length])
                            )
                        )
                    )
                else:
                    batch.append(torch.tensor(self.buffer[: self.sequence_length]))
    
                # Same for labels
                if len(self.buffer[: self.sequence_length]) != self.sequence_length:
                    label.append(
                        torch.tensor(
                            self.buffer[1 : self.sequence_length + 1]
                            + [self.tokenizer.eos_token_id] * (
                                self.sequence_length - len(self.buffer[1 : self.sequence_length + 1])
                            )
                        )
                    )
                else:
                    label.append(torch.tensor(self.buffer[1 : self.sequence_length + 1]))
    
                self.buffer = self.buffer[self.sequence_length:]  # Slice buffer
            
            yield torch.stack(batch), torch.stack(label)



    def __next__(self):
        bt.logging.info("Fetching next batch")
        batch, label = [], []
    
        for _ in range(self.batch_size):
            if len(self.buffer[: self.sequence_length]) != self.sequence_length:
                batch.append(
                    torch.tensor(
                        self.buffer[: self.sequence_length]
                        + [self.tokenizer.eos_token_id] * (
                            self.sequence_length - len(self.buffer[: self.sequence_length])
                        )
                    )
                )
            else:
                batch.append(torch.tensor(self.buffer[: self.sequence_length]))
    
            if len(self.buffer[: self.sequence_length]) != self.sequence_length:
                label.append(
                    torch.tensor(
                        self.buffer[1 : self.sequence_length + 1]
                        + [self.tokenizer.eos_token_id] * (
                            self.sequence_length - len(self.buffer[1 : self.sequence_length + 1])
                        )
                    )
                )
            else:
                label.append(torch.tensor(self.buffer[1 : self.sequence_length + 1]))
    
            self.buffer = self.buffer[self.sequence_length:]  # Slice buffer
            
    
        yield torch.stack(batch), torch.stack(label)
