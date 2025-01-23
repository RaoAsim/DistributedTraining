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
from datasets import load_dataset, load_from_disk

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# Modified version of https://github.com/RaoFoundation/pretraining/blob/main/pretrain/dataset.py
class DataLoader(IterableDataset):
    max_rows: int = 10_800_000

    def __init__(
        self, batch_size, sequence_length, download_complete, rows: typing.List[int], tokenizer=tokenizer
    ):
        bt.logging.info("Initializing DataLoader")
        start_time = time.time()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.base_url = "http://194.68.245.137:22154/get_batch"
        self.params = { }
        self.rows = rows
        self.buffer = []
        self.dataset=None
        self.download_complete=download_complete
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries
        self.fetch_data_for_page(min(self.rows), len(self.rows))

        self.total_batches = len(self.buffer) // (
            self.sequence_length * self.batch_size
        )
        bt.logging.info(f"DataLoader initialized in {time.time() - start_time:.2f} seconds")

    def fetch_data_for_page(self, offset, length):
        iterations = math.ceil(length / 100)
        all_texts = []  # To store all texts fetched from HTTP requests

        # Step 1: Fetch all data in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=9) as executor:  # Adjust workers as needed
            futures = []
            for iteration in range(iterations):
                iter_offset = offset + (iteration * 100)
                iter_length = min(100, length - (iteration * 100))

                # Submit tasks for parallel execution
                futures.append(
                    executor.submit(self._fetch_data, iter_offset, iter_length)
                )

            # Collect all texts from the responses
            for future in as_completed(futures):
                try:
                    texts = future.result()
                    all_texts.extend(texts)
                except Exception as e:
                    bt.logging.error(f"Error during data fetch: {e}")

        # Step 2: Tokenize all texts in parallel
        bt.logging.info(f"http time {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=7) as executor:  # Adjust workers as needed
            futures = [
                executor.submit(self._tokenize_text, text) for text in all_texts
            ]

            # Collect tokenized results and extend the buffer
            for future in as_completed(futures):
                try:
                    tokens = future.result()
                    self.buffer.extend(tokens + [self.tokenizer.eos_token_id])
                except Exception as e:
                    bt.logging.error(f"Error during tokenization: {e}")


    
    def load_dataset_from_disk(self):
        if not self.download_complete:
            print("Dataset not downloaded yet.")
        if self.dataset is None:
            try:
                self.dataset = load_from_disk(self.dataset_path)
            except Exception as e:
                print("error")

    def get_batch(self,offset: int, length: int):
        if not self.download_complete:
            return 
        try:
            if self.dataset is None:
                self.load_dataset_from_disk()

            rows = []
            rows = [
                {
                    "row_idx": idx,
                    "row": row,
                    "truncated_cells": []  # Assuming no truncation
                }
                for idx, row in enumerate(self.dataset.select(range(offset, offset + length)))
            ]
                

            # Define the features
            features = [
                {"feature_idx": idx,"name": name,"type": { "_type": _type,"dtype": dtype} if _type != "Sequence" else {"_type": _type,"feature": {"_type": "Value", "dtype": dtype}}}
                for idx, (name, dtype, _type) in enumerate([
                    ("text", "string", "Value"),
                    ("id", "string", "Value"),
                    ("dump", "string", "Value"),
                    ("url", "string", "Value"),
                    ("file_path", "string", "Value"),
                    ("language", "string", "Value"),
                    ("language_score", "float64", "Value"),
                    ("token_count", "int64", "Value"),
                    ("score", "float64", "Value"),
                    ("int_score", "int64", "Value"),
                    ("embedding", "float32", "Sequence"),
                    ("count", "int64", "Value"),
                ])
            ]
            

            #
            response = {
                "features": features,
                "rows": rows,
                "num_rows_total":10800000,
                "num_rows_per_page":length,
                "partial":False
            }

            return response
        except Exception as e:
            print(f"Unhandled error in /get_batch: {e}")
       

    def _fetch_data(self, offset, length):
        """Helper method to fetch data from the API."""
        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = self.get_batch(offset, length)

                # Extract texts from the response
                texts = [row["row"]["text"] for row in response["rows"]]
                return texts

            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(f"Failed to fetch data. Attempt {attempt}/{self.retry_limit}. Error: {e}")
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    bt.logging.error("Maximum retry limit reached. Unable to fetch data.")

    def _tokenize_text(self, text):
        """Helper method to tokenize a single text."""
        return self.tokenizer(text, truncation=True, return_attention_mask=False)["input_ids"]

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
