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
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT of OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from itertools import islice
import torch
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset


class SubsetLoader(IterableDataset):
    """
    ### MODIFIED ###
    This class now contains a robust and corrected implementation for creating batches
    from a continuous data stream. The previous flawed logic has been replaced.
    """

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        tokenizer: AutoTokenizer = None,
        dataset_iterator = None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.dataset_iterator = dataset_iterator
        self.buffer = [] # Token buffer

    def __iter__(self):
        return self

    def __next__(self):
        """
        This method creates a single batch of data. It pulls from the dataset_iterator
        to fill its token buffer as needed and correctly handles the end of the stream.
        """
        batch_input_tensors = []
        batch_label_tensors = []

        # This loop gathers enough samples to create one full batch.
        for _ in range(self.batch_size):
            # This inner loop ensures the buffer has enough tokens for at least one sequence.
            while len(self.buffer) < self.sequence_length + 1:
                try:
                    # Pull the next raw sample from the finite iterator.
                    next_sample = next(self.dataset_iterator)
                    content = next_sample["text"]
                    
                    # Tokenize the sample and add it to our buffer.
                    tokenized = self.tokenizer(content, truncation=True, max_length=self.sequence_length)["input_ids"]
                    tokenized.append(self.tokenizer.eos_token_id)
                    self.buffer.extend(tokenized)

                except StopIteration:
                    # This is triggered when the finite iterator (e.g., 3500 samples) is exhausted.
                    # If we have already collected some samples for a partial batch, we should return them.
                    if batch_input_tensors:
                        bt.logging.info(f"Stream exhausted. Returning a final partial batch of size {len(batch_input_tensors)}.")
                        return torch.stack(batch_input_tensors), torch.stack(batch_label_tensors)
                    # Otherwise, if there's no partial batch, we are completely done.
                    else:
                        raise StopIteration
        
            # If we are here, the buffer has enough tokens to create one sequence.
            input_seq = self.buffer[:self.sequence_length]
            label_seq = self.buffer[1:self.sequence_length + 1]

            # Consume the tokens for this sequence from the buffer.
            self.buffer = self.buffer[self.sequence_length:]

            batch_input_tensors.append(torch.tensor(input_seq))
            batch_label_tensors.append(torch.tensor(label_seq))

        # If the loop completes, we have a full batch.
        return torch.stack(batch_input_tensors), torch.stack(batch_label_tensors)


class DatasetLoader(SubsetLoader):
    name: str = "HuggingFaceFW/fineweb-edu"
    logger = bt.logging
    
    # This global iterator ensures we only have one continuous stream open.
    _global_dataset_iterator = None

    @classmethod
    def initialize_stream(cls, seed=None):
        """
        One-time initialization of the global dataset stream.
        This should be called once when the miner starts.
        """
        if cls._global_dataset_iterator is None:
            bt.logging.info("Initializing global dataset stream (one-time cold start)...")
            dataset = load_dataset(cls.name, streaming=True, split="train")
            shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=10000)
            cls._global_dataset_iterator = iter(shuffled_dataset)
            bt.logging.info("Global dataset stream initialized.")

    @classmethod
    def from_stream(cls, batch_size, sequence_length, tokenizer, num_samples):
        """
        Creates a new dataset instance that will yield a specific number of samples
        from the global stream. This is called in each loop of the miner.
        """
        if cls._global_dataset_iterator is None:
            raise Exception("Global stream not initialized. Call DatasetLoader.initialize_stream() first.")
        
        # Create a finite iterator for this specific batch of work
        finite_iterator = islice(cls._global_dataset_iterator, num_samples)
        
        # Return an instance that can be iterated over
        return cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            dataset_iterator=finite_iterator,
        )

# The old methods are not included here for brevity but can remain in your file for reference.
