import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


class BugReportsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        data_summary = data_row.summary
        label = data_row.label

        encoding = self.tokenizer.encode_plus(
            data_summary,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        return dict(
            ticket_text=data_summary,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor([label])
        )