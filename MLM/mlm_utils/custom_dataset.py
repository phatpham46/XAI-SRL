import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

class CustomDataset(Dataset):
    
    def __init__(self, data_path, file_name):
        self.data = []
        with open(data_path / file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_dict = json.loads(line)
                data_dict['token_id'] = json.loads(data_dict['token_id'])
                data_dict['attention_mask'] = json.loads(data_dict['attention_mask'])
                data_dict['token_type_ids'] = json.loads(data_dict['token_type_ids'])
                data_dict['labels'] = json.loads(data_dict['labels'])
                self.data.append(data_dict)
        self.data = self.data[:500]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # token_id = torch.from_numpy(np.array(sample['token_id'], dtype=np.int64))
        token_id = torch.tensor(sample['token_id'], dtype=torch.long)
        attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(sample['token_type_ids'], dtype=torch.long)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        return token_id, attention_mask, token_type_ids, labels
   
    # def read_csv(self, data_path, file_name):
    #     with open(data_path / file_name, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             data_dict = json.loads(line)
    #             data_dict['input_ids'] = json.loads(data_dict['token_id'])
    #             data_dict['input_masks'] = json.loads(data_dict['attention_mask'])
    #             #data_dict['token_type_ids'] = json.loads(data_dict['token_type_ids'])
    #             data_dict['lm_label_ids'] = json.loads(data_dict['labels'])
    #             self.data.append(data_dict)
    #     return self.data

