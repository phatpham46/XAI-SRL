import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from mlm_utils.model_utils import NUM_CPU

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
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        token_id = torch.tensor(sample['token_id'], dtype=torch.long)
        attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(sample['token_type_ids'], dtype=torch.long)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        return token_id, attention_mask, token_type_ids, labels
    
  
    def _get_sampler(self, local_rank):
        if local_rank == -1:
            return RandomSampler(self.data)
        else:
            return SequentialSampler(self.data)
        
    def generate_batches(self, local_rank, batch_size, dataset,
        drop_last=True):
        """
        A generator function which wraps the PyTorch DataLoader. It will
        ensure each tensor is on the write device location.
        """
       
        dataloader = DataLoader(
            dataset= dataset, 
            sampler= self._get_sampler(local_rank),
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=NUM_CPU)  

        return dataloader    

