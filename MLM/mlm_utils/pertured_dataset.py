import json
import torch
from MLM.mlm_utils.model_utils import NUM_CPU
from torch.utils.data import Dataset, DataLoader, SequentialSampler

class PerturedDataset(Dataset):
    
    def __init__(self, file_name, device):
        self.device = device
      
        self.data = []
        with open( file_name, 'r') as file:
            for i, line in enumerate(file):
                sample = json.loads(line)
                self.data.append(sample)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if 'label' in sample: # origin
            origin_uid = sample['uid']
            label = torch.tensor(sample['label'], dtype=torch.long)
            token_id = torch.tensor(sample['token_id'], dtype=torch.long)
            type_id = torch.tensor(sample['type_id'], dtype=torch.long)
            mask = torch.tensor(sample['mask'], dtype=torch.long)
            return origin_uid, label, token_id, type_id, mask
        else:  # masked
            origin_uid = sample['origin_uid'] 
            origin_id = torch.tensor(sample['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long)
            token_type_ids = torch.tensor(sample['token_type_ids'], dtype=torch.long)
            pos_tag_id = torch.tensor(sample['pos_tag_id'], dtype=torch.long)
            return origin_uid, origin_id, attention_mask, token_type_ids, pos_tag_id
    
    
    def generate_batches(self, dataset, batch_size):
        """
        A generator function which wraps the PyTorch DataLoader. It will
        ensure each tensor is on the write device location.
        """
       
        dataloader = DataLoader(
            dataset= dataset, 
            sampler= SequentialSampler(dataset),
            batch_size=batch_size,
            num_workers=NUM_CPU)  

        return dataloader  