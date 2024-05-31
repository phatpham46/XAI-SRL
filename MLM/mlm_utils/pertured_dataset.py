import json
import torch
from mlm_utils.model_utils import NUM_CPU
from torch.utils.data import Dataset, DataLoader, SequentialSampler

class PerturbedDataset(Dataset):
    
    def __init__(self, file_name, device):
        self.device = device
      
        self.data = []
        with open( file_name, 'r') as file:
            for i, line in enumerate(file):
                sample = json.loads(line)
                self.data.append(sample)
            
    def __len__(self):
        return len(self.data)
    
    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #     if 'label' in sample: # origin
    #         label = torch.tensor(sample['label'], dtype=torch.long)
            
    #         # dummy pos_tag_id
    #         pos_tag_id = torch.zeros(len(sample['token_id']), dtype=torch.long)
        
    #     elif 'masked_id' in sample and 'pos_tag_id' in sample: # mlm_output: dữ liệu có chứa token_mask: 103
    #         token_id = torch.tensor(sample['masked_id'], dtype=torch.long)
    #         pos_tag_id = torch.tensor(sample['pos_tag_id'], dtype=torch.long)
    #         label = torch.tensor(sample['label'], dtype=torch.long)
            
    #     else:  # masked for each method
    #         pos_tag_id = torch.tensor(sample['pos_tag_id'], dtype=torch.long)
            
    #         # dummy label
    #         label = torch.zeros(len(sample['token_id']), dtype=torch.long)
          
    #     origin_uid = sample['uid']
    #     token_id = torch.tensor(sample['token_id'], dtype=torch.long)
    #     type_id = torch.tensor(sample['type_id'], dtype=torch.long)
    #     mask = torch.tensor(sample['mask'], dtype=torch.long)
    #     return origin_uid, token_id, type_id, mask, label, pos_tag_id
    def __getitem__(self, idx):
        sample = self.data[idx]
       
        # Convert to tensor if exists, else create dummy tensor
        def to_tensor_or_dummy(key, shape, dtype, default_value=0):
            if key in sample:
                return torch.tensor(sample[key], dtype=dtype)
            else:
                return torch.full(shape, default_value, dtype=dtype)
        
        # Define the keys and their corresponding dummy shapes and types
        keys_info = {
            'label': (lambda: (len(sample['token_id']),), torch.long, 0),
            'masked_id': (lambda: (len(sample['token_id']),), torch.long, 0),
            'pos_tag_id': (lambda: (len(sample['token_id']),), torch.long, 0),
            'token_id': (lambda: (len(sample['token_id']),), torch.long, 0),
            'type_id': (lambda: (len(sample['token_id']),), torch.long, 0),
            'mask': (lambda: (len(sample['token_id']),), torch.long, 0)
        }
        
        tensors = {}
        for key, (shape_func, dtype, default_value) in keys_info.items():
            shape = shape_func()
            tensors[key] = to_tensor_or_dummy(key, shape, dtype, default_value)
        
        origin_uid = sample['uid']
        
        return (origin_uid, tensors['token_id'], tensors['type_id'], 
                tensors['mask'], tensors['label'], tensors['pos_tag_id'], tensors['masked_id'])
        
    
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