
import os
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['MODEL_NAME'])
        self.saved_data_dir = config['SAVED_DATA_DIR']
        os.makedirs(self.saved_data_dir, exist_ok=True)
        
    def prepare_data(self):
        # Download dataset
        dataset = load_dataset(self.config['DATASET_NAME'])
        
    def setup(self, stage=None):
        # Load dataset
        dataset = load_dataset(self.config['DATASET_NAME'])

        # Get full data
        train_texts_full = dataset['train']['text']
        train_labels_full = dataset['train']['label']
        test_texts_full = dataset['test']['text']
        test_labels_full = dataset['test']['label']
        
        # Update config with dataset info
        self.config['num_labels'] = dataset['train'].features['label'].num_classes
        self.config['label_names'] = dataset['train'].features['label'].names
        
        # For development, use smaller subsets
        train_texts, _, train_labels, _ = train_test_split(
            train_texts_full, train_labels_full, 
            train_size=self.config.get('TRAIN_SUBSET_RATIO', 0.1), 
            stratify=train_labels_full, 
            random_state=42
        )
        
        test_texts, _, test_labels, _ = train_test_split(
            test_texts_full, test_labels_full, 
            train_size=self.config.get('TEST_SUBSET_RATIO', 0.5), 
            stratify=test_labels_full, 
            random_state=42
        )
        
        # Split train into train/val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=0.2,
            stratify=train_labels,
            random_state=42
        )
        
        # Create datasets
        if stage in (None, 'fit'):
            self.train_dataset = self._create_dataset(train_texts, train_labels, 'train')
            self.val_dataset = self._create_dataset(val_texts, val_labels, 'val')
            
        if stage in (None, 'test'):
            self.test_dataset = self._create_dataset(test_texts, test_labels, 'test')
    
    def _create_dataset(self, texts, labels, split):
        """Create dataset with caching"""
        cache_path = os.path.join(
            self.saved_data_dir, 
            f"{self.config['DATASET_NAME']}_{self.config['MAX_LENGTH']}_{split}_subset_tokenized.pt"
        )
        
        if os.path.exists(cache_path):
            print(f"Loading cached {split} data...")
            data_saved = torch.load(cache_path, weights_only=True)
            return TensorDataset(
                data_saved['input_ids'], 
                data_saved['attention_masks'], 
                data_saved['labels']
            )
        else:
            print(cache_path,"==> did not find data!!")
            print(f"Tokenizing {split} data...")
            input_ids_list = []
            attention_masks_list = []
            
            for text in tqdm(texts, desc=f"Tokenizing {split}"):
                encoded_dict = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.config['MAX_LENGTH'],
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )
                input_ids_list.append(encoded_dict['input_ids'])
                attention_masks_list.append(encoded_dict['attention_mask'])
            
            input_ids_tensor = torch.cat(input_ids_list, dim=0)
            attention_masks_tensor = torch.cat(attention_masks_list, dim=0)
            labels_tensor = torch.tensor(labels)
            
            # Save to cache
            torch.save({
                'input_ids': input_ids_tensor,
                'attention_masks': attention_masks_tensor,
                'labels': labels_tensor
            }, cache_path)
            
            return TensorDataset(input_ids_tensor, attention_masks_tensor, labels_tensor)
    
 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config['BATCH_SIZE'], 
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config['BATCH_SIZE'], 
            shuffle=False,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config['BATCH_SIZE'], 
            shuffle=False,
            num_workers=4
        )
