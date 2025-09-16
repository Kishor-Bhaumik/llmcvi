
import os
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def load_custom_dataset(dataset_name, root_dir="datasets"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, root_dir)

    train_path = os.path.join(root_dir, str(dataset_name), "train.csv")
    test_path  = os.path.join(root_dir, str(dataset_name), "test.csv")

    if dataset_name == "snippets":
        train_df = pd.read_csv(train_path, sep="\t", header=None, names=["label", "text"], usecols=[0, 1])
        test_df  = pd.read_csv(test_path,  sep="\t", header=None, names=["label", "text"], usecols=[0, 1])
    elif dataset_name in ['biomedical', 'stackoverflow']:
        train_df = pd.read_csv(train_path, sep="\t", header=None, names=["label", "text"], usecols=[0, 2])
        test_df  = pd.read_csv(test_path,  sep="\t", header=None, names=["label", "text"], usecols=[0, 2])

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df, preserve_index=False)
    dataset = DatasetDict({"train": train_ds, "test": test_ds})

    unique_labels = sorted(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))
    label2id = {old: new for new, old in enumerate(unique_labels)}
    dataset = dataset.map(lambda e: {"label": label2id[e["label"]]})

    class_label = ClassLabel(
        num_classes=len(unique_labels), names=[str(l) for l in unique_labels]
    )
    dataset = dataset.cast_column("label", class_label)

    return dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
        self.saved_data_dir = config['SAVED_DATA_DIR']
        os.makedirs(self.saved_data_dir, exist_ok=True)
        
    def prepare_data(self):
        # Download dataset
        if self.config['DATASET_NAME'] in ['snippets', 'biomedical', 'stackoverflow']:
            dataset = load_custom_dataset(self.config['DATASET_NAME'])
        elif self.config['DATASET_NAME'] == "20ng":
            dataset = load_dataset("SetFit/20_newsgroups")
        else:
            dataset = load_dataset(self.config['DATASET_NAME'])

    def setup(self, stage=None):
        if self.config['DATASET_NAME'] in ['snippets', 'biomedical', 'stackoverflow']:
            dataset = load_custom_dataset(self.config['DATASET_NAME'])
        elif self.config['DATASET_NAME'] == "20ng":
            dataset = load_dataset("SetFit/20_newsgroups")
        else:
            dataset = load_dataset(self.config['DATASET_NAME'])

        # Get full data
        train_texts_full = dataset['train']['text']
        train_labels_full = dataset['train']['label']
        test_texts_full = dataset['test']['text']
        test_labels_full = dataset['test']['label']
        
        # Update config with dataset info
        if self.config['DATASET_NAME'] == "20ng":
            self.config['num_labels'] = 20
            self.config['label_names'] = list(range(20))
        else:
            self.config['num_labels'] = dataset['train'].features['label'].num_classes
            self.config['label_names'] = dataset['train'].features['label'].names

        train_texts_list = list(train_texts_full)
        train_labels_list = list(train_labels_full)
        train_texts, _, train_labels, _ = train_test_split(
            train_texts_list, train_labels_list, 
            train_size=self.config.get('TRAIN_SUBSET_RATIO', 0.1), 
            stratify=train_labels_list, 
            random_state=42 )

        # For development, use smaller subsets
        # train_texts, _, train_labels, _ = train_test_split(
        #     train_texts_full, train_labels_full, 
        #     train_size=self.config.get('TRAIN_SUBSET_RATIO', 0.1), 
        #     stratify=train_labels_full, 
        #     random_state=42
        # )
        test_texts_list = list(test_texts_full)
        test_labels_list = list(test_labels_full)

        test_texts, _, test_labels, _ = train_test_split(
            test_texts_list, test_labels_list, 
            train_size=self.config.get('TEST_SUBSET_RATIO', 0.5), 
            stratify=test_labels_list, 
            random_state=42
        )
        # test_texts, _, test_labels, _ = train_test_split(
        #     test_texts_full, test_labels_full, 
        #     train_size=self.config.get('TEST_SUBSET_RATIO', 0.5), 
        #     stratify=test_labels_full, 
        #     random_state=42
        # )
        
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
