import os
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
        self.saved_data_dir = config['SAVED_DATA_DIR']
        os.makedirs(self.saved_data_dir, exist_ok=True)
        
    def prepare_data(self):
        # Download dataset
        dataset = load_dataset(self.config['DATASET_NAME'])
        
    def _show_class_distribution(self, labels, title="Dataset Class Distribution"):
        """Display class distribution statistics"""
        label_counter = Counter(labels)
        total_samples = len(labels)
        
        print(f"\n=== {title} ===")
        print(f"Total samples: {total_samples:,}")
        print()
        
        # Sort by class label for consistent display
        for class_label in sorted(label_counter.keys()):
            count = label_counter[class_label]
            percentage = (count / total_samples) * 100
            class_name = self.config.get('label_names', [f"Class {i}" for i in range(len(set(labels)))])[class_label] if class_label < len(self.config.get('label_names', [])) else f"Class {class_label}"
            print(f"Class {class_label} ({class_name}): {count:,} samples ({percentage:.1f}%)")
        
        print("=" * 35)
        print()
    
    def _apply_few_shot_reduction(self, texts, labels):
        """Apply few-shot reduction based on config['few_shots']"""
        if 'few_shots' not in self.config:
            return texts, labels
            
        few_shots_config = self.config['few_shots']
        if not few_shots_config:
            return texts, labels
            
        print("Applying few-shot reduction...")
        
        # Convert to lists for easier manipulation
        texts_list = list(texts)
        labels_list = list(labels)
        
        # Group indices by class
        class_indices = {}
        for idx, label in enumerate(labels_list):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Apply few-shot reduction
        indices_to_keep = []
        
        for class_label, indices in class_indices.items():
            if class_label in few_shots_config:
                # Calculate how many samples to keep
                keep_percentage = few_shots_config[class_label]
                num_to_keep = int(len(indices) * (keep_percentage / 100.0))
                
                # Randomly sample indices to keep
                #np.random.seed(42)  # For reproducibility
                sampled_indices = np.random.choice(indices, size=num_to_keep, replace=False)
                indices_to_keep.extend(sampled_indices)
                
                print(f"Class {class_label}: Keeping {num_to_keep}/{len(indices)} samples ({keep_percentage}%)")
            else:
                # Keep all samples for classes not in few_shots config
                indices_to_keep.extend(indices)
                print(f"Class {class_label}: Keeping all {len(indices)} samples (100%)")
        
        # Sort indices to maintain original order
        indices_to_keep.sort()
        
        # Create reduced dataset
        reduced_texts = [texts_list[i] for i in indices_to_keep]
        reduced_labels = [labels_list[i] for i in indices_to_keep]
        
        print(f"Few-shot reduction complete: {len(reduced_texts)}/{len(texts_list)} samples retained")
        
        return reduced_texts, reduced_labels

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

        # Convert to lists for easier manipulation
        train_texts_list = list(train_texts_full)
        train_labels_list = list(train_labels_full)
        test_texts_list = list(test_texts_full)
        test_labels_list = list(test_labels_full)
        
        # Show original class distribution if requested
        if self.config.get('SHOW_CLASS_RATIO', False):
            self._show_class_distribution(train_labels_list, "Original Training Dataset Class Distribution")
            exit()
        
        # Apply few-shot reduction to full training data
        few_shot_texts, few_shot_labels = self._apply_few_shot_reduction(train_texts_list, train_labels_list)
        
        # Show class distribution after few-shot reduction if requested
        if self.config.get('SHOW_CLASS_RATIO', False):
            self._show_class_distribution(few_shot_labels, "After Few-Shot Reduction Class Distribution")
        
        # Split few-shot data into train/val with stratification to maintain proportions
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            few_shot_texts, few_shot_labels,
            test_size=0.2,  # 20% for validation
            stratify=few_shot_labels,  # Maintain few-shot proportions
            random_state=42)
        
        # For test data, use subset as before (or full test set based on config)
        if self.config.get('TEST_SUBSET_RATIO') and self.config.get('TEST_SUBSET_RATIO') < 1.0:
            test_texts, _, test_labels, _ = train_test_split(
                test_texts_list, test_labels_list, 
                train_size=self.config.get('TEST_SUBSET_RATIO', 0.5), 
                stratify=test_labels_list, 
                random_state=42
            )
        else:
            test_texts = test_texts_list
            test_labels = test_labels_list
        
        # Create datasets
        if stage in (None, 'fit'):
            self.train_dataset = self._create_dataset(train_texts, train_labels, 'train')
            self.val_dataset = self._create_dataset(val_texts, val_labels, 'val')
            
        if stage in (None, 'test'):
            self.test_dataset = self._create_dataset(test_texts, test_labels, 'test')
    
    def _create_dataset(self, texts, labels, split):
        """Create dataset with caching"""
        # Create cache filename that includes few-shot info for uniqueness
        few_shot_suffix = ""
        if 'few_shots' in self.config and self.config['few_shots']:
            # Create a string representation of few_shots for cache filename
            few_shot_str = "_".join([f"{k}_{v}" for k, v in sorted(self.config['few_shots'].items(), key=lambda x: str(x[0]))])
            few_shot_suffix = f"_fewshot_{few_shot_str}"
        
        cache_path = os.path.join(
            self.saved_data_dir, 
            f"{self.config['DATASET_NAME']}_{self.config['MAX_LENGTH']}_{split}{few_shot_suffix}_tokenized.pt"
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
            print(f"Cache not found: {cache_path}")
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
            num_workers=0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config['BATCH_SIZE'], 
            shuffle=False,
            num_workers=0
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config['BATCH_SIZE'], 
            shuffle=False,
            num_workers=0
        )