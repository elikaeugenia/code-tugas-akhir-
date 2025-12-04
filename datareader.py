import pandas as pd
import numpy as np
import re
import os
import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset 
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import random 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')  
except LookupError: 
    nltk.download('stopwords')

INDONESIAN_STOPWORDS = set(stopwords.words('indonesian'))

def load_normalization_dict(json_path=None):
    # Abaikan json_path karena mengambil dari Hugging Face
    try:
        dataset = load_dataset("theonlydo/indonesia-slang", split="train")
        normalization_dict = {row['slang']: row['formal'] for row in dataset}
        return normalization_dict
    except Exception as e:
        print(f"Gagal load kamus dari HuggingFace: {e}")
        return {}

class ShopeeComment(Dataset):
    def __init__(
        self,
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        normalization_file="normalization_dict.json", 
        random_state=2025,
        split="train",
        fold=0,
        augmentasi_file="augmentasi.json",
        typo_prob=0.1,         
        swap_prob=0.1,         
        delete_prob=0.1,      
        synonym_prob=0.1,      
        phrase_prob=0.1, 
    ):
        
        # Menyimpan pengaturan probabilitas untuk berbagai jenis augmentasi data
        self.typo_prob = typo_prob         
        self.swap_prob = swap_prob         
        self.delete_prob = delete_prob      
        self.synonym_prob = synonym_prob    
        self.phrase_prob = phrase_prob      

        # Menyimpan pengaturan file dan parameter data
        self.file_path = file_path          
        self.folds_file = folds_file        
        self.random_state = random_state    
        self.split = split                  
        self.fold = fold   # Fold ke berapa yang digunakan untuk training/validasi

        # Memuat data augmentasi dari file JSON untuk digunakan pada proses augmentasi
        self.augmentasi_data = self.load_augmentasi(augmentasi_file)
        
        # Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Load Normalization Dictionary
        self.normalization_dict = load_normalization_dict(normalization_file)
        
        # Load Dataset
        self.load_data()
        # Membuat self.folds_indices yang berisi train_indices dan val_indices (fold 0-4)
        self.setup_folds()

        # Mempersiapkan self.indices yang berisi data yang akan di training yang akan dipilih berdasarkan 'split' dan 'fold'
        self.setup_indices()
    
    # Load Augmentation Data     
    def load_augmentasi(self, file_path):
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_path, file_path)
    
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File augmentasi tidak ditemukan di {full_path}")
            return {}  # Kembalikan dict kosong agar tetap bisa jalan

    def random_typo(self, text):
        words = text.split()
        if len(words) < 1:
            return text
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 1:
            char_list = list(word)
            i = random.randint(0, len(char_list) - 2)
            char_list[i], char_list[i+1] = char_list[i+1], char_list[i]
            words[idx] = ''.join(char_list)
        return ' '.join(words)
    
    def random_swap(self, text):
        word = text.split()
        if len(word) < 2:
            return text
        idx1, idx2 = random.sample(range(len(word)), 2)
        word[idx1], word[idx2] = word[idx2], word[idx1]
        return ' '.join(word)
    
    def random_delete(self, text):
        word = text.split()
        if len(word) <= 1:
            return text
        idx = random.randint(0, len(word) - 1)
        del word[idx]
        return ' '.join(word)
    
    def augment_text(self, text):
        # Phrase replace
        if random.random() < self.phrase_prob:
            for phrase, replacements in self.augmentasi_data.get("replace_phrases", {}).items():
                if phrase in text:
                    text = text.replace(phrase, random.choice(replacements))
        # Synonym replacement
        words = text.split()
        if random.random() < self.synonym_prob:
            for i, word in enumerate(words):
                if word in self.augmentasi_data.get("synonyms", {}):
                    words[i] = random.choice(self.augmentasi_data["synonyms"][word])
        text = ' '.join(words)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in INDONESIAN_STOPWORDS]
        text = ' '.join(tokens)
        # Typo
        if random.random() < self.typo_prob:
            text = self.random_typo(text)
        # Swap
        if random.random() < self.swap_prob:
            text = self.random_swap(text)
        # Delete
        if random.random() < self.delete_prob:
            text = self.random_delete(text)
                
        return text
        
    def __len__(self):
        # Mengembalikan panjang data
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Hanya mengambil nomor indeks dari data yang akan diambil
        idx = self.indices[idx]
        
        # Mengambil data komentar dari rating
        komentar = str(self.df.iloc[idx]['comment'])
        rating = self.df.iloc[idx]['rating']
        # Mengubah rating menjadi label biner
        if rating >3:
            label = 1
        else:
            label = 0
        
        # Melakukan Pre-Processing
        comment_processed = self.preprocess_text(komentar)
        
        # Tokenisasi
        encoding = self.tokenizer(
            comment_processed,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        data = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'original_text': komentar,
            'processed_text': comment_processed,
            'original_rating': rating,
            'original_index': idx,
        }
        
        return data
    
    def preprocess_text(self, text):
        # CASEFOLDING : konversi ke huruf kecil
        text = text.lower()
        # CLEANSING : hapus url
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # CLEANSING : menghapus special karakter
        text = re.sub(r'[^\w\s]', '', text)
        # Menghapus spasi berlebih
        text = re.sub(r'\s+', ' ', text)
        # Replacement : mengganti kata dengan sinonim
        for phrase, replacements in self.augmentasi_data.get("replace_phrases", {}).items():
            if phrase in text:
                text = text.replace(phrase, random.choice(replacements))
        # Tokenisasi
        words = nltk.word_tokenize(text)
        # Normalization : dengan kamus dari file JSON
        words = [self.normalization_dict.get(word, word) for word in words]
        # STOPWORDS : menghapus kata yang tidak penting
        words = [word for word in words if word not in INDONESIAN_STOPWORDS]
        # Menggabungkan kembali kata-kata yang sudah di tokenisasi
        text = ' '.join(words)
        # Augmentasi : typo, swap, delete
        text = self.augment_text(text)
        
        return text             
        
    def setup_indices(self):
        # Mempersiapkan indices untuk data yang akan di training
        fold_key = f"fold_{self.fold}"
        
        if self.split == "train" :
            self.indices = self.folds[fold_key]['train_indices']
        else:
            self.indices = self.folds[fold_key]['val_indices']
       
    def setup_folds(self):
        # Check if folds file exists
        if os.path.exists(self.folds_file):
            self.load_folds()
        else: # Create folds if file does not exist
            self.create_folds()
            self.save_folds()
    
    # Load folds from JSON file
    def load_folds(self): 
        with open(self.folds_file, 'r') as f:
            folds_data = json.load(f)
            
        self.folds_indices = folds_data['fold_indices']
        self.folds = self.folds_indices
        print(f"Menggunakan {folds_data['n_folds']} folds dengan {folds_data['n_samples']} samples dataset") 
        
    # Create Stratified K-Folds
    def create_folds(self):
        print(f"Membuat 5-folds cross-validation dengan random state {self.random_state}") 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
     
        fold_indices = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['rating'])):
            fold_indices[f"fold_{fold}"] = {
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            }
        
        # Save fold indices to JSON file
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices' : fold_indices, 
                'n_samples' : len(self.df),
                'n_folds': 5,
                'random_state': self.random_state
            }, f)
            
        self.folds = fold_indices
        
    def save_folds(self):
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices' : self.folds,
                'n_samples' : len(self.df),
                'n_folds': 5,
                'random_state': self.random_state
            }, f)
    
    def load_data(self):
        self.df = pd.read_excel(self.file_path) # Load the dataset excel
        self.df.columns = ['userName', 'rating', 'timestamp', 'comment'] # Rename columns
        self.df = self.df.dropna(subset=['comment', 'rating']) # Drop rows with NaN in 'comment' and 'rating' column
        self.df['rating'] = self.df['rating'].astype(int) # Convert rating to int
        self.df = self.df[(self.df['rating'] >= 1) & (self.df['rating'] <= 5)] # Filter rating between 1 and 5

if __name__ == "__main__":
    dataset = ShopeeComment(fold=1, split="train") # Instansi kelas 
    random_index = random.randint(0, len(dataset) - 1) # Pilih indeks secara acak
    data = dataset[random_index] # Ambil data dengan indeks acak
    # data = dataset[163] # Ambil data pertama   
    print(f"Input IDs: {data['input_ids']}")
    print(f"Original Text: {data['original_text']}")
    print(f"Processed Text: {data['processed_text']}")
    print(f"Original Index: {data['original_index']}")
    print(f"Label (Rating): {data['labels']}")
