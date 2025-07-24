# %% tfidf
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import logging
import json
from typing import Dict, List, Any
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import torch

class NewsClassifier:
    def __init__(self, train_file_path: str, test_file_path: str, articles_dir: str, valid_file_path: str, use_gpu: bool = True):
        self.train_file_path = Path(train_file_path)
        self.test_file_path = Path(test_file_path)
        self.valid_file_path = Path(valid_file_path)  
        self.articles_dir = Path(articles_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.setup_logging()
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def setup_logging(self):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
            if self.use_gpu:
                self.logger.info("GPU 可用，將使用 GPU 加速")
                self.logger.info(f"使用的 GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("將使用 CPU 模式")

    def preprocess_text(self, text: str) -> str:
        """文本預處理：包括去除特殊字符、詞形還原和去除停用詞"""
        text = text.lower()
        text = re.sub(r'_{2,}', ' ', text)  # 去除連續兩個或以上的下劃線
        text = re.sub(r'-{2,}', ' ', text) # 去除連續兩個或以上的中劃線
        text = re.sub(r'\d+', '', text)  # 去除數字
        text = re.sub(r'[^\w\s]', '', text)  # 去除標點
        text = re.sub(r'\s+', ' ', text)  # 去除多餘空白
        text = re.sub(r"\*", "", text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)   
    
    def extract_text_from_json(self, content: List[Dict]) -> str:
        """從JSON內容中提取文本"""
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.extend(str(value) for value in item.values() if isinstance(value, (str, int, float)))
            elif isinstance(item, list):
                texts.append(self.extract_text_from_json(item))
            elif isinstance(item, (str, int, float)):
                texts.append(str(item))
        return ' '.join(texts)
    
    def load_json_content(self, json_file_name: str) -> str:
        """讀取並合併JSON文件中的文本內容"""
        try:
            full_path = self.articles_dir / json_file_name
            self.logger.info(f"正在讀取檔案: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                if isinstance(content, list):
                    text = self.extract_text_from_json(content)
                elif isinstance(content, dict):
                    text = ' '.join(str(value) for value in content.values() 
                                 if isinstance(value, (str, int, float)))
                else:
                    text = str(content)
                
                return self.preprocess_text(text)
        except Exception as e:
            self.logger.error(f"讀取文件出錯 {json_file_name}: {str(e)}")
            return ""
    
    def prepare_valid_data(self) -> pd.DataFrame:
        """準備驗證數據"""
        self.logger.info(f"開始讀取驗證資料: {self.valid_file_path}")
        
        with open(self.valid_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        for record in data:
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_data.append({
                        'id': record["metadata"]["id"],
                        'content': combined_content,
                        'rating': record["label"]["rating"]
                    })
                    self.logger.info(f"成功處理ID: {record['metadata']['id']}")
            except Exception as e:
                self.logger.error(f"處理驗證記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        df = pd.DataFrame(processed_data)
        self.logger.info(f"總共處理了 {len(df)} 筆驗證資料")
        return df

    def prepare_training_data(self) -> pd.DataFrame:
        """準備訓練數據，並合併訓練資料和驗證資料"""
        self.logger.info(f"開始讀取訓練資料: {self.train_file_path}")
        
        with open(self.train_file_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        processed_train_data = []
        for record in train_data:
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_train_data.append({
                        'id': record["metadata"]["id"],
                        'content': combined_content,
                        'rating': record["label"]["rating"]
                    })
                    self.logger.info(f"成功處理ID: {record['metadata']['id']}")
            except Exception as e:
                self.logger.error(f"處理訓練記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        df_train = pd.DataFrame(processed_train_data)
        self.logger.info(f"總共處理了 {len(df_train)} 筆訓練資料")
        
        # 合併訓練資料和驗證資料
        df_valid = self.prepare_valid_data()
        df_combined = pd.concat([df_train, df_valid], ignore_index=True)
        self.logger.info(f"總共處理了 {len(df_combined)} 筆訓練數據（包括驗證數據）")
        
        return df_combined

    def train_model(self):
        """訓練模型"""
        self.logger.info("開始訓練模型...")
        df = self.prepare_training_data()
        
        if len(df) == 0:
            self.logger.error("沒有可用的訓練數據！")
            return
        
        # 特徵提取
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        X = self.vectorizer.fit_transform(df['content'])
        y = df['rating']
        
        # 將特徵矩陣轉換為DMatrix格式
        dtrain = xgb.DMatrix(X, label=y)
        
        # 設定XGBoost參數
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': 6,
            'eta': 0.3,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'eval_metric': 'mlogloss',
            'nthread': 12
}

        # 訓練模型
        self.logger.info("開始XGBoost訓練...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=333,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=10
        )
        
        self.logger.info("模型訓練完成")

    def prepare_test_data(self) -> tuple:
        """準備測試數據"""
        self.logger.info(f"開始讀取測試資料: {self.test_file_path}")
        
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        ids = []
        for record in data:
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_data.append(combined_content)
                    ids.append(record["metadata"]["id"])
                    self.logger.info(f"成功處理測試ID: {record['metadata']['id']}")
            except Exception as e:
                self.logger.error(f"處理測試記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        self.logger.info(f"總共處理了 {len(processed_data)} 筆測試資料")
        return processed_data, ids


    def predict(self) -> List[int]:
        """預測測試數據"""
        self.logger.info("開始進行預測...")
        
        # 準備測試數據
        test_contents, test_ids = self.prepare_test_data()
        
        # 轉換特徵
        X_test = self.vectorizer.transform(test_contents)
        dtest = xgb.DMatrix(X_test)
        
        # 預測
        predictions = self.model.predict(dtest)
        predictions = predictions.astype(int)
        
        # 記錄預測結果
        self.logger.info("預測完成")
        self.logger.info(f"預測結果分布:\n{pd.Series(predictions).value_counts()}")
        
        # 將預測結果與ID配對
        results = list(zip(test_ids, predictions))
        self.logger.info("預測結果樣本:")
        for id_, pred in results[:5]:
            self.logger.info(f"ID: {id_}, 預測: {pred}")
        
        return predictions.tolist()

def main():
    # 設定檔案路徑
    train_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\train.json"
    test_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\test.json"
    articles_dir = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\articles"
    valid_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\valid.json"
    # 初始化分類器
    classifier = NewsClassifier(train_file_path, test_file_path, articles_dir,valid_file_path, use_gpu=True)
    
    # 訓練模型
    classifier.train_model()
    
    # 進行預測並獲取結果
    predictions = classifier.predict()
    
    # 輸出預測結果
    print("\n最終預測結果陣列:")
    print(predictions)
    
    file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\sample_submission.csv"
    df = pd.read_csv(file_path)
    df['rating'] = predictions
    df.to_csv(file_path, index=False)
    print("Submission file created: submission.csv")

if __name__ == "__main__":
    main()
# %%  doc2vec

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import logging
import json
from typing import Dict, List, Any
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import torch
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class NewsClassifier:
    def __init__(self, train_file_path: str, test_file_path: str, articles_dir: str, valid_file_path: str, use_gpu: bool = True):
        self.train_file_path = Path(train_file_path)
        self.test_file_path = Path(test_file_path)
        self.valid_file_path = Path(valid_file_path)  # 新增的验证数据路径
        self.articles_dir = Path(articles_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.setup_logging()
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def setup_logging(self):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
            if self.use_gpu:
                self.logger.info("GPU 可用，將使用 GPU 加速")
                self.logger.info(f"使用的 GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("將使用 CPU 模式")

    def preprocess_text(self, text: str) -> str:
        """文本預處理：包括去除特殊字符、詞形還原和去除停用詞"""
        text = text.lower()
        text = re.sub(r'_{2,}', ' ', text)  # 去除連續兩個或以上的下劃線
        text = re.sub(r'-{2,}', ' ', text) # 去除連續兩個或以上的中劃線
        text = re.sub(r'\d+', '', text)  # 去除數字
        text = re.sub(r'[^\w\s]', '', text)  # 去除標點
        text = re.sub(r'\s+', ' ', text)  # 去除多餘空白
        text = re.sub(r"\*", "", text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)   
    
    def extract_text_from_json(self, content: List[Dict]) -> str:
        """從JSON內容中提取文本"""
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.extend(str(value) for value in item.values() if isinstance(value, (str, int, float)))
            elif isinstance(item, list):
                texts.append(self.extract_text_from_json(item))
            elif isinstance(item, (str, int, float)):
                texts.append(str(item))
        return ' '.join(texts)
    
    def load_json_content(self, json_file_name: str) -> str:
        """讀取並合併JSON文件中的文本內容"""
        try:
            full_path = self.articles_dir / json_file_name
            self.logger.info(f"正在讀取檔案: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                if isinstance(content, list):
                    text = self.extract_text_from_json(content)
                elif isinstance(content, dict):
                    text = ' '.join(str(value) for value in content.values() 
                                 if isinstance(value, (str, int, float)))
                else:
                    text = str(content)
                
                return self.preprocess_text(text)
        except Exception as e:
            self.logger.error(f"讀取文件出錯 {json_file_name}: {str(e)}")
            return ""
    
    def prepare_valid_data(self) -> pd.DataFrame:
        """準備驗證數據"""
        self.logger.info(f"開始讀取驗證資料: {self.valid_file_path}")
        
        with open(self.valid_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        for record in data:
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_data.append({
                        'id': record["metadata"]["id"],
                        'content': combined_content,
                        'rating': record["label"]["rating"]
                    })
                    self.logger.info(f"成功處理ID: {record['metadata']['id']}")
            except Exception as e:
                self.logger.error(f"處理驗證記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        df = pd.DataFrame(processed_data)
        self.logger.info(f"總共處理了 {len(df)} 筆驗證資料")
        return df

    def prepare_training_data(self) -> pd.DataFrame:
        """準備訓練數據，並合併訓練資料和驗證資料"""
        self.logger.info(f"開始讀取訓練資料: {self.train_file_path}")
        
        with open(self.train_file_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        processed_train_data = []
        for record in train_data:
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_train_data.append({
                        'id': record["metadata"]["id"],
                        'content': combined_content,
                        'rating': record["label"]["rating"]
                    })
                    self.logger.info(f"成功處理ID: {record['metadata']['id']}")
            except Exception as e:
                self.logger.error(f"處理訓練記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        df_train = pd.DataFrame(processed_train_data)
        self.logger.info(f"總共處理了 {len(df_train)} 筆訓練資料")
        
        # 合併訓練資料和驗證資料
        df_valid = self.prepare_valid_data()
        df_combined = pd.concat([df_train, df_valid], ignore_index=True)
        self.logger.info(f"總共處理了 {len(df_combined)} 筆訓練數據（包括驗證數據）")
        
        return df_combined

    def train_model(self):
        """訓練模型"""
        self.logger.info("開始訓練模型...")
        df = self.prepare_training_data()
        
        if len(df) == 0:
            self.logger.error("沒有可用的訓練數據！")
            return

        # 使用 Doc2Vec 替換 TF-IDF
        documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(df['content'])]
        self.doc2vec_model = Doc2Vec(documents, vector_size=150, window=5, min_count=1, workers=4, epochs=40)

        # 轉換特徵
        X = np.array([self.doc2vec_model.dv[str(i)] for i in range(len(df))])
        y = df['rating']
        
        # 將特徵矩陣轉換為DMatrix格式
        dtrain = xgb.DMatrix(X, label=y)
        
        # 設定XGBoost參數
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': 5,
            'eta': 0.05,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'eval_metric': 'auc',
            'nthread': 12
        }

        # 訓練模型
        self.logger.info("開始XGBoost訓練...")
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=20
        )
        
        self.logger.info("模型訓練完成")

    def prepare_test_data(self) -> tuple:
        """準備測試數據"""
        self.logger.info(f"開始讀取測試資料: {self.test_file_path}")
        
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        ids = []
        for record in data:
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_data.append(combined_content)
                    ids.append(record["metadata"]["id"])
                    self.logger.info(f"成功處理測試ID: {record['metadata']['id']}")
            except Exception as e:
                self.logger.error(f"處理測試記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        self.logger.info(f"總共處理了 {len(processed_data)} 筆測試資料")
        
        # 將測試數據轉換為 Doc2Vec 特徵
        X_test = np.array([self.doc2vec_model.infer_vector(text.split()) for text in processed_data])
        
        return X_test, ids

    def predict(self) -> List[int]:
        """預測測試數據"""
        self.logger.info("開始進行預測...")
        
        # 準備測試數據
        X_test, test_ids = self.prepare_test_data()
        
        # 轉換特徵
        dtest = xgb.DMatrix(X_test)
        
        # 預測
        predictions = self.model.predict(dtest)
        predictions = predictions.astype(int)
        
        # 記錄預測結果
        self.logger.info("預測完成")
        self.logger.info(f"預測結果分布:\n{pd.Series(predictions).value_counts()}")
        
        # 將預測結果與ID配對
        results = list(zip(test_ids, predictions))
        self.logger.info("預測結果樣本:")
        for id_, pred in results[:5]:
            self.logger.info(f"ID: {id_}, 預測: {pred}")
        
        return predictions.tolist()

def main():
    # 設定檔案路徑
    train_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\train.json"
    test_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\test.json"
    articles_dir = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\articles"
    valid_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\valid.json"
    # 初始化分類器
    classifier = NewsClassifier(train_file_path, test_file_path, articles_dir,valid_file_path, use_gpu=True)
    
    # 訓練模型
    classifier.train_model()
    
    # 進行預測並獲取結果
    predictions = classifier.predict()
    
    # 輸出預測結果
    print("\n最終預測結果陣列:")
    print(predictions)
    
    file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\sample_submission.csv"
    df = pd.read_csv(file_path)
    df['rating'] = predictions
    df.to_csv(file_path, index=False)
    print("Submission file created: submission.csv")

if __name__ == "__main__":
    main()

# %% lstm

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

class NewsClassifier:
    def __init__(self, train_file_path: str, test_file_path: str, articles_dir: str, valid_file_path: str, 
                 max_words: int = 50000, max_len: int = 500, embedding_dim: int = 128):
        self.train_file_path = Path(train_file_path)
        self.test_file_path = Path(test_file_path)
        self.valid_file_path = Path(valid_file_path)
        self.articles_dir = Path(articles_dir)
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.setup_logging()
        self.setup_nltk()
        self.setup_gpu()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU available: {gpus}")
            except RuntimeError as e:
                self.logger.error(f"GPU setup error: {e}")
        else:
            self.logger.info("No GPU available, using CPU")
            
    def setup_nltk(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """文本預處理：包括去除特殊字符、詞形還原和去除停用詞"""
        text = text.lower()
        text = re.sub(r'_{2,}', ' ', text)
        text = re.sub(r'-{2,}', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"\*", "", text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_text_from_json(self, content: List[Dict]) -> str:
        """從JSON內容中提取文本"""
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.extend(str(value) for value in item.values() if isinstance(value, (str, int, float)))
            elif isinstance(item, list):
                texts.append(self.extract_text_from_json(item))
            elif isinstance(item, (str, int, float)):
                texts.append(str(item))
        return ' '.join(texts)
    
    def load_json_content(self, json_file_name: str) -> str:
        """讀取並合併JSON文件中的文本內容"""
        try:
            full_path = self.articles_dir / json_file_name
            with open(full_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    text = self.extract_text_from_json(content)
                elif isinstance(content, dict):
                    text = ' '.join(str(value) for value in content.values() 
                                 if isinstance(value, (str, int, float)))
                else:
                    text = str(content)
                return self.preprocess_text(text)
        except Exception as e:
            self.logger.error(f"讀取文件出錯 {json_file_name}: {str(e)}")
            return ""

    def prepare_data(self, file_path: Path, is_training: bool = True) -> tuple:
        """準備數據"""
        self.logger.info(f"開始讀取資料: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        ids = []
        ratings = [] if is_training else None
        
        for record in tqdm(data, desc="處理資料"):
            try:
                article_contents = []
                for json_file in record["metadata"]["premise_articles"].values():
                    json_file_name = os.path.basename(json_file)
                    content = self.load_json_content(json_file_name)
                    if content:
                        article_contents.append(content)
                
                if article_contents:
                    combined_content = " ".join(article_contents)
                    processed_data.append(combined_content)
                    ids.append(record["metadata"]["id"])
                    if is_training:
                        ratings.append(record["label"]["rating"])
            except Exception as e:
                self.logger.error(f"處理記錄出錯 {record.get('metadata', {}).get('id', 'unknown')}: {str(e)}")
        
        return processed_data, ids, ratings


    def create_model(self):
        """使用Functional API創建模型"""
        # 定義輸入層
        inputs = Input(name='inputs', shape=[self.max_len])
        
        # 嵌入層
        x = Embedding(self.max_words + 1, self.embedding_dim, input_length=self.max_len)(inputs)
        
        # LSTM層
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)
        
        # 全連接層
        x = Dense(128, activation="relu", name="FC1")(x)
        x = Dropout(0.5)(x)
        
        # 輸出層 (3個分類)
        outputs = Dense(3, activation="softmax", name="FC2")(x)
        
        # 創建模型
        model = Model(inputs=inputs, outputs=outputs)

        
        model.compile(optimizer=Adam(learning_rate = 0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        model.summary()
        return model

    def train_model(self):
        """訓練模型"""
        self.logger.info("開始準備訓練數據...")
        with tf.device('/CPU:0'):
            # 準備訓練和驗證數據
            train_texts, train_ids, train_ratings = self.prepare_data(self.train_file_path)
            valid_texts, valid_ids, valid_ratings = self.prepare_data(self.valid_file_path)
            
            # 合併訓練和驗證數據
            all_texts = train_texts + valid_texts
            all_ratings = train_ratings + valid_ratings
            
            # 創建和訓練tokenizer
            self.logger.info("開始訓練tokenizer...")
            self.tokenizer = Tokenizer(num_words=self.max_words)
            self.tokenizer.fit_on_texts(all_texts)
            
            # 轉換文本為序列
            sequences = self.tokenizer.texts_to_sequences(all_texts)
            X = pad_sequences(sequences, maxlen=self.max_len)
            
            # 將標籤轉換為one-hot編碼
            y = tf.keras.utils.to_categorical(all_ratings, num_classes=3)
            
            # 分割訓練集和驗證集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
            )
        
        # 創建和編譯模型
        self.logger.info("創建模型...")
        self.model = self.create_model()
        
        # 設置回調函數
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.6, 
                patience=3, 
                verbose=1
            )
        ]
        
        # 訓練模型
        self.logger.info("開始訓練模型...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=20,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        self.logger.info("模型訓練完成")
        return history

    def predict(self) -> List[int]:
        """預測測試數據"""
        self.logger.info("開始進行預測...")
        with tf.device('/CPU:0'):
            # 準備測試數據
            test_texts, test_ids, _ = self.prepare_data(self.test_file_path, is_training=False)
            
            # 轉換文本為序列
            sequences = self.tokenizer.texts_to_sequences(test_texts)
            X_test = pad_sequences(sequences, maxlen=self.max_len)
            
        # 預測
        predictions = self.model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        
        self.logger.info("預測完成")
        self.logger.info(f"預測結果分布:\n{pd.Series(predictions).value_counts()}")
        
        return predictions.tolist()

def main():
    # 設定檔案路徑
    train_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\train.json"
    test_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\test.json"
    articles_dir = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\articles"
    valid_file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\valid.json"
    
    # 初始化分類器
    classifier = NewsClassifier(
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        articles_dir=articles_dir,
        valid_file_path=valid_file_path
    )
    
    # 訓練模型
    history = classifier.train_model()
    
    # 進行預測並獲取結果
    predictions = classifier.predict()
    
    # 輸出預測結果到CSV
    file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW2\2024-generative-information-retrieval-hw-2\sample_submission.csv"
    df = pd.read_csv(file_path)
    df['rating'] = predictions
    df.to_csv(file_path, index=False)
    print("Submission file created: submission.csv")

if __name__ == "__main__":
    main()
    
# %%  Groq


from groq import Groq
import os

import time
pred = []
client = Groq(
    api_key="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"
)


for i in rangelen(len(test)):
    article_id = test['id'].iloc[i]
    article = test['content'].iloc[i]
    

    
 
        # 發送請求，並要求只返回 0、1、2 的數字
    chat_completion = client.chat.completions.create(
    messages=[{
                "role": "user",
                "content": f"classify it into a numerical format: 0 (False), 1 (Partial True), 2 (True). Only return the number without any additional text.{article}"
            }],
            model="llama-3.1-8b-instant",  # 使用正確的模型
            stream=False,
        )

        # 輸出模型的回應（只包含數字）
    result = chat_completion.choices[0].message.content.strip()
    print(result)
    
    pred.append({"id": article_id, "result": result})
    time.sleep(1)

print(pred)