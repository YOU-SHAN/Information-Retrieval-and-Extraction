
import os
from PIL import Image
import json
image_t = r"C:\\Users\\yoush\\Desktop\\資訊檢索與擷取\\HW3\\2024-information-retrieval-extraction-homework-3\\train_images\\train_images\\"

file_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW3\2024-information-retrieval-extraction-homework-3\test.jsonl"
i=0
new = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
  
        data = json.loads(line.strip())

        i += 1     
        
        print(i)
        a = []
        for item in data["dialogue"]:
          a.append(item["message"])
          print(item["message"])
        new.append(' '.join(a))
        print("*"*100)

ans_id = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
 
        data = json.loads(line.strip())     
        
        ans_id.append(data["dialogue_id"])
 
# %%1 第一次繳交分數低

import numpy as np
import os
import torch
from torchvision import models, transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
ans = []
# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load image feature extraction model (ResNet50)
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove last fully connected layer
resnet_model = resnet_model.to(device)  # Move model to GPU if available
resnet_model.eval()  # Set to evaluation mode

# Load sentence embedding model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Image transformation preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize a list to hold the top 30 image indices for each text
top_30_indices_all_texts = []

# Process each text in 'new' (list of text inputs)
for i in range(len(new)):
    print("*" * i)
    text = new[i]
    
    # 1. Get text embedding
    text_embedding = sentence_model.encode([text])
    text_embedding = torch.tensor(text_embedding).to(device)  # Move to GPU if available
    
    # 2. Load images and extract features
    image_folder = "C:/Users/yoush/Desktop/資訊檢索與擷取/HW3/2024-information-retrieval-extraction-homework-3/test_images/test_images"
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
    
    # List to store image features
    image_embeddings = []
    
    # Extract features for all images
    for image_path in image_paths:
        # Read and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        
        # Get image features
        with torch.no_grad():
            image_features = resnet_model(image_tensor).flatten().cpu().numpy()  # Move back to CPU for further processing
        
        # Add image features to list
        image_embeddings.append(image_features)
    
    # Convert image features to NumPy array
    image_embeddings = np.array(image_embeddings)
    
    # 3. Dimensionality reduction to match text embedding dimension
    # We'll use PCA to reduce image features to 384 dimensions
    pca = PCA(n_components=384)
    image_embeddings_reduced = pca.fit_transform(image_embeddings)
    
    # 4. Calculate similarity between text and reduced image embeddings
    similarities = cosine_similarity(text_embedding.cpu().numpy(), image_embeddings_reduced)
    
    # 5. Get top 30 most similar images
    top_30_indices = similarities[0].argsort()[-30:][::-1]
    
    # Store the top 30 indices for this text in the list
    top_30_indices_all_texts.append(top_30_indices)
    
    # Output top 30 most similar image paths for this text
    print(f"Top 30 most similar images for text {i + 1}:")
    a = []
    for idx in top_30_indices:
        file_name_without_extension = os.path.splitext(os.path.basename(image_paths[idx]))[0]  # 去掉副檔名
        print(file_name_without_extension)
        a.append(file_name_without_extension)
    
    result_str = ' '.join(a)
    print(result_str)
    ans.append(result_str)
    

# %%2 過baseline 0.70333

import os
import time
import torch
from PIL import Image
import open_clip


def log_info(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

# 設備與環境檢查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Using device: {device}")
log_info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    log_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


image_folder = "C:/Users/yoush/Desktop/資訊檢索與擷取/HW3/2024-information-retrieval-extraction-homework-3/test_images/test_images"
log_info(f"Image Folder: {image_folder}")


image_paths = [
    os.path.join(image_folder, f) 
    for f in os.listdir(image_folder) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
]
log_info(f"Total Images Found: {len(image_paths)}")


start_time = time.time()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
log_info(f"Model Loaded (Time: {time.time() - start_time:.2f} seconds)")

def get_image_features_gpu(image_paths):
    log_info(f"Extracting features for {len(image_paths)} images")
    start_time = time.time()
    
    images = []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path)).unsqueeze(0)
            images.append(image)
        except Exception as e:
            log_info(f"Error processing image {path}: {e}")
    
    images = torch.cat(images).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
    
    log_info(f"Image feature extraction complete (Time: {time.time() - start_time:.2f} seconds)")
    return image_features

def get_text_features_gpu(texts):
    log_info(f"Extracting features for {len(texts)} texts")
    start_time = time.time()
    
    text_tokens = tokenizer(texts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    
    log_info(f"Text feature extraction complete (Time: {time.time() - start_time:.2f} seconds)")
    return text_features


def calculate_similarity_gpu(text_features, image_features):
    # 在 GPU 上計算相似度
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 計算所有文字-圖片對的相似度
    similarities = text_features @ image_features.T
    
    return similarities

def find_top_30_images_gpu(texts, image_folder):
    # 獲取所有圖片路徑
    image_paths = [
        os.path.join(image_folder, f) 
        for f in os.listdir(image_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    
    # 一次性提取所有圖片特徵
    image_features = get_image_features_gpu(image_paths)
    
    # 一次性提取所有文字特徵
    text_features = get_text_features_gpu(texts)
    
    # 計算相似度
    similarities = calculate_similarity_gpu(text_features, image_features)
    
    # 對每個文字找出前 30 張相似圖片
    top_30_results = []
    for text_similarity in similarities:
        # 找出相似度最高的 30 張圖片
        top_30_indices = text_similarity.topk(30).indices
        top_30_images = [
            os.path.splitext(os.path.basename(image_paths[idx]))[0] 
            for idx in top_30_indices
        ]
        result = " ".join(top_30_images) 
        top_30_results.append(result)
    
    return top_30_results

log_info("Starting image retrieval process")
start_total_time = time.time()
    
    # 找出前 30 張最相似的圖片
top_30_results = find_top_30_images_gpu(new, image_folder)
    
log_info(f"Total Process Time: {time.time() - start_total_time:.2f} seconds")
    
    # 寫入結果
for i, result in enumerate(top_30_results):
        log_info(f"Text {i+1} Top 30 Images: {' '.join(result)}")


  
# %%
import os
import time
import torch
import numpy as np
from PIL import Image
import open_clip
from torchvision import transforms
from typing import List, Tuple
import logging
import torch.nn.functional as F
from tqdm import tqdm

class ImageRetrievalSystem:
    def __init__(self, image_folder: str, model_name: str = 'ViT-L-14', pretrained: str = 'laion2b_s32b_b82k'):
        self.setup_logging()
        self.image_folder = image_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # 初始化模型
        self.model, self.preprocess = self.initialize_model(model_name, pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # 快取設置
        self.image_features_cache = {}
        self.text_features_cache = {}
        
        # 圖片路徑初始化
        self.image_paths = self.get_image_paths()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_retrieval.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_model(self, model_name: str, pretrained: str) -> Tuple:
        try:
            start_time = time.time()
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            model = model.to(self.device)
            model.eval()
            self.logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            return model, preprocess
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_image_paths(self) -> List[str]:
        image_paths = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        self.logger.info(f"Found {len(image_paths)} images")
        return image_paths

    def enhanced_preprocess(self, image: Image) -> torch.Tensor:
        """增強的圖像預處理"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def expand_query(self, text: str) -> List[str]:
        """查詢擴充"""
        expanded = [
            text,
            f"a photo of {text}",
            f"an image showing {text}",
            f"a picture of {text}",
            f"a clear image of {text}"
        ]
        return expanded

    @torch.no_grad()
    def get_image_features(self, batch_size: int = 32) -> torch.Tensor:
        """批次處理圖像特徵提取"""
        if not self.image_features_cache:
            all_features = []
            for i in tqdm(range(0, len(self.image_paths), batch_size), desc="Processing images"):
                batch_paths = self.image_paths[i:i + batch_size]
                batch_images = []
                
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert('RGB')
                        image_tensor = self.preprocess(image).unsqueeze(0)
                        batch_images.append(image_tensor)
                    except Exception as e:
                        self.logger.error(f"Error processing image {path}: {e}")
                        continue
                
                if batch_images:
                    batch_tensor = torch.cat(batch_images).to(self.device)
                    features = self.model.encode_image(batch_tensor)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features.cpu())
                    
            self.image_features_cache = torch.cat(all_features)
            
        return self.image_features_cache.to(self.device)

    @torch.no_grad()
    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        """文字特徵提取"""
        cache_key = tuple(texts)
        if cache_key not in self.text_features_cache:
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            self.text_features_cache[cache_key] = text_features.cpu()
            
        return self.text_features_cache[cache_key].to(self.device)

    def calculate_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor, 
                           temperature: float = 0.07) -> torch.Tensor:
        """計算相似度"""
        return (text_features @ image_features.T) / temperature

    def post_process_results(self, similarities: torch.Tensor, top_k: int = 30) -> List[str]:
        """後處理和結果重排序"""
        scores = similarities.cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            image_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
            results.append(image_name)
            
        return results

    def retrieve_images(self, query_texts: List[str], top_k: int = 30) -> List[str]:
        """主要檢索函數"""
        try:
            start_time = time.time()
            self.logger.info(f"Starting retrieval for {len(query_texts)} queries")

 
            expanded_queries = []
            for text in query_texts:
                expanded_queries.extend(self.expand_query(text))

        
            image_features = self.get_image_features()
            text_features = self.get_text_features(expanded_queries)

    
            similarities = self.calculate_similarity(text_features, image_features)

      
            n_expansions = len(self.expand_query(""))
            aggregated_similarities = similarities.view(-1, n_expansions, image_features.shape[0]).mean(dim=1)

     
            results = self.post_process_results(aggregated_similarities[0], top_k)
            
            self.logger.info(f"Retrieval completed in {time.time() - start_time:.2f} seconds")
            return results

        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            raise


image_folder = "C:/Users/yoush/Desktop/資訊檢索與擷取/HW3/2024-information-retrieval-extraction-homework-3/test_images/test_images/"
    
 
retrieval_system = ImageRetrievalSystem(image_folder)
    
query_texts = new
    

results = retrieval_system.retrieve_images(query_texts)
print("Top 30 retrieved images:", " ".join(results))

# %%結果寫入
import pandas as pd

data = {
    "dialogue_id": ans_id,
    "photo_id": top_30_results
}
df = pd.DataFrame(data)
output_path = r"C:\Users\yoush\Desktop\資訊檢索與擷取\HW3\2.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"資料已成功寫入 {output_path}")