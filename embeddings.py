import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class AdvancedEmbeddings:
    """Advanced embedding techniques for recommendation system"""
    
    def __init__(self):
        """Initialize Advanced Embeddings"""
        self.embeddings = {}
        self.embedding_models = {}
        
    def create_word2vec_embeddings(
        self,
        games_df: pd.DataFrame,
        embedding_dim: int = 100
    ) -> np.ndarray:
        """
        Create Word2Vec embeddings from game descriptions
        
        Args:
            games_df: DataFrame containing game information
            embedding_dim: Embedding dimension (default 100)
        
        Returns:
            Embedding matrix (n_games x embedding_dim)
        """
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("[WARNING] gensim not installed. Install with: pip install gensim")
            return self._fallback_embeddings(len(games_df), embedding_dim)
        
        # Tokenize descriptions
        sentences = []
        for desc in games_df['short_description'].fillna(''):
            words = str(desc).lower().split()
            sentences.append([w for w in words if len(w) > 2])
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=sentences,
            vector_size=embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            sg=1  # Skip-gram model
        )
        
        # Create embeddings for each game
        embeddings = []
        for desc in games_df['short_description'].fillna(''):
            words = str(desc).lower().split()
            word_vectors = [model.wv[w] for w in words if w in model.wv]
            
            if word_vectors:
                embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(embedding_dim)
            
            embeddings.append(embedding)
        
        self.embedding_models['word2vec'] = model
        print(f"[OK] Created Word2Vec embeddings ({embedding_dim}D)")
        
        return np.array(embeddings)
    
    def create_tfidf_embeddings(
        self,
        games_df: pd.DataFrame,
        max_features: int = 100
    ) -> Tuple[np.ndarray, object]:
        """
        Create TF-IDF embeddings from genres and descriptions
        
        Args:
            games_df: DataFrame containing game information
            max_features: Maximum number of features
        
        Returns:
            Tuple of (embedding matrix, vectorizer object)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Combine text features
        games_df_copy = games_df.copy()
        games_df_copy['combined_text'] = (
            games_df_copy['genres'].fillna('') + ' ' +
            games_df_copy['categories'].fillna('') + ' ' +
            games_df_copy['short_description'].fillna('')
        )
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        embeddings = vectorizer.fit_transform(games_df_copy['combined_text']).toarray()
        
        self.embedding_models['tfidf'] = vectorizer
        print(f"[OK] Created TF-IDF embeddings ({max_features}D)")
        
        return embeddings, vectorizer
    
    def create_bert_embeddings(
        self,
        games_df: pd.DataFrame,
        model_name: str = 'distilbert-base-uncased'
    ) -> np.ndarray:
        """
        Create BERT embeddings from game descriptions
        
        Args:
            games_df: DataFrame containing game information
            model_name: Pre-trained BERT model name
        
        Returns:
            Embedding matrix
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            print("[WARNING] transformers not installed. Install with: pip install transformers torch")
            return self._fallback_embeddings(len(games_df), 384)
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        embeddings = []
        
        for idx, desc in enumerate(games_df['short_description'].fillna('')):
            # Truncate to 512 tokens (BERT max)
            desc = str(desc)[:512]
            
            # Tokenize
            inputs = tokenizer(
                desc,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            embeddings.append(cls_embedding[0])
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} games...")
        
        self.embedding_models['bert'] = model
        print(f"[OK] Created BERT embeddings (768D)")
        
        return np.array(embeddings)
    
    def create_sentence_transformer_embeddings(
        self,
        games_df: pd.DataFrame,
        model_name: str = 'all-MiniLM-L6-v2'
    ) -> np.ndarray:
        """
        Create Sentence Transformer embeddings
        
        Args:
            games_df: DataFrame containing game information
            model_name: SentenceTransformer model name
        
        Returns:
            Embedding matrix
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[WARNING] sentence-transformers not installed. Install with: pip install sentence-transformers")
            return self._fallback_embeddings(len(games_df), 384)
        
        # Load model
        model = SentenceTransformer(model_name)
        
        # Create sentences combining game info
        sentences = []
        for _, row in games_df.iterrows():
            sentence = f"{row['name']} {row['genres']} {row['short_description']}"
            sentences.append(sentence)
        
        # Encode sentences
        embeddings = model.encode(
            sentences,
            show_progress_bar=True,
            batch_size=32
        )
        
        self.embedding_models['sentence_transformer'] = model
        print(f"[OK] Created Sentence Transformer embeddings ({embeddings.shape[1]}D)")
        
        return embeddings
    
    def _fallback_embeddings(self, n_games: int, embedding_dim: int) -> np.ndarray:
        """
        Fallback to random embeddings if library not available
        """
        print(f"[WARNING] Using fallback random embeddings ({embedding_dim}D)")
        return np.random.randn(n_games, embedding_dim)
    
    def combine_embeddings(
        self,
        embeddings_list: List[np.ndarray],
        weights: List[float] = None
    ) -> np.ndarray:
        """
        Combine multiple embeddings with weights
        
        Args:
            embeddings_list: List of embedding matrices
            weights: Weights for each embedding (default: equal)
        
        Returns:
            Combined embedding matrix
        """
        if weights is None:
            weights = [1.0 / len(embeddings_list)] * len(embeddings_list)
        
        # Normalize embeddings to same dimension
        min_dim = min(e.shape[1] for e in embeddings_list)
        
        combined = np.zeros((embeddings_list[0].shape[0], min_dim))
        
        for emb, weight in zip(embeddings_list, weights):
            combined += weight * emb[:, :min_dim]
        
        # Normalize
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        
        return combined
    
    def dimensionality_reduction(
        self,
        embeddings: np.ndarray,
        target_dim: int = 50,
        method: str = 'pca'
    ) -> np.ndarray:
        """
        Reduce embedding dimensionality
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension
            method: 'pca' or 'tsne'
        
        Returns:
            Reduced embeddings
        """
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=target_dim)
            return reducer.fit_transform(embeddings)
        
        elif method == 'tsne':
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=target_dim, random_state=42)
                return reducer.fit_transform(embeddings)
            except ImportError:
                print("[WARNING] Using PCA instead (TSNE requires sklearn 1.2+)")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=target_dim)
                return reducer.fit_transform(embeddings)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding
            embeddings: All embeddings
            top_k: Number of results
        
        Returns:
            Tuple of (indices, similarities)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = cosine_similarity([query_norm], embeddings_norm)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices.tolist(), top_similarities.tolist()
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """
        Get embedding statistics
        
        Args:
            embeddings: Embedding matrix
        
        Returns:
            Dictionary of statistics
        """
        return {
            'shape': embeddings.shape,
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'sparsity': float(np.sum(embeddings == 0) / embeddings.size)
        }

print("[OK] AdvancedEmbeddings class ready!")
