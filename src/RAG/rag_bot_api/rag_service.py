# rag_service.py

from faiss_search.chunk_searcher import ChunkSearcher
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGService:
    def __init__(self):
        self.searcher = ChunkSearcher()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def answer(self, question: str, k: int = 3):
        query_embedding = self.model.encode([question], convert_to_tensor=False)
        results = self.searcher.search(np.array(query_embedding), k=k)

        context = "\n\n".join([result["text"] for result in results])
        return {
            "question": question,
            "context_used": [result["text"][:200] + "..." for result in results],
            "context": context
        }