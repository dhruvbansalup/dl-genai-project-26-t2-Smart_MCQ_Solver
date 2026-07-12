# RAG
# kb->Embedder & Indexing->Reteriver->Reranker->Generator
import faiss
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer


class RAGPipeline:
    def __init__(
        self,
        knowledge_base,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=20,# top-k docs to retrieve
        rerank_k=5 # top-k docs to rerank
    ):

        self.knowledge_base = knowledge_base
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.top_k = top_k
        self.rerank_k = rerank_k

        print(f"Loading Embedding Model: {self.embedding_model}")
        self.embedder=SentenceTransformer(self.embedding_model)

        print(f"Loading Reranker Model: {self.reranker_model}")
        self.reranker = CrossEncoder(self.reranker_model)

        self.documents = [] # cached kb docs
        self.index=None # cached vector_store index
        self.embedding_dim=None

    def fit(self, train_df):
        '''
        build the kb, embeddings & index the embeddings
        '''

        # Building KB
        self.knowledge_base.build(train_df)
        print(f"Knowledge Base built with {self.knowledge_base.__len__()} documents")
        self.documents = self.knowledge_base.get_documents()

        #Making Embeddings
        embeddings=self.embedder.encode(
            self.documents,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(embeddings) # Normalizing embeddings

        # Indexing Embeddings
        self.embedding_dim=embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        print(f"Embeddings indexed with {len(embeddings)} vectors")


    def _embed_query(self, query):
        """
        Helper to embed the query
        """
        query_embedding = self.embedder.encode(
            query,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype(np.float32) # (embedding_dim,)
        query_embedding=query_embedding.reshape(1, -1) # (1, embedding_dim)

        faiss.normalize_L2(query_embedding) # Normalizing embeddings
        return query_embedding


    def _rerank(self, query, retrieved_docs):
        """
        Helper to rerank the retrieved docs for given query
        """
        # Building pairs
        pairs = [[query, doc] for doc in retrieved_docs]

        # Getting scores
        rerank_scores = self.reranker.predict(pairs)

        # Reranking based on scores
        reranked_indices = np.argsort(rerank_scores)[::-1][:self.rerank_k]
        return [retrieved_docs[i] for i in reranked_indices]

    def retrieve(self, query):
        """
        Retrieve top docs from the knowledge base for given query
        """

        query_embedding = self._embed_query(query)

        # Search top-k docs from the index for given query embedding
        if self.index is None:
            raise RuntimeError("Knowledge base is not fitted. fit() before search")
        distances, indices = self.index.search(query_embedding, self.top_k)
        indices=indices[0]

        # Reterive top_k docs from the knowledge base
        retrieved_docs = [self.documents[i] for i in indices]

        # Rerank to get top rerank_k docs
        retrieved_docs=self._rerank(query, retrieved_docs)

        return retrieved_docs

    def _build_context(self, reterieved_docs):
        """
        Build context from the retrieved docs
        """

        if  len(reterieved_docs)==0:
            return ""

        context=[]

        for i, doc in enumerate(reterieved_docs):
            context.append(f"Document {i+1}: {doc}")

        return "\n[SEP]\n".join(context)

    def augment_dataframe(self, df):
        """
        Retrieve and augment the df with context for each prompt
        """

        df=df.copy()

        contexts=[]

        print("Retrieving context for each prompt...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving context"):
            retrieved_docs=self.retrieve(query=row['prompt'])
            context=self._build_context(retrieved_docs)
            contexts.append(context)

        df["context"]=contexts

        return df