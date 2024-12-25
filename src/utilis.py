import json
import pandas as pd

from pymilvus import connections, Collection, model, AnnSearchRequest, WeightedRanker
from pymilvus.model.sparse import BM25EmbeddingFunction  # type: ignore
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer  # type: ignore


class ATSTracker:
    def __init__(
        self,
        resume_catalog_uri: str = None,
        milvusdb_uri: str = None,
        collection_name: str = None,
    ):
        self.resume_catalog_uri = resume_catalog_uri
        self.milvusdb_uri = milvusdb_uri
        self.collection_name = collection_name

        self.collection = None
        self.resumes_df = None

        self.bm25_ef = None
        self.sentence_transformer_ef = None
        self.dense_dimension = 768

    def initialize_database(self):
        connections.connect(uri=self.milvusdb_uri)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def initialize_catalog(self):
        self.resumes_df = pd.read_csv(self.resume_catalog_uri)

    def initialize_sparse_embeddings(self):
        analyzer = build_default_analyzer(language="en")
        self.bm25_ef = BM25EmbeddingFunction(analyzer)

        # fit the model on the corpus to get the statstics of the corpus
        corpus = self.resumes_df["resume"].tolist()
        self.bm25_ef.fit(corpus)

    def generate_bm25_embedding(self, query):
        bm25_csr_matrix = self.bm25_ef.encode_documents([query])
        bm25_dict_embedding = {
            idx: val for idx, val in zip(bm25_csr_matrix.indices, bm25_csr_matrix.data)
        }
        return bm25_dict_embedding

    def initialize_dense_embeddings(self):
        # initialize the SentenceTransformerEmbeddingFunction
        self.sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
            model_name="bert-base-uncased",  # specify the model name
            device="cpu",  # specify the device to use, e.g., 'cpu' or 'cuda:0'
        )

    def generate_dense_embedding(self, query):
        return self.sentence_transformer_ef.encode_documents([query])[0]

    def search(self, query, sparse_weight=0.5, dense_weight=0.5, top_k=5):
        dense_embedding = self.generate_dense_embedding(query)
        request_1 = AnnSearchRequest(
            [dense_embedding], "dense", {"metric_type": "IP", "params": {}}, limit=top_k
        )

        sparse_embedding = self.generate_bm25_embedding(query)
        request_2 = AnnSearchRequest(
            [sparse_embedding],
            "sparse",
            {"metric_type": "IP", "params": {}},
            limit=top_k,
        )

        reqs = [request_1, request_2]
        ranker = WeightedRanker(sparse_weight, dense_weight)

        results = self.collection.hybrid_search(
            reqs=reqs,
            rerank=ranker,
            limit=top_k,
            output_fields=["resume_id", "domain", "uri"],
        )[0]

        if len(results) == 0:
            return []
        else:
            results = [
                {
                    "distance": round(item.distance * 100, 2),
                    "resume_id": item.get("resume_id"),
                    "domain": item.get("domain"),
                    "uri": item.get("uri"),
                }
                for item in results
            ]
            return json.dumps(results, indent=4)
