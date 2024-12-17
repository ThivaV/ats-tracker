import pandas as pd
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class HandleATS:
    def __init__(
        self,
        resume_catalog_uri: str = None,
        milvusdb_uri: str = None,
        collection_name: str = None,
    ):
        # milvus database connection string
        self.milvusdb_uri = milvusdb_uri
        self.collection_name = collection_name
        self.resume_catalog_uri = resume_catalog_uri

        self.collection = None
        self.bge_m3 = None
        self.catalog_df = None

    def initialize_database(self):
        # connect to milvus db
        connections.connect(uri=self.milvusdb_uri)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def initialize_embedding(self):
        # please set the use_fp16 to False when you are using cpu.
        # by default the return options is:
        #  return_dense True
        #  return_sparse True
        #  return_colbert_vecs False
        self.bge_m3 = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",  # specify the model name
            device="cpu",  # specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=False,  # specify whether to use fp16. Set to `False` if `device` is `cpu`.
        )

    def initialize_catalog(self):
        self.catalog_df = pd.read_csv(self.resume_catalog_uri)

    def encode_documents(self, doc: str = None):
        """BGE-M3 returns both dense and sparse encodings"""
        embeddings = self.bge_m3.encode_documents([doc])

        dense = embeddings["dense"][0]
        sparse_list = list(embeddings["sparse"])

        # convert csr matrix to dictionary
        row_index = 0
        sparse = {
            idx: val
            for idx, val in zip(
                sparse_list[row_index].indices, sparse_list[row_index].data
            )
        }

        return dense, sparse

    def search(
        self,
        query: str = None,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        limit: int = 10,
    ) -> list[any]:
        resumes = []
        if query:
            # fetch the dense & sprse encoding of query
            dense_query_embeds, sparse_query_embeds = self.encode_documents(query)

            # dense
            dense_search_params = {"metric_type": "IP", "params": {}}
            dense_req = AnnSearchRequest(
                [dense_query_embeds], "dense_vector", dense_search_params, limit=limit
            )

            # sparse
            sparse_search_params = {"metric_type": "IP", "params": {}}
            sparse_req = AnnSearchRequest(
                [sparse_query_embeds],
                "sparse_vector",
                sparse_search_params,
                limit=limit,
            )

            # rerank
            rerank = WeightedRanker(sparse_weight, dense_weight)
            resp = self.collection.hybrid_search(
                [sparse_req, dense_req],
                rerank=rerank,
                limit=limit,
                output_fields=["id"],
            )[0]

            resumes = [hit.get("id") for hit in resp]
        return resumes

    def retrieve_resume_info(self, resumes: list) -> pd.DataFrame:
        resumes = [int(resume) for resume in resumes]
        filtered_resumes = self.catalog_df[self.catalog_df["resume_id"].isin(resumes)]
        return filtered_resumes
