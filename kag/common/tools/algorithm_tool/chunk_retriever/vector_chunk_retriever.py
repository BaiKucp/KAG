# -*- coding: utf-8 -*-
# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

import knext.common.cache
import logging

from kag.interface import RetrieverABC, VectorizeModelABC, ChunkData, RetrieverOutput
from kag.interface.solver.model.schema_utils import SchemaUtils
from kag.common.config import LogicFormConfiguration
from kag.common.tools.search_api.search_api_abc import SearchApiABC

from knext.schema.client import CHUNK_TYPE

logger = logging.getLogger()
chunk_cached_by_query_map = knext.common.cache.LinkCache(maxsize=100, ttl=300)


@RetrieverABC.register("vector_chunk_retriever")
class VectorChunkRetriever(RetrieverABC):
    def __init__(
        self,
        vectorize_model: VectorizeModelABC = None,
        search_api: SearchApiABC = None,
        top_k: int = 10,
        score_threshold=0.85,
        **kwargs,
    ):
        super().__init__(top_k, **kwargs)
        self.vectorize_model = vectorize_model or VectorizeModelABC.from_config(
            self.kag_config.all_config["vectorize_model"]
        )
        self.search_api = search_api or SearchApiABC.from_config(
            {"type": "openspg_search_api"}
        )
        self.schema_helper: SchemaUtils = SchemaUtils(
            LogicFormConfiguration(
                {
                    "KAG_PROJECT_ID": self.kag_project_config.project_id,
                    "KAG_PROJECT_HOST_ADDR": self.kag_project_config.host_addr,
                }
            )
        )
        self.score_threshold = score_threshold

    def invoke(self, task, **kwargs) -> RetrieverOutput:
        query = task.arguments["query"]
        top_k = kwargs.get("top_k", self.top_k)
        try:
            cached = chunk_cached_by_query_map.get(query)
            if cached and len(cached.chunks) > top_k:
                return cached
            if not query:
                logger.error("chunk query is emtpy", exc_info=True)
                return RetrieverOutput(
                    retriever_method=self.name,
                    err_msg="query is empty",
                )
            query_vector = self.vectorize_model.vectorize(query)
            chunk_label = self.schema_helper.get_label_within_prefix(CHUNK_TYPE)
            logger.info(f"[VectorChunkRetriever] 使用标签: {chunk_label}, score_threshold: {self.score_threshold}")
            
            # ============================================================
            # 智能教材搜索：优先使用已分类的目标教材（支持多教材）
            # ============================================================
            chunk_labels = []
            
            # 教材名到 Chunk 标签的映射
            TEXTBOOK_TO_CHUNK = {
                "pharmacology": "Pharmacology.Chunk",
                "pathology": "Pathology.Chunk",
                "physiology": "physiology.Chunk",
            }
            
            # 尝试获取当前问题的教材分类结果
            try:
                # 直接导入，不检查 sys.modules（因为模块名可能带路径前缀）
                from kag_compat_patch import get_current_textbooks
                target_textbooks = get_current_textbooks()
                logger.info(f"[VectorChunkRetriever] 获取到目标教材: {target_textbooks}")
                if target_textbooks:
                    # 将每个教材名转为 Chunk 标签
                    for tb in target_textbooks:
                        tb_lower = tb.lower()
                        if tb_lower in TEXTBOOK_TO_CHUNK:
                            chunk_labels.append(TEXTBOOK_TO_CHUNK[tb_lower])
                    
                    if chunk_labels:
                        logger.info(f"[VectorChunkRetriever] 使用目标教材 Chunk 标签: {chunk_labels}")
            except ImportError:
                logger.debug("[VectorChunkRetriever] kag_compat_patch 未加载")
            except Exception as e:
                logger.debug(f"[VectorChunkRetriever] 获取目标教材失败: {e}")
            
            # 如果没有目标教材，从 Neo4j 动态获取所有 Chunk 标签
            if not chunk_labels:
                try:
                    import os
                    from neo4j import GraphDatabase
                    
                    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
                    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
                    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
                    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
                    
                    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                    with driver.session(database=neo4j_database) as session:
                        result = session.run("""
                            CALL db.labels() YIELD label
                            WHERE label ENDS WITH '.Chunk'
                            RETURN label
                        """)
                        chunk_labels = [record["label"] for record in result]
                    driver.close()
                    
                    logger.info(f"[VectorChunkRetriever] 从 Neo4j 动态获取到 {len(chunk_labels)} 个 Chunk 标签: {chunk_labels}")
                except Exception as e:
                    logger.warning(f"[VectorChunkRetriever] 动态获取 Chunk 标签失败: {e}")
                    # 回退到硬编码的默认值
                    chunk_labels = ["Pathology.Chunk", "Pharmacology.Chunk", "physiology.Chunk"]
                    logger.info(f"[VectorChunkRetriever] 使用默认 Chunk 标签: {chunk_labels}")
            
            top_k_docs = []
            for ns_label in chunk_labels:
                try:
                    docs = self.search_api.search_vector(
                        label=ns_label,
                        property_key="content",
                        query_vector=query_vector,
                        topk=top_k,
                        ef_search=top_k * 7,
                    )
                    if docs:
                        logger.info(f"[VectorChunkRetriever] {ns_label} content 搜索返回 {len(docs)} 个结果")
                        top_k_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"[VectorChunkRetriever] {ns_label} 搜索失败: {e}")
            
            logger.info(f"[VectorChunkRetriever] 合并后共 {len(top_k_docs)} 个结果")
            if top_k_docs:
                for i, doc in enumerate(top_k_docs[:3]):
                    logger.info(f"[VectorChunkRetriever]   [{i}] score={doc.get('score', 'N/A')}, id={doc.get('node', {}).get('id', 'N/A')[:30]}")
            
            # name 搜索也使用多 namespace
            top_k_docs_name = []
            for ns_label in chunk_labels:
                try:
                    docs = self.search_api.search_vector(
                        label=ns_label,
                        property_key="name",
                        query_vector=query_vector,
                        topk=top_k / 2,
                        ef_search=top_k / 2 * 3,
                    )
                    if docs:
                        top_k_docs_name.extend(docs)
                except Exception as e:
                    pass
            
            logger.info(f"[VectorChunkRetriever] name 搜索返回 {len(top_k_docs_name)} 个结果")
            top_k_docs = top_k_docs_name + top_k_docs

            merged = {}
            chunk_map = {}
            chunks = []
            for item in top_k_docs:
                score = item.get("score", 0.0)
                if score >= self.score_threshold:
                    chunk = ChunkData(
                        content=item["node"].get("content", ""),
                        title=item["node"]["name"],
                        chunk_id=item["node"]["id"],
                        score=score,
                    )
                    if chunk.chunk_id not in merged:
                        merged[chunk.chunk_id] = score
                    if merged[chunk.chunk_id] < score:
                        merged[chunk.chunk_id] = score
                    chunk_map[chunk.chunk_id] = chunk

            sorted_chunk_ids = sorted(merged.items(), key=lambda x: -x[1])
            for item in sorted_chunk_ids:
                chunk_id, score = item
                chunk = chunk_map[chunk_id]
                chunks.append(
                    ChunkData(
                        content=chunk.content,
                        title=chunk.title,
                        chunk_id=chunk.chunk_id,
                        score=score,
                    )
                )
            out = RetrieverOutput(chunks=chunks, retriever_method=self.name)
            chunk_cached_by_query_map.put(query, out)
            return out

        except Exception as e:
            logger.error(f"run calculate_sim_scores failed, info: {e}", exc_info=True)
            return RetrieverOutput(retriever_method=self.name, err_msg=str(e))

    def schema(self):
        return {
            "name": "vector_chunk_retriever",
            "description": "Retrieve relevant text chunks from document store using vector similarity search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for retrieving relevant text chunks",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }

    @property
    def input_indices(self):
        return ["chunk"]
