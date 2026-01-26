# -*- coding: utf-8 -*-
"""
Neo4j 直连 Search API 实现
绕过 OpenSPG Server，直接使用 Neo4j 向量索引
"""

import logging
from typing import List

from kag.common.tools.search_api.search_api_abc import SearchApiABC

logger = logging.getLogger(__name__)


@SearchApiABC.register("neo4j_search_api")
class Neo4jSearchAPI(SearchApiABC):
    """
    直接连接 Neo4j 的 Search API 实现
    解决 OpenSPG API 检索返回空结果的问题
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._driver = None
        self._database = "medicine"
        
        # Neo4j 连接配置
        import os
        self._uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._user = os.environ.get("NEO4J_USER", "neo4j")
        self._password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
    
    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri, 
                auth=(self._user, self._password)
            )
            logger.info(f"[Neo4jSearchAPI] 连接成功: {self._uri}")
        return self._driver
    
    def search_vector(
        self, label, property_key, query_vector, topk=10, ef_search=None, params=None
    ) -> List:
        """
        使用 Neo4j 向量索引进行搜索
        
        Args:
            label: 节点标签，如 "Medicine.Chunk" 或 "Pathology.Chunk"
            property_key: 属性名，如 "content" 或 "name"
            query_vector: 查询向量
            topk: 返回数量
        """
        driver = self._get_driver()
        if driver is None:
            return []
        
        # 构建索引名
        # 格式：_<namespace>_<type>_<property>_vector_index
        # 例如：_medicine_chunk_content_vector_index
        label_parts = label.lower().replace(".", "_")
        index_name = f"_{label_parts}_{property_key}_vector_index"
        
        logger.info(f"[Neo4jSearchAPI] 搜索: label={label}, property={property_key}, index={index_name}, topk={topk}")
        
        try:
            with driver.session(database=self._database) as session:
                result = session.run(f"""
                    CALL db.index.vector.queryNodes($index_name, $topk, $query_vector)
                    YIELD node, score
                    RETURN node.id AS id, node.name AS name, node.content AS content, 
                           labels(node) AS labels, score
                """, index_name=index_name, topk=topk, query_vector=query_vector)
                
                nodes = []
                for record in result:
                    nodes.append({
                        "score": record["score"],
                        "node": {
                            "id": record["id"],
                            "name": record["name"] or "",
                            "content": record["content"] or "",
                            "__labels__": record["labels"]
                        }
                    })
                
                logger.info(f"[Neo4jSearchAPI] 找到 {len(nodes)} 个节点 (最高分: {nodes[0]['score']:.3f})" if nodes else "[Neo4jSearchAPI] 未找到节点")
                return nodes
                
        except Exception as e:
            logger.warning(f"[Neo4jSearchAPI] 索引 {index_name} 搜索失败: {e}")
            # 尝试使用通用 Medicine 索引
            if "medicine" not in index_name:
                fallback_index = f"_medicine_{label.split('.')[-1].lower()}_{property_key}_vector_index"
                logger.info(f"[Neo4jSearchAPI] 尝试回退索引: {fallback_index}")
                try:
                    with driver.session(database=self._database) as session:
                        result = session.run(f"""
                            CALL db.index.vector.queryNodes($index_name, $topk, $query_vector)
                            YIELD node, score
                            RETURN node.id AS id, node.name AS name, node.content AS content, 
                                   labels(node) AS labels, score
                        """, index_name=fallback_index, topk=topk, query_vector=query_vector)
                        
                        nodes = []
                        for record in result:
                            nodes.append({
                                "score": record["score"],
                                "node": {
                                    "id": record["id"],
                                    "name": record["name"] or "",
                                    "content": record["content"] or "",
                                    "__labels__": record["labels"]
                                }
                            })
                        
                        if nodes:
                            logger.info(f"[Neo4jSearchAPI] 回退索引找到 {len(nodes)} 个节点")
                        return nodes
                except Exception as e2:
                    logger.warning(f"[Neo4jSearchAPI] 回退索引也失败: {e2}")
            return []
    
    def search_text(
        self, query_string, label_constraints=None, topk=10, params=None
    ) -> List:
        """文本搜索 - 使用 CONTAINS 匹配"""
        driver = self._get_driver()
        if driver is None:
            return []
        
        try:
            with driver.session(database=self._database) as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.name CONTAINS $query OR n.content CONTAINS $query
                    RETURN n.id AS id, n.name AS name, n.content AS content, 
                           labels(n) AS labels
                    LIMIT $topk
                """, query=query_string, topk=topk)
                
                nodes = []
                for record in result:
                    nodes.append({
                        "score": 0.8,  # 文本匹配给固定分数
                        "node": {
                            "id": record["id"],
                            "name": record["name"] or "",
                            "content": record["content"] or "",
                            "__labels__": record["labels"]
                        }
                    })
                return nodes
        except Exception as e:
            logger.warning(f"[Neo4jSearchAPI] 文本搜索失败: {e}")
            return []
    
    def search_custom(self, custom_query, params=None) -> List:
        """自定义 Cypher 查询"""
        driver = self._get_driver()
        if driver is None:
            return []
        
        try:
            with driver.session(database=self._database) as session:
                result = session.run(custom_query, **(params or {}))
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning(f"[Neo4jSearchAPI] 自定义查询失败: {e}")
            return []
