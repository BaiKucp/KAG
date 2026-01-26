"""
Neo4j 直连溯源模块
直接连接 Docker 中的 Neo4j，绕过 OpenSPG API
支持多教材动态溯源：根据 Chunk 标签自动检测命名空间 (如 Pharmacology.*, Pathology.*, physiology.*)
"""

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Neo4j 连接配置
# Docker 容器内使用: bolt://release-openspg-neo4j:7687
# 本地开发使用: bolt://localhost:7687
# 通过环境变量覆盖
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")

# 默认数据库（Medicine 项目统一使用 medicine 数据库）
NEO4J_DEFAULT_DATABASE = os.environ.get("NEO4J_DATABASE", "medicine")

# 缓存 driver（只需一个，可以连接不同数据库）
_neo4j_driver = None


def get_neo4j_driver():
    """获取 Neo4j driver（单例）"""
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # 测试连接
            _neo4j_driver.verify_connectivity()
            logger.info("Neo4j 直连初始化成功: %s, database=%s", NEO4J_URI, NEO4J_DEFAULT_DATABASE)
        except ImportError:
            logger.warning("neo4j 包未安装，请运行: pip install neo4j")
            return None
        except Exception as e:
            logger.warning("Neo4j 连接失败: %s", e)
            return None
    return _neo4j_driver


def get_database_for_namespace(namespace: str) -> str:
    """
    根据 namespace 返回对应的 Neo4j 数据库名。
    
    Medicine 项目统一使用 'medicine' 数据库，不同教材通过标签前缀区分。
    
    Args:
        namespace: 命名空间，如 "Pharmacology", "Pathology"
        
    Returns:
        数据库名，统一为 "medicine"
    """
    # Medicine 项目所有教材都存储在同一个数据库中
    return NEO4J_DEFAULT_DATABASE


# ============================================================================
# 智能检索功能：教材分类 + AtomicQuery 匹配 + Chunk 检索
# ============================================================================

# 从 medical_generator_prompt 导入教材分类功能
from medical_generator_prompt import TEXTBOOK_KEYWORDS, classify_textbook


def search_atomic_queries(
    query: str, 
    namespace: str = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    通过关键词匹配搜索 AtomicQuery
    
    Args:
        query: 用户问题
        namespace: 命名空间（如 "Pharmacology"），None 表示搜索所有
        top_k: 返回数量
        
    Returns:
        匹配的 AtomicQuery 列表，每个包含 id, title, namespace
    """
    driver = get_neo4j_driver()
    if driver is None:
        return []
    
    # 构建标签匹配
    if namespace:
        label = f"`{namespace}.AtomicQuery`"
    else:
        # 搜索所有命名空间的 AtomicQuery
        label = "AtomicQuery"
    
    # 提取查询中的关键词
    keywords = [kw for kw in query if len(kw) >= 2][:10]  # 取前10个关键词
    
    # 简单的关键词匹配查询
    cypher = f"""
    MATCH (aq:{label})
    WHERE any(kw IN $keywords WHERE aq.title CONTAINS kw)
       OR aq.title CONTAINS $query_short
    RETURN aq.id AS id, aq.title AS title, labels(aq) AS labels
    LIMIT $top_k
    """
    
    try:
        db_name = get_database_for_namespace(namespace or "")
        with driver.session(database=db_name) as session:
            result = session.run(
                cypher, 
                keywords=list(query),  # 按字符搜索
                query_short=query[:20],  # 取前20字符
                top_k=top_k
            )
            
            matches = []
            for record in result:
                # 从标签中提取命名空间
                ns = None
                for label in record["labels"]:
                    if ".AtomicQuery" in label:
                        ns = label.split(".")[0]
                        break
                
                matches.append({
                    "id": record["id"],
                    "title": record["title"],
                    "namespace": ns
                })
            
            logger.info("AtomicQuery 匹配: 找到 %d 条", len(matches))
            return matches
            
    except Exception as e:
        logger.warning("AtomicQuery 搜索失败: %s", e)
    
    return []


def search_chunks_via_atomic_query_similarity(
    query: str,
    target_namespace: str = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    通过 AtomicQuery 向量相似度匹配获取相关 Chunk
    
    策略：
    1. 向量匹配用户问题到 AtomicQuery.title
    2. 通过 sourceChunk 关系获取相关 Chunk
    3. 优先返回目标教材的结果
    
    Args:
        query: 用户问题
        target_namespace: 目标教材命名空间
        top_k: 返回数量
        
    Returns:
        Chunk 列表，包含来源 AtomicQuery 信息
    """
    import requests
    
    driver = get_neo4j_driver()
    if driver is None:
        return []
    
    # 直接使用已知的 AtomicQuery 向量索引名
    aq_index_name = "_medicine_atomic_query_title_vector_index"
    
    # 向量化配置
    VECTORIZE_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
    VECTORIZE_BASE_URL = "https://api.siliconflow.cn/v1"
    VECTORIZE_MODEL = "Qwen/Qwen3-Embedding-8B"
    
    # 生成查询向量
    try:
        headers = {
            "Authorization": f"Bearer {VECTORIZE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": VECTORIZE_MODEL, "input": query[:500]}
        response = requests.post(
            f"{VECTORIZE_BASE_URL}/embeddings", 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        response.raise_for_status()
        query_vector = response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.warning("向量化查询失败: %s", e)
        return []
    
    # 通过 AtomicQuery 向量搜索获取 Chunk
    try:
        db_name = get_database_for_namespace("")
        with driver.session(database=db_name) as session:
            # 先向量搜索 AtomicQuery，然后通过 sourceChunk 关系获取 Chunk
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    $index_name,
                    $top_k,
                    $query_vector
                ) YIELD node as aq, score as aq_score
                WHERE aq_score > 0.6
                
                MATCH (aq)-[:sourceChunk]->(c)
                WHERE any(l IN labels(c) WHERE l CONTAINS '.Chunk')
                
                RETURN DISTINCT 
                    c.id AS chunk_id, 
                    c.content AS content, 
                    labels(c) AS chunk_labels,
                    aq.title AS matched_question,
                    aq_score
            """, index_name=aq_index_name, query_vector=query_vector, top_k=top_k * 2)
            
            chunks = []
            for record in result:
                # 从标签中提取命名空间（优先非 Medicine）
                ns = None
                for label in record["chunk_labels"]:
                    if ".Chunk" in label and label != "Medicine.Chunk":
                        ns = label.split(".")[0]
                        break
                if not ns:
                    for label in record["chunk_labels"]:
                        if ".Chunk" in label:
                            ns = label.split(".")[0]
                            break
                
                chunks.append({
                    "id": record["chunk_id"],
                    "content": record["content"] or "",
                    "namespace": ns,
                    "score": record["aq_score"],
                    "matched_question": record["matched_question"],
                    "source": "atomic_query_similarity"
                })
            
            # 按教材匹配度和相似度排序
            def sort_key(chunk):
                ns_match = 1 if chunk.get("namespace") == target_namespace else 0
                return (ns_match, chunk.get("score", 0))
            
            chunks.sort(key=sort_key, reverse=True)
            chunks = chunks[:top_k]
            
            if chunks:
                logger.info(
                    "通过 AtomicQuery 相似度获取 Chunk: %d 条, 匹配问题示例: %s",
                    len(chunks), 
                    chunks[0].get("matched_question", "")[:50] if chunks else ""
                )
            
            return chunks
            
    except Exception as e:
        logger.warning("AtomicQuery 向量检索失败: %s", e)
    
    return []


def search_chunks_via_sub_knowledge_unit(
    query: str,
    target_namespace: str = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    通过 SubKnowledgeUnit 向量相似度匹配获取相关 Chunk
    
    策略：
    1. 向量匹配用户问题到 SubKnowledgeUnit.name
    2. 通过 sourceChunk 关系获取相关 Chunk
    3. 优先返回目标教材的结果
    
    Args:
        query: 用户问题
        target_namespace: 目标教材命名空间
        top_k: 返回数量
        
    Returns:
        Chunk 列表，包含来源 SubKnowledgeUnit 信息
    """
    import requests
    
    driver = get_neo4j_driver()
    if driver is None:
        return []
    
    # 直接使用已知的 SubKnowledgeUnit 向量索引名
    sku_index_name = "_medicine_sub_knowledge_unit_name_vector_index"
    
    # 向量化配置
    VECTORIZE_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
    VECTORIZE_BASE_URL = "https://api.siliconflow.cn/v1"
    VECTORIZE_MODEL = "Qwen/Qwen3-Embedding-8B"
    
    # 生成查询向量
    try:
        headers = {
            "Authorization": f"Bearer {VECTORIZE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": VECTORIZE_MODEL, "input": query[:500]}
        response = requests.post(
            f"{VECTORIZE_BASE_URL}/embeddings", 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        response.raise_for_status()
        query_vector = response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.warning("向量化查询失败: %s", e)
        return []
    
    # 通过 SubKnowledgeUnit 向量搜索获取 Chunk
    try:
        db_name = get_database_for_namespace("")
        with driver.session(database=db_name) as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    $index_name,
                    $top_k,
                    $query_vector
                ) YIELD node as sku, score as sku_score
                WHERE sku_score > 0.6
                
                MATCH (sku)-[:sourceChunk]->(c)
                WHERE any(l IN labels(c) WHERE l CONTAINS '.Chunk')
                
                RETURN DISTINCT 
                    c.id AS chunk_id, 
                    c.content AS content, 
                    labels(c) AS chunk_labels,
                    sku.name AS matched_knowledge_point,
                    sku_score
            """, index_name=sku_index_name, query_vector=query_vector, top_k=top_k * 2)
            
            chunks = []
            for record in result:
                ns = None
                for label in record["chunk_labels"]:
                    if ".Chunk" in label and label != "Medicine.Chunk":
                        ns = label.split(".")[0]
                        break
                if not ns:
                    for label in record["chunk_labels"]:
                        if ".Chunk" in label:
                            ns = label.split(".")[0]
                            break
                
                chunks.append({
                    "id": record["chunk_id"],
                    "content": record["content"] or "",
                    "namespace": ns,
                    "score": record["sku_score"],
                    "matched_knowledge_point": record["matched_knowledge_point"],
                    "source": "sub_knowledge_unit"
                })
            
            def sort_key(chunk):
                ns_match = 1 if chunk.get("namespace") == target_namespace else 0
                return (ns_match, chunk.get("score", 0))
            
            chunks.sort(key=sort_key, reverse=True)
            chunks = chunks[:top_k]
            
            if chunks:
                logger.info(
                    "通过 SubKnowledgeUnit 相似度获取 Chunk: %d 条, 匹配知识点: %s",
                    len(chunks), 
                    chunks[0].get("matched_knowledge_point", "")[:40] if chunks else ""
                )
            
            return chunks
            
    except Exception as e:
        logger.warning("SubKnowledgeUnit 向量检索失败: %s", e)
    
    return []


def get_chunks_from_atomic_queries(
    atomic_query_ids: List[str],
    namespace: str = None,
    query: str = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    通过 AtomicQuery 的 sourceChunk 关系获取 Chunk，并按向量相似度排序
    
    Args:
        atomic_query_ids: AtomicQuery ID 列表
        namespace: 命名空间
        query: 用户问题（用于计算向量相似度）
        top_k: 返回的最大 chunk 数量
        
    Returns:
        Chunk 列表，按向量相似度排序，每个包含 id, content, namespace, score
    """
    if not atomic_query_ids:
        return []
    
    driver = get_neo4j_driver()
    if driver is None:
        return []
    
    # 如果有 query，先获取 query 的向量
    query_vector = None
    if query:
        import requests
        VECTORIZE_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
        VECTORIZE_BASE_URL = "https://api.siliconflow.cn/v1"
        VECTORIZE_MODEL = "Qwen/Qwen3-Embedding-8B"
        
        try:
            headers = {
                "Authorization": f"Bearer {VECTORIZE_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {"model": VECTORIZE_MODEL, "input": query[:500]}
            resp = requests.post(f"{VECTORIZE_BASE_URL}/embeddings", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            query_vector = resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning("获取 query 向量失败: %s", e)
    
    # 构建查询 - 如果有 query_vector，同时获取 chunk 的向量用于计算相似度
    if query_vector:
        cypher = """
        MATCH (aq)-[:sourceChunk]->(c)
        WHERE aq.id IN $aq_ids
        RETURN DISTINCT c.id AS id, c.content AS content, labels(c) AS labels, c._content_vector AS vec
        """
    else:
        cypher = """
        MATCH (aq)-[:sourceChunk]->(c)
        WHERE aq.id IN $aq_ids
        RETURN DISTINCT c.id AS id, c.content AS content, labels(c) AS labels
        """
    
    try:
        db_name = get_database_for_namespace(namespace or "")
        with driver.session(database=db_name) as session:
            result = session.run(cypher, aq_ids=atomic_query_ids)
            
            chunks = []
            for record in result:
                # 从标签中提取命名空间
                ns = None
                for label in record["labels"]:
                    if ".Chunk" in label:
                        ns = label.split(".")[0]
                        break
                
                chunk = {
                    "id": record["id"],
                    "content": record["content"] or "",
                    "namespace": ns,
                    "score": 0.5  # 默认分数
                }
                
                # 如果有向量，计算相似度
                if query_vector and record.get("vec"):
                    chunk_vec = record["vec"]
                    # 余弦相似度计算
                    import numpy as np
                    q_vec = np.array(query_vector)
                    c_vec = np.array(chunk_vec)
                    similarity = np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8)
                    chunk["score"] = float(similarity)
                
                chunks.append(chunk)
            
            # 按相似度分数排序，取前 top_k 个
            chunks.sort(key=lambda x: x["score"], reverse=True)
            chunks = chunks[:top_k]
            
            logger.info("从 AtomicQuery 获取 Chunk: 总计 %d 条, 取相似度最高 %d 条, 最高分: %.3f",
                       len(result._records) if hasattr(result, '_records') else len(chunks),
                       len(chunks),
                       chunks[0]["score"] if chunks else 0)
            return chunks
            
    except Exception as e:
        logger.warning("获取 Chunk 失败: %s", e)
    
    return []


def search_chunks_direct(
    query: str,
    namespace: str = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    直接从 Neo4j 使用向量搜索 Chunk（绕过 OpenSPG API）
    
    使用 Qwen3-Embedding-8B 模型生成查询向量，
    然后调用 Neo4j 向量索引进行搜索
    """
    import requests
    
    driver = get_neo4j_driver()
    if driver is None:
        return []
    
    # 向量化配置（与 kag_config.yaml 一致）
    VECTORIZE_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
    VECTORIZE_BASE_URL = "https://api.siliconflow.cn/v1"
    VECTORIZE_MODEL = "Qwen/Qwen3-Embedding-8B"
    
    # 生成查询向量
    try:
        headers = {
            "Authorization": f"Bearer {VECTORIZE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": VECTORIZE_MODEL, "input": query[:1000]}
        response = requests.post(
            f"{VECTORIZE_BASE_URL}/embeddings", 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        response.raise_for_status()
        query_vector = response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.warning("向量化查询失败: %s", e)
        return []
    
    # Neo4j 向量搜索
    try:
        db_name = get_database_for_namespace(namespace or "")
        with driver.session(database=db_name) as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    '_medicine_chunk_content_vector_index',
                    $top_k,
                    $query_vector
                ) YIELD node, score
                WHERE score > 0.5
                RETURN node.id AS id, node.content AS content, labels(node) AS labels, score
            """, query_vector=query_vector, top_k=top_k)
            
            chunks = []
            for record in result:
                # 从标签中提取命名空间
                ns = None
                for label in record["labels"]:
                    if ".Chunk" in label and label != "Medicine.Chunk":
                        ns = label.split(".")[0]
                        break
                if not ns:
                    for label in record["labels"]:
                        if ".Chunk" in label:
                            ns = label.split(".")[0]
                            break
                
                chunks.append({
                    "id": record["id"],
                    "content": record["content"] or "",
                    "namespace": ns,
                    "score": record["score"]
                })
            
            logger.info("Neo4j 向量搜索: 找到 %d 条 (top score: %.3f)", 
                       len(chunks), chunks[0]["score"] if chunks else 0)
            return chunks
            
    except Exception as e:
        logger.warning("Neo4j 向量搜索失败: %s", e)
    
    return []


def search_chunks_with_textbook_filter(
    query: str,
    target_namespace: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    带教材过滤的向量检索
    
    检索所有 Chunk，但优先返回目标教材的内容
    
    Args:
        query: 用户问题
        target_namespace: 目标教材命名空间（如 "Pharmacology"）
        top_k: 返回数量
        
    Returns:
        Chunk 列表，按教材匹配度和相似度排序
    """
    import requests
    
    driver = get_neo4j_driver()
    if driver is None:
        return []
    
    # 向量化配置
    VECTORIZE_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
    VECTORIZE_BASE_URL = "https://api.siliconflow.cn/v1"
    VECTORIZE_MODEL = "Qwen/Qwen3-Embedding-8B"
    
    # 生成查询向量
    try:
        headers = {
            "Authorization": f"Bearer {VECTORIZE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": VECTORIZE_MODEL, "input": query[:1000]}
        response = requests.post(
            f"{VECTORIZE_BASE_URL}/embeddings", 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        response.raise_for_status()
        query_vector = response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.warning("向量化查询失败: %s", e)
        return []
    
    # Neo4j 向量搜索
    try:
        db_name = get_database_for_namespace("")
        with driver.session(database=db_name) as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    '_medicine_chunk_content_vector_index',
                    $top_k,
                    $query_vector
                ) YIELD node, score
                WHERE score > 0.5
                RETURN node.id AS id, node.content AS content, labels(node) AS labels, score
            """, query_vector=query_vector, top_k=top_k * 2)  # 多检索一些
            
            all_chunks = []
            for record in result:
                # 从标签中提取命名空间（优先非 Medicine）
                ns = None
                for label in record["labels"]:
                    if ".Chunk" in label and label != "Medicine.Chunk":
                        ns = label.split(".")[0]
                        break
                if not ns:
                    for label in record["labels"]:
                        if ".Chunk" in label:
                            ns = label.split(".")[0]
                            break
                
                all_chunks.append({
                    "id": record["id"],
                    "content": record["content"] or "",
                    "namespace": ns,
                    "score": record["score"]
                })
            
            # 按教材匹配度 + 相似度排序
            def sort_key(chunk):
                ns_match = 1 if chunk.get("namespace") == target_namespace else 0
                return (ns_match, chunk.get("score", 0))
            
            all_chunks.sort(key=sort_key, reverse=True)
            
            # 返回前 top_k 个
            result_chunks = all_chunks[:top_k]
            
            # 统计
            ns_match = sum(1 for c in result_chunks if c.get("namespace") == target_namespace)
            logger.info("带教材过滤的向量检索: 总计%d条, 目标教材(%s)匹配%d条, 最高分:%.3f",
                       len(result_chunks), target_namespace, ns_match, 
                       result_chunks[0]["score"] if result_chunks else 0)
            
            return result_chunks
            
    except Exception as e:
        logger.warning("带教材过滤的向量检索失败: %s", e)
    
    return []


def smart_retrieve(
    query: str,
    default_namespace: str = "Medicine",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    多策略综合检索（教材分类优先 + 向量相似度 + 知识图谱关系）
    
    策略优先级：
    1. 教材分类：根据问题关键词判断应该检索哪个教材
    2. 先从分类结果对应的教材中向量检索
    3. 如果向量检索结果不足，补充知识图谱关系检索
    4. 最终按相似度排序返回
    
    Args:
        query: 用户问题
        default_namespace: 默认命名空间
        top_k: 返回数量
        
    Returns:
        {
            "textbook_classification": {...},
            "atomic_queries": [...],
            "chunks": [...],
            "effective_namespace": "Pharmacology"
        }
    """
    import requests
    
    # ============================================================
    # 策略1: 教材分类
    # ============================================================
    classification = classify_textbook(query)
    effective_namespace = classification["primary"] or default_namespace
    logger.info("教材分类: %s (置信度: %.2f)", effective_namespace, classification.get("confidence", 0))
    
    all_chunks = []
    
    # ============================================================
    # 策略2: 向量检索（带教材过滤）
    # ============================================================
    vector_chunks = search_chunks_with_textbook_filter(
        query=query,
        target_namespace=effective_namespace,
        top_k=top_k * 2  # 多检索一些，后续过滤
    )
    
    for chunk in vector_chunks:
        chunk["source"] = "vector"
    all_chunks.extend(vector_chunks)
    
    # ============================================================
    # 策略2.5: 通过 AtomicQuery 相似度获取 Chunk（精准问答匹配）
    # ============================================================
    aq_sim_chunks = search_chunks_via_atomic_query_similarity(
        query=query,
        target_namespace=effective_namespace,
        top_k=top_k
    )
    
    for chunk in aq_sim_chunks:
        # 如果不在已有列表中，添加进去
        if not any(c["id"] == chunk["id"] for c in all_chunks):
            all_chunks.append(chunk)
    
    # ============================================================
    # 策略2.6: 通过 SubKnowledgeUnit 相似度获取 Chunk（子知识点匹配）
    # ============================================================
    sku_chunks = search_chunks_via_sub_knowledge_unit(
        query=query,
        target_namespace=effective_namespace,
        top_k=top_k
    )
    
    for chunk in sku_chunks:
        if not any(c["id"] == chunk["id"] for c in all_chunks):
            all_chunks.append(chunk)
    
    # ============================================================
    # 策略3: 知识图谱关系检索（通过 AtomicQuery 关键词匹配）
    # ============================================================
    atomic_queries = search_atomic_queries(
        query=query,
        namespace=effective_namespace,
        top_k=top_k
    )
    
    if atomic_queries:
        aq_ids = [aq["id"] for aq in atomic_queries]
        kg_chunks = get_chunks_from_atomic_queries(aq_ids, effective_namespace, query=query, top_k=3)
        
        # 标记来源并添加
        for chunk in kg_chunks:
            chunk["source"] = "knowledge_graph"
            # 如果不在已有列表中，添加进去
            if not any(c["id"] == chunk["id"] for c in all_chunks):
                all_chunks.append(chunk)
    
    # ============================================================
    # 策略4: 合并、去重、排序
    # ============================================================
    # 按教材匹配度和来源优先级排序
    def sort_key(chunk):
        # 教材匹配度（目标教材优先）
        ns_match = 1 if chunk.get("namespace") == effective_namespace else 0
        # 向量相似度分数
        score = chunk.get("score", 0.5)
        # 来源优先级（AtomicQuery相似度 > SubKnowledgeUnit > 知识图谱 > 向量）
        source = chunk.get("source", "")
        if source == "atomic_query_similarity":
            source_priority = 4
        elif source == "sub_knowledge_unit":
            source_priority = 3
        elif source == "knowledge_graph":
            source_priority = 2
        else:
            source_priority = 1
        
        return (ns_match, source_priority, score)
    
    all_chunks.sort(key=sort_key, reverse=True)
    
    # 去重并限制数量
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            unique_chunks.append(chunk)
            if len(unique_chunks) >= top_k * 2:  # 先多取一些，后续过滤
                break
    
    # ============================================================
    # 严格教材过滤：当分类置信度高时，只保留目标教材的 Chunk
    # ============================================================
    confidence = classification.get("confidence", 0)
    target_namespaces = [effective_namespace]  # 主要教材
    
    # 如果有 secondary 教材且置信度不是100%，也可以包含
    if confidence < 1.0 and classification.get("secondary"):
        target_namespaces.extend(classification["secondary"])
    
    if confidence >= 0.7:
        # 高置信度：严格过滤，只保留目标教材
        filtered_chunks = [
            c for c in unique_chunks 
            if c.get("namespace") in target_namespaces
        ]
        
        if filtered_chunks:
            logger.info(
                "严格教材过滤: 置信度%.2f, 目标教材=%s, 过滤后保留 %d/%d 条",
                confidence, target_namespaces, len(filtered_chunks), len(unique_chunks)
            )
            unique_chunks = filtered_chunks[:top_k]
        else:
            # 过滤后为空，回退到所有结果但记录警告
            logger.warning(
                "严格过滤后无结果! 目标教材=%s 在检索结果中无匹配。"
                "可能原因: 1)该教材尚未导入知识库 2)知识点不在已导入章节中。"
                "回退到所有检索结果。",
                target_namespaces
            )
            unique_chunks = unique_chunks[:top_k]
    else:
        # 低置信度：保留所有结果但优先排序目标教材
        unique_chunks = unique_chunks[:top_k]
        logger.info("低置信度(%.2f)，不执行严格过滤，返回所有教材结果", confidence)
    
    # LLM 验证（已禁用，向量相似度已足够筛选，启用会增加约 6 秒延迟）
    # if unique_chunks:
    #     chunks_to_verify = unique_chunks[:3]
    #     verified_chunks = verify_chunk_relevance_with_llm(query, chunks_to_verify, top_k=3)
    #     verified_ids = {c["id"] for c in verified_chunks}
    #     for chunk in unique_chunks:
    #         chunk["llm_verified"] = chunk["id"] in verified_ids
    
    result = {
        "textbook_classification": classification,
        "atomic_queries": atomic_queries,
        "chunks": unique_chunks,
        "effective_namespace": effective_namespace,
        "strict_filter_applied": confidence >= 0.7
    }
    
    # 统计信息
    ns_match_count = sum(1 for c in unique_chunks if c.get("namespace") == effective_namespace)
    llm_verified_count = sum(1 for c in unique_chunks if c.get("llm_verified") == True)
    logger.info("智能检索完成: 教材=%s, 匹配教材的Chunk=%d/%d, LLM验证通过=%d",
                effective_namespace, ns_match_count, len(unique_chunks), llm_verified_count)
    
    return result


def verify_chunk_relevance_with_llm(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    使用 LLM 验证 Chunk 与问题的相关性
    
    只对相似度最高的 top_k 个 Chunk 进行验证，
    过滤掉 LLM 认为不相关的内容
    
    Args:
        query: 用户问题
        chunks: Chunk 列表（已按相似度排序）
        top_k: 验证的 Chunk 数量
        
    Returns:
        经过 LLM 验证的 Chunk 列表
    """
    import requests
    
    if not chunks:
        return []
    
    # 只验证前 top_k 个
    chunks_to_verify = chunks[:top_k]
    
    # LLM 配置
    LLM_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
    LLM_BASE_URL = "https://api.siliconflow.cn/v1"
    LLM_MODEL = "Qwen/Qwen3-8B"
    
    verified_chunks = []
    
    for chunk in chunks_to_verify:
        content = chunk.get("content", "")[:500]
        if not content.strip():
            continue
        
        # 构建验证 prompt（更严格的验证条件）
        prompt = f"""请严格判断以下内容片段是否能**直接回答**用户的问题。

用户问题：{query}

内容片段：
{content}

判断标准（必须同时满足）：
1. 内容必须**直接**讨论问题的核心主题
2. 内容必须能提供回答问题所需的**关键信息**
3. 仅仅包含相似的关键词但主题不同的内容，应判断为"不相关"

例如：
- 问题问"损伤的原因"，内容讲"损伤后的修复"→ 不相关
- 问题问"药物作用"，内容讲"药物代谢"→ 不相关

根据以上标准，这段内容是否能直接回答问题？
只回答"相关"或"不相关"。"""

        try:
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            answer = response.json()["choices"][0]["message"]["content"].strip()
            is_relevant = "相关" in answer and "不相关" not in answer
            
            if is_relevant:
                chunk["llm_verified"] = True
                verified_chunks.append(chunk)
                logger.info("LLM 验证通过: %s... (来源: %s)", 
                           content[:30], chunk.get("source", "unknown"))
            else:
                logger.info("LLM 验证未通过: %s...", content[:30])
                
        except Exception as e:
            logger.warning("LLM 验证失败: %s，保留 Chunk", e)
            chunk["llm_verified"] = "error"
            verified_chunks.append(chunk)
    
    if not verified_chunks:
        logger.warning("LLM 验证后无有效结果，使用原始 Chunk")
        return chunks_to_verify
    
    logger.info("LLM 验证完成: %d/%d 个 Chunk 通过", len(verified_chunks), len(chunks_to_verify))
    return verified_chunks


def detect_namespace_from_chunk_labels(chunk_biz_id: str) -> Optional[str]:
    """
    从 Chunk 的 Neo4j 标签中动态检测命名空间前缀。
    
    例如：
    - 如果 Chunk 有标签 `Pharmacology.Chunk`，返回 "Pharmacology"
    - 如果有标签 `Pathology.Chunk`，返回 "Pathology"
    
    Args:
        chunk_biz_id: Chunk 的业务 ID
        
    Returns:
        检测到的命名空间（如 "Pharmacology"），或 None
    """
    driver = get_neo4j_driver()
    if driver is None:
        return None
    
    # 查询 Chunk 节点的所有标签
    query = """
    MATCH (c)
    WHERE c.id = $chunk_id
    RETURN labels(c) AS labels
    LIMIT 1
    """
    
    try:
        db_name = get_database_for_namespace("")
        with driver.session(database=db_name) as session:
            result = session.run(query, chunk_id=chunk_biz_id)
            record = result.single()
            
            if record and record["labels"]:
                # 优先查找非 Medicine 的命名空间（更精确）
                for label in record["labels"]:
                    if ".Chunk" in label:
                        namespace = label.split(".")[0]
                        # 优先返回具体教材命名空间（包括小写的 physiology）
                        if namespace in ["Pharmacology", "Pathology", "physiology"]:
                            logger.info("从标签检测到原始命名空间: %s", namespace)
                            return namespace
                
                # 如果没有找到，再检查是否有 Medicine
                for label in record["labels"]:
                    if ".Chunk" in label:
                        namespace = label.split(".")[0]
                        logger.info("从标签检测到命名空间: %s", namespace)
                        return namespace
    except Exception as e:
        logger.warning("检测命名空间失败: %s", e)
    
    return None


def fetch_chunk_provenance_direct(namespace: str, chunk_biz_id: str) -> Optional[Dict[str, Any]]:
    """
    直接从 Neo4j 查询 Chunk 的溯源信息
    
    策略：
    1. 首先尝试直接读取 Chunk 的 section_title/chapter_title 属性
    2. 然后通过 section_id 属性找到 Section -> Chapter -> Textbook 关系链
    
    Args:
        namespace: 命名空间，如 "Pharmaco8"
        chunk_biz_id: Chunk 的业务 ID
        
    Returns:
        包含 textbook, chapter, section 等信息的字典，或 None
    """
    driver = get_neo4j_driver()
    if driver is None:
        return None
    
    chunk_label = f"`{namespace}.Chunk`"
    section_label = f"`{namespace}.Section`"
    chapter_label = f"`{namespace}.Chapter`"
    textbook_label = f"`{namespace}.Textbook`"
    
    # 主查询：通过 Chunk 的 section_id 属性找到 Section，然后溯源到 Textbook
    # 注意：section_id 可能带引号，需要用 replace 去除
    query = f"""
    MATCH (c:{chunk_label})
    WHERE c.id = $chunk_id
    
    // 获取 Chunk 的直接属性
    WITH c, 
         c.section_title AS direct_section_title,
         c.chapter_title AS direct_chapter_title,
         replace(replace(c.section_id, '"', ''), "'", "") AS clean_section_id
    
    // 通过 section_id 找 Section
    OPTIONAL MATCH (s:{section_label})
    WHERE s.id = clean_section_id
    
    // 从 Section 向上找 Chapter
    OPTIONAL MATCH (ch:{chapter_label})-[:hasSections]->(s)
    
    // 从 Chapter 向上找 Textbook
    OPTIONAL MATCH (b:{textbook_label})-[:hasChapters]->(ch)
    
    RETURN 
        coalesce(b.name, '') AS textbook,
        coalesce(ch.name, direct_chapter_title, '') AS chapter,
        coalesce(s.name, direct_section_title, '') AS section
    LIMIT 1
    """
    
    try:
        db_name = get_database_for_namespace(namespace)
        with driver.session(database=db_name) as session:
            result = session.run(query, chunk_id=chunk_biz_id)
            record = result.single()
            
            if record:
                out = {}
                # 去除可能的引号
                textbook = (record["textbook"] or "").strip().strip('"').strip("'")
                chapter = (record["chapter"] or "").strip().strip('"').strip("'")
                section = (record["section"] or "").strip().strip('"').strip("'")
                
                if textbook:
                    out["textbook"] = textbook
                if chapter:
                    out["chapter"] = chapter
                if section:
                    out["section"] = section
                
                if out:
                    logger.info("直连溯源成功: %s", out)
                    return out
                else:
                    logger.warning("直连溯源返回空结果 chunk_id=%s", chunk_biz_id)
    except Exception as e:
        logger.error("直连查询失败: %s", e)
    
    return None


def fetch_merged_chunk_content(
    namespace: str, 
    chunk_biz_id: str, 
    query: str = "",
    vectorize_model=None
) -> Optional[str]:
    """
    获取当前 Chunk 及其相邻 Chunk（按 ID 排序），合并并补全断句。
    
    策略：
    1. 按 ID 排序获取同 Section 的所有 Chunk
    2. 找到当前 Chunk 位置，取前后各 1 个相邻 Chunk
    3. 去除重叠部分合并
    4. 补全首尾断句
    """
    driver = get_neo4j_driver()
    if driver is None:
        return None
    
    chunk_label = f"`{namespace}.Chunk`"
    
    # 获取同 Section 的所有 Chunk，按 sequence 排序（sequence 才是正确顺序）
    cypher = f"""
    MATCH (target:{chunk_label})
    WHERE target.id = $chunk_id
    WITH target.section_id AS sid
    
    MATCH (c:{chunk_label})
    WHERE c.section_id = sid
    RETURN c.id AS cid, c.content AS content, c.sequence AS seq
    ORDER BY c.sequence
    """
    
    try:
        db_name = get_database_for_namespace(namespace)
        with driver.session(database=db_name) as session:
            result = session.run(cypher, chunk_id=chunk_biz_id)
            records = list(result)
            
            if not records:
                return None
            
            # 构建有序列表，找到目标位置
            chunks = [{"id": r["cid"], "content": r["content"] or ""} for r in records]
            target_idx = None
            for i, c in enumerate(chunks):
                if c["id"] == chunk_biz_id:
                    target_idx = i
                    break
            
            if target_idx is None:
                return None
            
            # 只取相邻的 Chunk（前一个、当前、后一个）
            target_content = chunks[target_idx]["content"]
            prev_content = chunks[target_idx - 1]["content"] if target_idx > 0 else ""
            next_content = chunks[target_idx + 1]["content"] if target_idx < len(chunks) - 1 else ""
            
            logger.info("当前 Chunk 位置: %d/%d", target_idx, len(chunks))
            
            # 核心逻辑：确保开头是段落/句子的开始
            result = _ensure_paragraph_start(target_content, prev_content)
            
            # 补全结尾断句（如果需要）
            result = _ensure_sentence_end(result, next_content)
            
            # 按段落筛选：只保留与查询相关的段落
            if vectorize_model and query:
                result = _filter_relevant_paragraphs(result, query, vectorize_model)
            
            # 最终清理
            result = _fix_sentence_boundaries(result)
            
            logger.info("最终内容长度: %d", len(result))
            return result
            
    except Exception as e:
        logger.error("合并 Chunk 失败: %s", e)
    
    return None


def _filter_relevant_paragraphs(content: str, query: str, vectorize_model, min_similarity: float = 0.6) -> str:
    """
    按段落筛选内容，只保留与查询相关的段落。
    
    策略：
    1. 按 \n 切分段落
    2. 计算每个段落与查询的相似度
    3. 只保留相似度 >= 阈值的段落（阈值 0.6）
    """
    import numpy as np
    
    if not content or not query or not vectorize_model:
        return content
    
    try:
        query_vec = vectorize_model.vectorize(query[:500])
        
        # 按 \n 切分段落（同时处理真正换行符和字面 \n 字符串）
        import re
        # 先把字面的 \\n 或 \n 替换为真正换行符
        normalized = content.replace('\\n', '\n')
        paragraphs = normalized.split('\n')
        relevant_paragraphs = []
        filtered_count = 0
        last_was_title = False  # 跟踪上一个是否为标题
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 标题总是保留
            if para.startswith('#'):
                relevant_paragraphs.append(para)
                last_was_title = True
                continue
            
            # 紧跟在标题后的第一个段落保留
            if last_was_title:
                relevant_paragraphs.append(para)
                last_was_title = False
                continue
            
            last_was_title = False
            
            # 太短的段落保留（可能是序号等）
            if len(para) < 15:
                relevant_paragraphs.append(para)
                continue
            
            # 计算相似度
            try:
                para_vec = vectorize_model.vectorize(para[:300])
                dot = np.dot(query_vec, para_vec)
                norm = np.linalg.norm(query_vec) * np.linalg.norm(para_vec)
                similarity = float(dot / norm) if norm > 0 else 0
                
                if similarity >= min_similarity:
                    relevant_paragraphs.append(para)
                else:
                    filtered_count += 1
                    logger.info("段落过滤 (相似度 %.3f < %.2f): %s...", similarity, min_similarity, para[:40])
            except Exception as e:
                logger.debug("段落向量化失败: %s", e)
                relevant_paragraphs.append(para)
        
        if relevant_paragraphs:
            result = '\n'.join(relevant_paragraphs)
            logger.info("段落筛选完成: 保留 %d 个，过滤 %d 个", len(relevant_paragraphs), filtered_count)
            return result
        
    except Exception as e:
        logger.warning("段落筛选失败: %s", e)
    
    return content


def _ensure_paragraph_start(target_content: str, prev_content: str) -> str:
    """
    确保内容开头是完整句子的开始。
    
    策略：通过检测与前 Chunk 的重叠来判断
    1. 取当前 Chunk 开头的一部分，在前 Chunk 中查找
    2. 如果找到（有重叠），从前 Chunk 的重叠位置向前找最近的句号
    3. 从句号之后开始就是完整句子的开始
    """
    import re
    
    target_content = target_content.strip()
    if not target_content:
        return target_content
    
    # 1. 检测开头是否已经是明显的段落开始（标题/序号）
    if re.match(r'^(#|[一二三四五六七八九十]|（[一二三四五六七八九十]|[0-9])', target_content):
        return target_content
    
    # 2. 通过重叠检测判断开头是否完整
    if prev_content:
        # 取当前 chunk 开头 30-50 个字符去前 chunk 中查找
        for sample_len in range(min(50, len(target_content)), 10, -5):
            sample = target_content[:sample_len]
            pos = prev_content.find(sample)
            
            if pos != -1:
                # 找到重叠！从 pos 向前找最近的句号
                sentence_ends = ('。', '！', '？', '.', '!', '?')
                
                for i in range(pos - 1, -1, -1):
                    if prev_content[i] in sentence_ends:
                        # 找到句号，从句号之后到当前 chunk 开头就是需要补全的部分
                        prefix = prev_content[i + 1:pos].strip()
                        if prefix:
                            logger.info("通过重叠检测补全开头（%d 字符）: %s...", len(prefix), prefix[:30])
                            return prefix + target_content
                        break
                
                # 没找到句号，说明前 chunk 开头就是句子开始
                prefix = prev_content[:pos].strip()
                if prefix:
                    logger.info("从前 Chunk 开头补全（%d 字符）", len(prefix))
                    return prefix + target_content
                break
    
    # 3. 如果当前 chunk 中有标题，从标题开始
    first_newline = target_content.find('\n')
    if first_newline > 0 and first_newline < 200:
        after_newline = target_content[first_newline + 1:].lstrip()
        if after_newline and re.match(r'^(#|[一二三四五六七八九十]|（[一二三四五六七八九十]|[0-9])', after_newline):
            logger.info("从当前 Chunk 的标题开始（跳过 %d 字符）", first_newline)
            return target_content[first_newline:].lstrip()
    
    return target_content


def _find_sentence_start_in_prev(target_content: str, prev_content: str) -> str:
    """
    在 prev_content 中找到与 target_content 重叠的部分，
    然后向前找到句子的开始（句号后面）
    """
    # 先找重叠：target 开头在 prev 中的位置
    target_sample = target_content[:100]
    
    for overlap_len in range(min(50, len(target_sample)), 5, -1):
        overlap_text = target_sample[:overlap_len]
        pos = prev_content.find(overlap_text)
        if pos != -1:
            # 找到重叠，向前找句子开始（句号、换行、标题后面）
            prefix_area = prev_content[:pos]
            
            # 从后向前找最近的句子结束符
            for i in range(len(prefix_area) - 1, -1, -1):
                if prefix_area[i] in ('。', '！', '？', '\n', '#'):
                    return prefix_area[i+1:].lstrip()
            
            # 没找到句子结束符，返回整个 prefix
            return prefix_area
    
    return ""


def _ensure_sentence_end(content: str, next_content: str) -> str:
    """
    确保内容结尾是完整句子（以句号结尾）
    如果不是，从 next_content 补全
    """
    content = content.rstrip()
    if not content:
        return content
    
    sentence_ends = ('。', '！', '？', '.', '!', '?')
    
    if content[-1] in sentence_ends:
        return content
    
    # 结尾不完整，从 next_content 补全
    if next_content:
        # 找 content 结尾在 next_content 中的重叠
        content_end = content[-100:] if len(content) > 100 else content
        
        for overlap_len in range(min(50, len(content_end)), 5, -1):
            overlap_text = content_end[-overlap_len:]
            pos = next_content.find(overlap_text)
            if pos != -1:
                # 找到重叠，取到下一个句号
                suffix_area = next_content[pos + overlap_len:]
                for i, c in enumerate(suffix_area):
                    if c in sentence_ends:
                        logger.info("从后 Chunk 补全句子结尾（%d字符）", i+1)
                        return content + suffix_area[:i+1]
                break
    
    return content


# 以下是旧代码，已删除


def _find_prefix_from_overlap(prev_content: str, target_start: str, max_check: int = 100) -> str:
    """
    从前一个 Chunk 的结尾找到与当前 Chunk 开头重叠的部分，返回重叠之前的内容作为前缀。
    
    例如：
    prev_content = "...缺血、缺氧是细胞损伤最常见的原因。"
    target_start = "见的原因。\n生物性因素..."
    返回 "缺血、缺氧是细胞损伤最常" 作为前缀
    """
    # 取当前内容开头的一小部分用于匹配
    target_sample = target_start[:max_check]
    
    # 在前一个 Chunk 的结尾部分找重叠
    prev_end = prev_content[-500:] if len(prev_content) > 500 else prev_content
    
    # 尝试不同长度的重叠匹配
    for overlap_len in range(min(50, len(target_sample)), 5, -1):
        overlap_text = target_sample[:overlap_len]
        pos = prev_end.find(overlap_text)
        if pos != -1:
            # 找到重叠，返回重叠之前的部分（找到最近的句子开头）
            prefix = prev_end[:pos]
            # 从最后一个句号开始截取
            for i in range(len(prefix) - 1, -1, -1):
                if prefix[i] in ('。', '！', '？', '.', '!', '?', '\n', '#'):
                    return prefix[i+1:].lstrip()
            return prefix.lstrip()
    
    return ""


def _find_suffix_from_overlap(target_end: str, next_content: str, max_check: int = 100) -> str:
    """
    从后一个 Chunk 的开头找到与当前 Chunk 结尾重叠的部分，返回重叠之后的内容作为后缀。
    """
    # 取当前内容结尾的一小部分用于匹配
    target_sample = target_end[-max_check:] if len(target_end) > max_check else target_end
    
    # 在后一个 Chunk 的开头部分找重叠
    next_start = next_content[:500] if len(next_content) > 500 else next_content
    
    # 尝试不同长度的重叠匹配
    for overlap_len in range(min(50, len(target_sample)), 5, -1):
        overlap_text = target_sample[-overlap_len:]
        pos = next_start.find(overlap_text)
        if pos != -1:
            # 找到重叠，返回重叠之后的部分（到最近的句子结尾）
            suffix = next_start[pos + overlap_len:]
            # 截取到第一个句号
            for i, c in enumerate(suffix):
                if c in ('。', '！', '？', '.', '!', '?'):
                    return suffix[:i+1]
            return ""  # 没找到句号就不补
    
    return ""


def _merge_overlapping_chunks(contents: list) -> str:
    """
    合并有重叠的 Chunk 列表，去除重叠部分
    """
    if not contents:
        return ""
    if len(contents) == 1:
        return contents[0]
    
    result = contents[0]
    
    for i in range(1, len(contents)):
        current = contents[i]
        # 尝试找重叠：result 的结尾 vs current 的开头
        overlap_len = _find_overlap(result, current)
        if overlap_len > 0:
            result = result + current[overlap_len:]
        else:
            result = result + "\n" + current
    
    return result


def _find_overlap(text1: str, text2: str, min_overlap: int = 10, max_overlap: int = 200) -> int:
    """
    找两段文本的重叠长度（text1 结尾 = text2 开头）
    """
    for length in range(min(max_overlap, len(text1), len(text2)), min_overlap - 1, -1):
        if text1[-length:] == text2[:length]:
            return length
    return 0


def _fix_sentence_boundaries(text: str) -> str:
    """
    修复首尾断句和清理混入的 chunk id：
    - 清理混入的 64 位 hex chunk id
    - 如果开头不是句子开头，删除到第一个句号后
    - 如果结尾不是句子结尾，删除到最后一个句号
    """
    import re
    
    if not text:
        return text
    
    text = text.strip()
    
    # 清理混入的 chunk id（多种格式）
    # 1. 64 位 hex 字符串（可能带引号、点号、换行）
    text = re.sub(r'["\']?[a-f0-9]{64}["\']?[\s\n.]*', '', text)
    
    # 2. 清理中间的断裂模式：引号+换行+残余句子（如 '最常" \n失。'）
    # 模式：引号 + 空白/换行 + 1-5个字符 + 句号
    text = re.sub(r'["\'][\\n\s]*[^\n。！？.!?]{0,5}[。！？.!?]?[\s\n]*', '', text)
    
    # 3. 清理开头的 hex 字符
    if text and text[0] in 'abcdef0123456789':
        match = re.match(r'^[a-f0-9\s\n.]+', text)
        if match and len(match.group()) > 20:
            text = text[match.end():]
    
    text = text.strip()
    
    # 句子结束符
    sentence_ends = ('。', '！', '？', '.', '!', '?')
    
    # 1. 用正则清理开头的残余模式
    text = re.sub(r'^[\s\n\\n]*[^\s\n#]{1,8}[\s\n\\n]+', '', text)
    text = text.lstrip('\n\r\t ')
    
    # 2. 如果开头是不完整句子，且后面紧跟着标题（#），就删除开头不完整部分
    # 检测模式：开头一段内容 + 换行 + # 标题
    match = re.match(r'^[^#\n]*?\n\s*#', text, re.DOTALL)
    if match:
        # 找到 # 的位置
        hash_pos = text.find('\n#')
        if hash_pos == -1:
            hash_pos = text.find('#')
        if hash_pos > 0 and hash_pos < 150:  # 开头 150 字符内有标题
            text = text[hash_pos:].lstrip('\n\r\t ')
            logger.info("删除标题前的不完整开头（%d字符）", hash_pos)
    
    # 3. 检测开头是否是不完整的词语残余
    first_newline = text.find('\n')
    if 0 < first_newline < 10:
        first_part = text[:first_newline].strip()
        if not first_part.startswith('#') and len(first_part) < 8:
            text = text[first_newline + 1:].lstrip('\n\r\t ')
            logger.info("清理开头残余词语: '%s'", first_part)
    
    # 3. 如果开头是断句残余（少于 5 字符后跟句号）
    first_sentence_end = -1
    for i, c in enumerate(text[:20]):
        if c in sentence_ends:
            first_sentence_end = i
            break
    
    if 0 < first_sentence_end < 5:
        text = text[first_sentence_end + 1:].lstrip('\n\r\t ')
    
    # 4. 如果开头是标点符号，删除到第一个句号后
    first_char = text[0] if text else ''
    incomplete_start_chars = (',', '，', '、', '）', ')', ']', '】', '；', ';', '：', ':')
    
    if first_char in incomplete_start_chars:
        for i, c in enumerate(text):
            if c in sentence_ends:
                text = text[i+1:].lstrip('\n\r\t ')
                break
    
    # 检查结尾是否完整
    if text and text[-1] not in sentence_ends:
        # 找最后一个句号，删除之后的内容
        for i in range(len(text) - 1, -1, -1):
            if text[i] in sentence_ends:
                text = text[:i+1]
                break
    
    return text


def close_neo4j_driver():
    """关闭 Neo4j 连接"""
    global _neo4j_driver
    if _neo4j_driver:
        _neo4j_driver.close()
        _neo4j_driver = None
