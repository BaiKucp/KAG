# ============================================================
# KAG 兼容性补丁 - 必须在所有 KAG 导入之前执行
# ============================================================
import kag_compat_patch  # noqa: F401  # 给 RetrieverOutput 添加 chunk_datas 属性

import argparse
import asyncio
import json
import logging
import re

import medical_generator_prompt  # noqa: F401  # 关键：触发 PromptABC.register

from typing import Any, Dict, List, Optional

from kag.common.conf import init_env, KAG_CONFIG
from kag.interface import SolverPipelineABC, VectorizeModelABC
from kag.solver.reporter.trace_log_reporter import TraceLogReporter
from kag.common.tools.graph_api.graph_api_abc import GraphApiABC

logger = logging.getLogger(__name__)


def _clean_chunk_content(content: str, chunk_id: str) -> str:
    """
    清理 chunk content，去除开头可能包含的 chunk id
    
    有些 chunk 的 content 格式为: "chunk_id\n实际内容..." 或 '"chunk_id\n...'
    这个函数会检测并去除开头的 id
    """
    import re
    
    if not content:
        return content
    
    content = content.strip()
    
    # 去除开头可能的引号
    if content.startswith('"') or content.startswith("'"):
        content = content[1:]
    
    # 如果提供了 chunk_id 且 content 以它开头，去除它
    if chunk_id and content.startswith(chunk_id):
        content = content[len(chunk_id):].lstrip('\n').lstrip()
    
    # 无论 chunk_id 是否提供，都检查并移除开头的 64 位 hex 字符串（sha256 hash）
    # 支持换行符(\n, \\n)或其他空白字符分隔
    hex_pattern = r'^[a-f0-9]{64}(\\n|\n|\s)+'
    content = re.sub(hex_pattern, '', content, count=1)
    
    return content



def _iter_exception_chain(exc: BaseException, max_depth: int = 8):
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < max_depth:
        yield cur
        cur = cur.__cause__ or cur.__context__
        depth += 1


def _is_rate_limited(exc: BaseException) -> bool:
    for e in _iter_exception_chain(exc):
        name = e.__class__.__name__
        msg = str(e)
        if "RateLimitError" in name:
            return True
        if "RATE_LIMIT_EXCEEDED" in msg:
            return True
        if "Error code: 429" in msg:
            return True
        if " code: 429" in msg:
            return True
    return False


async def _ainvoke_with_backoff(
    pipeline: SolverPipelineABC,
    query: str,
    reporter: TraceLogReporter,
    *,
    max_retries: int = 6,
    base_sleep_s: float = 1.5,
    max_sleep_s: float = 30.0,
):
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return await pipeline.ainvoke(query, reporter=reporter)
        except Exception as e:
            last_exc = e
            if not _is_rate_limited(e):
                raise
            if attempt >= max_retries:
                break
            sleep_s = min(max_sleep_s, base_sleep_s * (2**attempt))
            sleep_s += random.uniform(0, 0.5)
            logger.warning(
                "LLM 触发 429 限流，%.1fs 后重试（%d/%d）",
                sleep_s,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"LLM 触发 429 限流，重试 {max_retries} 次仍失败") from last_exc


def _escape_dsl_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _table_to_dicts(table) -> List[Dict[str, Any]]:
    if not table or not getattr(table, "header", None) or not getattr(table, "data", None):
        return []
    return [{table.header[i]: row[i] for i in range(len(table.header))} for row in table.data]


def _compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    import math
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def filter_relevant_chunks(
    query: str,
    references: List[Dict[str, Any]],
    vectorize_model: VectorizeModelABC,
    answer: str = "",  # AI 生成的回答
    min_similarity: float = 0.5,
    max_chunks: int = 3  # 最多返回的chunk数量
) -> List[Dict[str, Any]]:
    """
    过滤引用，只保留与 AI 回答相似度足够高的 chunk。
    
    Args:
        query: 原始用户查询（备用）
        references: 原始引用列表
        vectorize_model: 向量化模型
        answer: AI 生成的回答（优先用于相似度计算）
        min_similarity: 最小相似度阈值
        max_chunks: 最多返回的chunk数量
        
    Returns:
        过滤后的引用列表（按与回答的相似度降序，最多 max_chunks 个）
    """
    if not references:
        return references
    
    try:
        # 优先用 AI 回答计算相似度，如果没有回答则用问题
        text_for_similarity = answer if answer else query
        text_vec = vectorize_model.vectorize(text_for_similarity[:1000])  # 限制长度
        scored_refs = []  # 存储 (相似度, ref) 元组
        
        for ref in references:
            info = ref.get("info", [])
            if not isinstance(info, list):
                continue
            
            # 取第一个 info 的 content 计算相似度
            for item in info:
                content = item.get("content", "")
                if not content:
                    continue
                
                # 只用前500字符计算相似度
                content_vec = vectorize_model.vectorize(content[:500])
                similarity = _compute_similarity(text_vec, content_vec)
                
                logger.info("Chunk 与回答相似度: %.3f (阈值: %.2f) - %s...", 
                           similarity, min_similarity, content[:50])
                
                if similarity >= min_similarity:
                    # 添加相似度分数到引用
                    item["similarity_score"] = round(similarity, 3)
                    scored_refs.append((similarity, ref))
                break
        
        # 按相似度降序排序
        scored_refs.sort(key=lambda x: x[0], reverse=True)
        
        # 同一 section 只保留相似度最高的
        seen_sections = set()
        deduplicated = []
        for similarity, ref in scored_refs:
            # 获取 section 信息
            info = ref.get("info", [])
            section = ""
            if info and isinstance(info, list):
                provenance = info[0].get("provenance", {})
                section = provenance.get("section", "")
            
            # 如果 section 为空或未见过，保留
            if not section or section not in seen_sections:
                deduplicated.append(ref)
                if section:
                    seen_sections.add(section)
                    logger.info("保留 section '%s' 的最高相似度引用 (%.3f)", section[:30], similarity)
        
        # 取前 max_chunks 个
        filtered = deduplicated[:max_chunks]
        
        # 按章节和小节顺序排序（而非相似度）
        def get_chapter_section_key(ref):
            info = ref.get("info", [])
            if info and isinstance(info, list):
                provenance = info[0].get("provenance", {})
                chapter = provenance.get("chapter", "") or ""
                section = provenance.get("section", "") or ""
                # 返回 (章节, 小节) 用于排序
                return (chapter, section)
            return ("", "")
        
        filtered.sort(key=get_chapter_section_key)
        
        logger.info("相似度过滤+去重+排序: %d/%d 个引用 (阈值: %.2f)", 
                   len(filtered), len(references), min_similarity)
        return filtered
        
    except Exception as e:
        logger.warning("相似度过滤失败: %s，返回原始引用", e)
        return references[:max_chunks]  # 即使失败也限制数量


def fetch_chunk_provenance(
    graph_api: GraphApiABC, namespace: str, chunk_biz_id: str
) -> Optional[Dict[str, Any]]:
    """Chunk -> KnowledgeUnit -> Section -> Chapter -> Textbook 溯源
    
    适配 schema: Textbook-[:hasChapters]->Chapter-[:hasSections]->Section-[:hasKnowledgeUnit]->KnowledgeUnit-[:sourceChunk]->Chunk
    """

    chunk_label = f"`{namespace}.Chunk`"
    knowledge_label = f"`{namespace}.KnowledgeUnit`"
    section_label = f"`{namespace}.Section`"
    chapter_label = f"`{namespace}.Chapter`"
    textbook_label = f"`{namespace}.Textbook`"

    cid = _escape_dsl_string(chunk_biz_id)

    # 主查询：KnowledgeUnit-[:sourceChunk]->Chunk，然后向上溯源
    dsl_1 = f"""
    MATCH (c:{chunk_label})
    WHERE c.id = "{cid}"
    MATCH (k:{knowledge_label})-[:sourceChunk]->(c)
    MATCH (s:{section_label})-[:hasKnowledgeUnit]->(k)
    MATCH (ch:{chapter_label})-[:hasSections]->(s)
    OPTIONAL MATCH (b:{textbook_label})-[:hasChapters]->(ch)
    RETURN b.name AS textbook, ch.name AS chapter, s.name AS section, k.name AS knowledge, c.pageNumber AS pageNumber, c.sequence AS sequence
    LIMIT 1
    """

    # 备用查询1：Chunk 直接有 hasKnowledgeUnit 关系（schema 中定义了）
    dsl_2 = f"""
    MATCH (c:{chunk_label})
    WHERE c.id = "{cid}"
    MATCH (c)-[:hasKnowledgeUnit]->(k:{knowledge_label})
    OPTIONAL MATCH (s:{section_label})-[:hasKnowledgeUnit]->(k)
    OPTIONAL MATCH (ch:{chapter_label})-[:hasSections]->(s)
    OPTIONAL MATCH (b:{textbook_label})-[:hasChapters]->(ch)
    RETURN b.name AS textbook, ch.name AS chapter, s.name AS section, k.name AS knowledge, c.pageNumber AS pageNumber, c.sequence AS sequence
    LIMIT 1
    """
    
    # 备用查询2：直接从 Section 的 hasKnowledgeUnit 跳到 Chunk (如果中间没有 KnowledgeUnit)
    dsl_3 = f"""
    MATCH (c:{chunk_label})
    WHERE c.id = "{cid}"
    OPTIONAL MATCH (s:{section_label})-[:hasKnowledgeUnit]->(:KnowledgeUnit)-[:sourceChunk]->(c)
    OPTIONAL MATCH (ch:{chapter_label})-[:hasSections]->(s)
    OPTIONAL MATCH (b:{textbook_label})-[:hasChapters]->(ch)
    RETURN b.name AS textbook, ch.name AS chapter, s.name AS section, c.pageNumber AS pageNumber, c.sequence AS sequence
    LIMIT 1
    """

    for dsl in (dsl_1, dsl_2, dsl_3):
        try:
            table = graph_api.execute_dsl(dsl)
            rows = _table_to_dicts(table)
            if rows:
                out = {}
                for k, v in rows[0].items():
                    if v is None:
                        continue
                    out[k] = v
                if out:  # 只有当获取到有效数据时才返回
                    logger.debug("溯源成功: %s", out)
                    return out
        except Exception:
            logger.exception("execute_dsl failed")

    logger.warning("无法溯源 chunk: %s", chunk_biz_id)
    return None


async def qa_with_provenance(query: str, config_path: str) -> Dict[str, Any]:
    """
    执行问答并进行知识溯源
    
    流程：
    1. 教材分类：根据问题关键词判断最可能的教材来源
    2. AtomicQuery 匹配：通过原子问快速定位相关内容
    3. KAG Pipeline 检索 + 生成：执行完整检索和回答生成
    4. 溯源：从 Chunk 标签动态检测教材章节来源
    
    Args:
        query: 用户问题
        config_path: KAG 配置文件路径
        
    Returns:
        包含 query, answer, status, references, smart_retrieval 的字典
    """
    init_env(config_file=config_path)

    # ============================================================
    # 智能检索阶段：教材分类 + AtomicQuery 匹配
    # ============================================================
    try:
        from neo4j_provenance import smart_retrieve, classify_textbook
        
        # 1. 教材分类
        textbook_info = classify_textbook(query)
        logger.info("教材分类: primary=%s, confidence=%.2f", 
                   textbook_info.get("primary"), textbook_info.get("confidence", 0))
        
        # 2. 智能检索（AtomicQuery + Chunk）
        smart_result = smart_retrieve(query, top_k=5)
        logger.info("智能检索: AtomicQuery=%d, Chunk=%d",
                   len(smart_result.get("atomic_queries", [])),
                   len(smart_result.get("chunks", [])))
        
    except Exception as e:
        logger.warning("智能检索失败，使用默认流程: %s", e)
        textbook_info = None
        smart_result = None

    # ============================================================
    # 动态设置 namespace（解决配置与数据标签不匹配问题）
    # ============================================================
    # 默认使用 Pharmacology（如果无法判断）
    effective_namespace = "Pharmacology"
    if textbook_info and textbook_info.get("primary"):
        effective_namespace = textbook_info["primary"]
    
    # 临时覆盖配置中的 namespace
    original_namespace = KAG_CONFIG.all_config.get("project", {}).get("namespace", "Medicine")
    if effective_namespace != original_namespace:
        logger.info("动态切换 namespace: %s -> %s", original_namespace, effective_namespace)
        KAG_CONFIG.all_config["project"]["namespace"] = effective_namespace

    # ============================================================
    # KAG Pipeline 检索 + 生成
    # ============================================================
    try:
        pipeline = SolverPipelineABC.from_config(KAG_CONFIG.all_config["kag_solver_pipeline"])
        reporter = TraceLogReporter()

        # 直接使用原始查询调用 pipeline
        answer = await _ainvoke_with_backoff(pipeline, query, reporter)
        report, status = reporter.generate_report_data()
    finally:
        # 恢复原始 namespace（避免影响后续请求）
        KAG_CONFIG.all_config["project"]["namespace"] = original_namespace

    # ============================================================
    # 调试日志：检查 report 和 answer 中的引用情况
    # ============================================================
    logger.info("="*50)
    logger.info("[调试] report.reference 数量: %d", len(report.reference or []))
    logger.info("[调试] answer 前200字符: %s", answer[:200] if answer else "空")
    logger.info("[调试] answer 中是否包含 <reference 标签: %s", "<reference" in (answer or ""))
    if report.reference:
        for i, ref in enumerate(report.reference[:3]):
            logger.info("[调试] reference[%d] 类型: %s, 内容: %s", i, type(ref).__name__, str(ref)[:100])
    logger.info("="*50)

    # 这里的 report.reference 是 RefDocSet/RefDoc（OpenSPGReporter 结构），统一转 dict
    references: List[Dict[str, Any]] = []
    
    # 方法1: 从 report.reference 提取
    for ref in (report.reference or []):
        if hasattr(ref, "to_dict"):
            references.append(ref.to_dict())
        elif isinstance(ref, dict):
            references.append(ref)

    # 方法2: 如果 references 为空，从 retriever_result 中提取 chunks
    if not references:
        stream_data = getattr(reporter, "report_stream_data", {})
        for key, value in stream_data.items():
            if "_kag_retriever_result" in key or "retriever_result" in key:
                if isinstance(value, dict):
                    retriever_output = value.get("content")
                    if retriever_output and hasattr(retriever_output, "chunks"):
                        chunks = retriever_output.chunks or []
                        for chunk in chunks[:5]:  # 取最相关的前5个
                            if hasattr(chunk, "to_dict"):
                                chunk_dict = chunk.to_dict()
                            elif isinstance(chunk, dict):
                                chunk_dict = chunk
                            else:
                                chunk_dict = {"content": str(chunk)}
                            
                            # 调试：打印 chunk 的所有键
                            logger.info("chunk_dict 键: %s", list(chunk_dict.keys()))
                            
                            # 尝试多种 id 字段名
                            chunk_id = (
                                chunk_dict.get("id") 
                                or chunk_dict.get("biz_id") 
                                or chunk_dict.get("chunk_id")
                                or chunk_dict.get("_id")
                                or ""
                            )
                            chunk_name = chunk_dict.get("name", "")
                            
                            logger.info("提取的 chunk_id: %s, chunk_name: %s", chunk_id[:50] if chunk_id else "空", chunk_name[:50] if chunk_name else "空")
                            
                            references.append({
                                "id": chunk_id,
                                "type": "chunk",
                                "info": [{
                                    "document_id": chunk_id,
                                    "document_name": chunk_name,
                                    "content": _clean_chunk_content(chunk_dict.get("content", str(chunk_dict)), chunk_id)
                                }]
                            })
                        if references:
                            logger.info("从 retriever_result.chunks 提取到 %d 条引用", len(references))
                            break

    namespace = KAG_CONFIG.all_config.get("project", {}).get("namespace", "")
    
    try:
        from neo4j_provenance import (
            fetch_chunk_provenance_direct, 
            fetch_merged_chunk_content,
            detect_namespace_from_chunk_labels
        )
        use_direct_neo4j = True
        logger.info("使用 Neo4j 直连溯源（支持动态命名空间检测）")
    except ImportError:
        use_direct_neo4j = False
        fetch_merged_chunk_content = None
        detect_namespace_from_chunk_labels = None
        logger.info("使用 OpenSPG API 溯源")
    
    if not use_direct_neo4j:
        graph_api: GraphApiABC = GraphApiABC.from_config({"type": "openspg_graph_api"})

    # OpenSPGReporter 的 reference 结构一般为：[{id,type,info:[{document_id,document_name,content,...}, ...]}]
    for ref_set in references:
        info = ref_set.get("info")
        if not isinstance(info, list):
            continue
        for item in info:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("document_id")
            if not chunk_id:
                continue
            # 动态检测命名空间：从 Chunk 标签中推断教材来源
            detected_ns = None
            if detect_namespace_from_chunk_labels:
                detected_ns = detect_namespace_from_chunk_labels(str(chunk_id))
            
            # 使用检测到的命名空间，否则回退到配置中的 namespace
            effective_namespace = detected_ns if detected_ns else namespace
            logger.info("溯源使用命名空间: %s (检测: %s, 配置: %s)", 
                       effective_namespace, detected_ns, namespace)
            
            # 优先使用 Neo4j 直连
            if use_direct_neo4j:
                prov = fetch_chunk_provenance_direct(namespace=effective_namespace, chunk_biz_id=str(chunk_id))
                
                # 尝试获取语义合并的 Chunk 内容（解决断句+相关性问题）
                if fetch_merged_chunk_content:
                    # 获取向量化模型
                    try:
                        vectorize_config = KAG_CONFIG.all_config.get("vectorize_model")
                        vec_model = VectorizeModelABC.from_config(vectorize_config) if vectorize_config else None
                    except:
                        vec_model = None
                    
                    merged_content = fetch_merged_chunk_content(
                        namespace=effective_namespace, 
                        chunk_biz_id=str(chunk_id),
                        query=query,  # 传递原始查询
                        vectorize_model=vec_model
                    )
                    if merged_content:
                        # 清理并替换原始内容
                        item["content"] = _clean_chunk_content(merged_content, chunk_id)
                        logger.info("已用语义合并内容替换原始 Chunk (长度: %d)", len(merged_content))
            else:
                prov = fetch_chunk_provenance(graph_api, namespace=namespace, chunk_biz_id=str(chunk_id))
            
            logger.info("溯源结果 chunk_id=%s: %s", chunk_id, prov)
            if prov:
                # 如果 Neo4j 查询到了 textbook，直接使用；否则 fallback 到 document_name
                if not prov.get("textbook"):
                    doc_name = item.get("document_name")
                    if isinstance(doc_name, str) and doc_name.strip():
                        prov["textbook"] = doc_name.strip()
                item["provenance"] = prov
            else:
                # 溯源失败时，检查 Neo4j 数据是否存在
                logger.warning("溯源失败 chunk_id=%s，请检查 Neo4j 中是否建立了 Textbook->Chapter->Section->KnowledgeUnit->Chunk 关系链", chunk_id)

    # 相似度过滤：只保留与 AI 回答高度相关的 chunk
    try:
        vectorize_config = KAG_CONFIG.all_config.get("vectorize_model")
        if vectorize_config and references:
            vectorize_model = VectorizeModelABC.from_config(vectorize_config)
            references = filter_relevant_chunks(
                query=query,  # 备用
                references=references,
                vectorize_model=vectorize_model,
                answer=answer,  # 使用 AI 回答计算相似度
                min_similarity=0.5
            )
    except Exception as e:
        logger.warning("相似度过滤失败: %s", e)

    # 无命中兜底：只记录日志
    if not references:
        logger.warning("未检索到相关知识库内容，答案可能基于模型通用知识")

    result = {
        "query": query,
        "answer": answer,
        "status": status,
        "references": references,
    }
    
    # 添加智能检索信息（如果可用）
    if textbook_info:
        result["textbook_classification"] = textbook_info
    if smart_result:
        result["smart_retrieval"] = {
            "atomic_queries_count": len(smart_result.get("atomic_queries", [])),
            "chunks_count": len(smart_result.get("chunks", [])),
            "effective_namespace": smart_result.get("effective_namespace")
        }
    
    return result


async def qa_with_provenance_stream(query: str, config_path: str):
    """
    流式问答函数（async generator）
    
    流程:
    1. 初始化环境和知识检索
    2. 根据问题向量相似度筛选相关 chunks
    3. 获取溯源信息
    4. LLM 流式生成回答
    5. 返回溯源引用
    
    Yields:
        dict: 包含 type 和相应数据的字典
              - {"type": "thinking", "message": "..."}
              - {"type": "delta", "delta": "..."}
              - {"type": "references", "references": [...]}
              - {"type": "done"}
    """
    from kag.common.conf import init_env, KAG_CONFIG
    from kag.interface import SolverPipelineABC, VectorizeModelABC
    from kag.solver.reporter.trace_log_reporter import TraceLogReporter
    
    init_env(config_file=config_path)
    
    # ============================================================
    # 教材分类 + 动态 namespace 切换（必须在 Pipeline 构建前完成）
    # ============================================================
    try:
        from neo4j_provenance import classify_textbook
        textbook_info = classify_textbook(query)
        effective_namespace = textbook_info.get("primary") or "Pharmacology"
        logger.info("流式问答教材分类: %s -> namespace: %s", textbook_info, effective_namespace)
        
        # 检测非医学问题，直接返回友好提示
        if effective_namespace == "NON_MEDICAL":
            logger.info("检测到非医学问题，返回友好提示: %s", query[:50])
            yield {"type": "thinking", "message": "正在分析问题类型..."}
            
            # 使用正确的流式输出格式
            friendly_message = """抱歉，我是医学学习助手，专注于**病理学**、**药理学**和**生理学**等医学领域的知识问答。

您的问题似乎不属于医学相关领域，我无法为您提供准确的解答。

如果您有关于医学基础知识、疾病机制、药物作用等方面的问题，欢迎随时向我提问！"""
            
            # 模拟流式输出
            for i in range(0, len(friendly_message), 2):
                yield {"type": "delta", "delta": friendly_message[i:i+2]}
                await asyncio.sleep(0.02)
            
            yield {"type": "sources", "sources": []}
            yield {"type": "done"}
            return
            
    except Exception as e:
        logger.warning("教材分类失败: %s", e)
        effective_namespace = "Pharmacology"
        textbook_info = None
    
    # 注意：不动态修改 namespace！
    # 向量索引是针对 Medicine.Chunk 的，所以必须使用 Medicine namespace 进行检索
    # 教材分类结果将在溯源时通过原始标签（Pharmacology/Pathology）来判断
    original_namespace = KAG_CONFIG.all_config.get("project", {}).get("namespace", "Medicine")
    logger.info("使用配置 namespace: %s (教材分类: %s)", original_namespace, effective_namespace)
    
    # 设置当前教材分类结果，供 VectorChunkRetriever 使用（优化搜索效率）
    # 支持多教材：primary + secondary
    try:
        from kag_compat_patch import set_current_textbooks
        target_textbooks = [effective_namespace]
        # 如果有次要教材且置信度不是 100%，也添加到搜索列表
        if textbook_info and textbook_info.get("confidence", 1.0) < 1.0:
            secondary = textbook_info.get("secondary", [])
            if secondary:
                target_textbooks.extend(secondary)
        set_current_textbooks(target_textbooks)
    except Exception as e:
        logger.debug("设置当前教材失败: %s", e)
    
    # Step 1: 发送 thinking 状态
    yield {"type": "thinking", "message": "正在从知识库检索..."}
    
    # Step 2: 执行知识检索（使用原始 Medicine namespace）
    try:
        logger.info("构建 Pipeline，使用 namespace: %s", original_namespace)
        pipeline = SolverPipelineABC.from_config(KAG_CONFIG.all_config["kag_solver_pipeline"])
        reporter = TraceLogReporter()
        
        answer = await _ainvoke_with_backoff(pipeline, query, reporter)
    except Exception as e:
        yield {"type": "error", "error": str(e)}
        return
    
    report, status = reporter.generate_report_data()
    
    # ============================================================
    # 调试日志：检查 report 和 answer 中的引用情况
    # ============================================================
    logger.info("="*50)
    logger.info("[调试-流式] report.reference 数量: %d", len(report.reference or []))
    logger.info("[调试-流式] answer 前200字符: %s", answer[:200] if answer else "空")
    logger.info("[调试-流式] answer 中是否包含 <reference 标签: %s", "<reference" in (answer or ""))
    if report.reference:
        for i, ref in enumerate(report.reference[:3]):
            logger.info("[调试-流式] reference[%d] 类型: %s, 内容: %s", i, type(ref).__name__, str(ref)[:100])
    logger.info("="*50)
    
    # Step 3: 初始化引用列表（稍后在重编号阶段按需填充）
    yield {"type": "thinking", "message": "正在分析知识来源..."}
    
    references: List[Dict[str, Any]] = []
    
    # 注意：移除了冗余的"简化溯源"逻辑（之前做了8次Neo4j查询并去重排序，但顺序改变导致匹配困难）
    # 新逻辑：只在重编号阶段对 LLM 实际引用的条目做按需溯源，效率更高
    logger.info("[溯源优化] 跳过预溯源，将在重编号阶段按需查询")

    
    # ============================================================
    # 核心溯源逻辑：解析 LLM 回答中的 [1]、[2] 等引用标记，按需溯源
    # ============================================================
    if answer:
        try:
            import re
            from kag_compat_patch import get_last_refer_data, get_last_rerank_chunks
            from neo4j_provenance import fetch_chunk_provenance_direct, detect_namespace_from_chunk_labels
            
            # 从回答中提取所有引用编号，按出现顺序排列（去重）
            ref_pattern = r'\[(\d+)\]'
            raw_mentions = re.findall(ref_pattern, answer)
            
            # 建立有序映射：原始编号 -> 新连续编号(1, 2, 3...)
            ordered_mentions = []
            seen = set()
            for m in raw_mentions:
                m_int = int(m)
                if m_int not in seen:
                    ordered_mentions.append(m_int)
                    seen.add(m_int)
            
            if ordered_mentions:
                # 核心映射逻辑
                old_to_new = {old: i + 1 for i, old in enumerate(ordered_mentions)}
                logger.info("[引用对齐] 建立映射表: %s", old_to_new)
                
                # 1. 替换回答中的原始编号为新编号
                def replace_func(match):
                    old_num = int(match.group(1))
                    return f"[{old_to_new.get(old_num, old_num)}]"
                
                answer = re.sub(ref_pattern, replace_func, answer)
                
                # 2. 获取数据源
                refer_data = get_last_refer_data()
                rerank_chunks = get_last_rerank_chunks()
                
                logger.info("[按需溯源] refer_data=%d条, rerank_chunks=%d条", 
                           len(refer_data), len(rerank_chunks))
                
                # 3. 只对 LLM 实际引用的条目做按需溯源
                filtered_refs = []
                for old_num, new_num in old_to_new.items():
                    idx = old_num - 1
                    
                    # 从 refer_data 获取基本信息
                    content = ""
                    refer_content_preview = ""
                    if 0 <= idx < len(refer_data):
                        ref = refer_data[idx]
                        content = ref.get('content', '')[:500]
                        refer_content_preview = content[:50] if content else "(空)"
                    
                    # 从 rerank_chunks 获取完整 chunk_id 并实时溯源
                    prov = None
                    full_chunk_id = ""
                    if 0 <= idx < len(rerank_chunks):
                        chunk = rerank_chunks[idx]
                        if hasattr(chunk, "to_dict"):
                            chunk_dict = chunk.to_dict()
                        elif isinstance(chunk, dict):
                            chunk_dict = chunk
                        else:
                            chunk_dict = {}
                        
                        full_chunk_id = chunk_dict.get("chunk_id") or chunk_dict.get("id") or ""
                        chunk_content_preview = chunk_dict.get("content", "")[:50] if chunk_dict.get("content") else "(空)"
                        
                        if full_chunk_id:
                            detected_ns = detect_namespace_from_chunk_labels(str(full_chunk_id)) or "Medicine"
                            prov = fetch_chunk_provenance_direct(namespace=detected_ns, chunk_biz_id=str(full_chunk_id))
                            # 验证日志：对比 refer_data 和 rerank_chunks 的内容
                            logger.info("[按需溯源] idx=%d old_num=%d new_num=%d", idx, old_num, new_num)
                            logger.info("[按需溯源]   refer_data内容: %s...", refer_content_preview)
                            logger.info("[按需溯源]   chunk内容: %s...", chunk_content_preview)
                            logger.info("[按需溯源]   prov=%s", prov)
                    
                    filtered_refs.append({
                        "id": full_chunk_id or f"ref_{new_num}",
                        "type": "chunk",
                        "ref_number": new_num,
                        "info": [{
                            "document_id": full_chunk_id,
                            "document_name": "",
                            "content": content,
                            "detected_namespace": "Medicine",
                            "provenance": prov
                        }]
                    })
                
                # 按新编号排序
                filtered_refs.sort(key=lambda x: x.get("ref_number", 0))
                
                logger.info("[按需溯源] 完成，共 %d 个引用", len(filtered_refs))
                references = filtered_refs
            else:
                logger.info("[引用解析] 回答中没有检测到引用标记")
        except Exception as e:
            logger.warning("[按需溯源] 失败: %s", e)
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # LLM 验证步骤已移除
    # 原因：LLM 现在通过 [1], [2] 等编号明确标注引用来源，
    #       我们直接解析这些编号来确定引用，无需再验证
    # ============================================================
    
    logger.info("[溯源完成] 最终返回 %d 个引用", len(references))

    # Step 4: 流式发送回答
    yield {"type": "thinking", "message": "正在生成回答..."}
    
    if answer:
        import re
        answer = re.sub(r'<reference\s+id="[^"]*">\s*</reference>', '', answer)
        
        chunk_size = 2
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i+chunk_size]
            yield {"type": "delta", "delta": chunk}
            await asyncio.sleep(0.02)
    
    # Step 5: 发送溯源引用
    if references:
        yield {"type": "references", "references": references}
    
    yield {"type": "done"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../kag_config.yaml",
        help="配置文件路径（默认：../kag_config.yaml）",
    )
    parser.add_argument("query", nargs="?", default="药理学的基本概念是什么？")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("kag.interface.common.llm_client").setLevel(logging.ERROR)

    try:
        result = asyncio.run(qa_with_provenance(args.query, args.config))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        # 避免把超长 prompt/文档刷到终端
        msg = str(e)
        if "RATE_LIMIT_EXCEEDED" in msg or "Error code: 429" in msg:
            print("运行失败：触发 LLM 限流（429）。建议稍等 30-120 秒后重试，或降低本地 max_rate。")
        else:
            print(f"运行失败：{e}")


if __name__ == "__main__":
    main()
