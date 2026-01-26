# ============================================================
# KAG 兼容性补丁 - 让 RetrieverOutput 兼容 LLMGenerator
# 
# 问题：LLMGenerator 检查 isinstance(task.result, KAGRetrievedResponse)
#       但 RetrieverOutput 不是 KAGRetrievedResponse 的实例
#       且 RetrieverOutput 只有 chunks 属性，没有 chunk_datas
# 
# 解决：直接修改 LLMGenerator.invoke 方法，让它同时支持两种类型
# 
# 使用：在导入任何 KAG 模块前 import 此模块
# ============================================================

# 全局变量：存储最后一次传给 LLM 的 chunks，供溯源使用
_last_rerank_chunks = []

# 全局变量：存储当前问题的教材分类结果，供 VectorChunkRetriever 使用
# 格式：列表，如 ["Pharmacology"] 或 ["Pharmacology", "Pathology"]
_current_textbooks = []

# 全局变量：存储 chunk_id 到引用编号的映射，供溯源解析 LLM 引用使用
# 格式：{"chunk_id_1": 1, "chunk_id_2": 2, ...}
_chunk_to_ref_number = {}

# 全局变量：存储最后一次传给 LLM 的 refer_data（带编号的参考资料列表）
# 用于溯源时根据 LLM 引用编号查找对应的内容
_last_refer_data = []

def get_last_rerank_chunks():
    """获取最后一次传给 LLM 的 chunks（供溯源模块使用）"""
    return _last_rerank_chunks

def get_chunk_to_ref_number():
    """获取 chunk_id 到引用编号的映射（供溯源解析 LLM 引用使用）"""
    return _chunk_to_ref_number

def get_last_refer_data():
    """获取最后一次传给 LLM 的 refer_data（供溯源根据编号查找引用）"""
    return _last_refer_data

def get_current_textbooks():
    """获取当前问题的教材分类结果列表（供 VectorChunkRetriever 使用）"""
    return _current_textbooks

def set_current_textbooks(textbooks):
    """
    设置当前问题的教材分类结果
    Args:
        textbooks: 单个教材字符串或教材列表，如 "Pharmacology" 或 ["Pharmacology", "Pathology"]
    """
    global _current_textbooks
    if isinstance(textbooks, str):
        _current_textbooks = [textbooks] if textbooks else []
    else:
        _current_textbooks = list(textbooks) if textbooks else []
    print(f"[kag_compat_patch] 设置当前教材: {_current_textbooks}")

def patch_llm_generator():
    """
    修改 LLMGenerator.invoke 方法，让它支持 RetrieverOutput
    """
    try:
        from kag.solver.generator.llm_generator import LLMGenerator
        from kag.solver.executor.retriever.local_knowledge_base.kag_retriever.kag_hybrid_executor import (
            KAGRetrievedResponse,
            to_reference_list,
        )
        from kag.interface.solver.retriever_abc import RetrieverOutput
        from kag.interface.solver.reporter_abc import ReporterABC
        from typing import Optional
        import json
        
        # 保存原始方法的辅助函数
        def to_task_context_str(context):
            if not context or "task" not in context:
                return ""
            return f"""{context['name']}:{context['task']}
thought: {context['result']}.{context.get('thought', '')}"""
        
        def extra_reference(references):
            return [
                {
                    "content": reference["content"],
                    "document_name": reference["document_name"],
                    "id": reference["id"],
                }
                for reference in references
            ]
        
        # 保存原始的 invoke 方法
        _original_invoke = LLMGenerator.invoke
        
        def patched_invoke(self, query, context, **kwargs):
            """修补版本的 invoke，支持 RetrieverOutput"""
            reporter: Optional[ReporterABC] = kwargs.get("reporter", None)
            results = []
            rerank_queries = []
            chunks = []
            graph_data = context.variables_graph
            tasks = []
            
            for task in context.gen_task(False):
                tasks.append(task)
                # 同时支持 KAGRetrievedResponse 和 RetrieverOutput
                if self.chunk_reranker:
                    task_chunks = None
                    if isinstance(task.result, KAGRetrievedResponse):
                        task_chunks = task.result.chunk_datas
                    elif isinstance(task.result, RetrieverOutput) and task.result.chunks:
                        task_chunks = task.result.chunks
                    elif hasattr(task.result, 'chunks') and task.result.chunks:
                        task_chunks = task.result.chunks
                    
                    if task_chunks:
                        rerank_queries.append(
                            task.arguments.get("rewrite_query", task.arguments["query"])
                        )
                        chunks.append(task_chunks)
                
                results.append(to_task_context_str(task.get_task_context()))
            
            rerank_chunks = self.chunk_reranker.invoke(query, rerank_queries, chunks)
            
            # ============================================================
            # 教材过滤：根据问题分类只保留目标教材的 chunk（支持多教材）
            # ============================================================
            try:
                from medical_generator_prompt import classify_textbook
                textbook_info = classify_textbook(query)
                
                # 收集所有允许的教材（primary + secondary）
                allowed_namespaces = set()
                if textbook_info:
                    primary = textbook_info.get("primary", "")
                    if primary:
                        allowed_namespaces.add(primary.lower())
                    secondary = textbook_info.get("secondary", [])
                    for s in secondary:
                        if s:
                            allowed_namespaces.add(s.lower())
                
                print(f"[kag_compat_patch] 教材分类: {allowed_namespaces}, chunks数量: {len(rerank_chunks)}")
                
                if allowed_namespaces and rerank_chunks:
                    # 获取 Neo4j driver 用于查询标签
                    neo4j_driver = None
                    try:
                        from neo4j_provenance import get_neo4j_driver, NEO4J_DEFAULT_DATABASE
                        neo4j_driver = get_neo4j_driver()
                    except:
                        pass
                    
                    def get_chunk_namespace(chunk) -> str:
                        """获取 chunk 所属的教材命名空间"""
                        chunk_id = None
                        # ChunkData 的字段是 chunk_id，不是 biz_id 或 id
                        if hasattr(chunk, "chunk_id"):
                            chunk_id = chunk.chunk_id
                        elif hasattr(chunk, "biz_id"):
                            chunk_id = chunk.biz_id
                        elif hasattr(chunk, "id"):
                            chunk_id = chunk.id
                        elif isinstance(chunk, dict):
                            chunk_id = chunk.get("chunk_id") or chunk.get("biz_id") or chunk.get("id")
                        
                        if not chunk_id or not neo4j_driver:
                            print(f"[kag_compat_patch] chunk_id={chunk_id}, driver={neo4j_driver is not None}")
                            return ""
                        
                        try:
                            with neo4j_driver.session(database=NEO4J_DEFAULT_DATABASE) as session:
                                result = session.run(
                                    "MATCH (c) WHERE c.id = $id RETURN labels(c) as labels LIMIT 1",
                                    id=str(chunk_id)
                                )
                                record = result.single()
                                if record and record["labels"]:
                                    for label in record["labels"]:
                                        if ".Chunk" in label and label != "Medicine.Chunk":
                                            ns = label.split(".")[0].lower()
                                            print(f"[kag_compat_patch] chunk={chunk_id[:20]}... -> ns={ns}")
                                            return ns
                        except Exception as e:
                            print(f"[kag_compat_patch] 查询标签失败: {e}")
                        return ""
                    
                    filtered_chunks = []
                    for chunk in rerank_chunks:
                        chunk_ns = get_chunk_namespace(chunk)
                        # 保留匹配任一允许教材的 chunk
                        if chunk_ns in allowed_namespaces:
                            filtered_chunks.append(chunk)
                        elif not chunk_ns:
                            # 无法判断时也保留
                            filtered_chunks.append(chunk)
                        else:
                            print(f"[kag_compat_patch] 移除 chunk: {chunk_ns} not in {allowed_namespaces}")
                    
                    if filtered_chunks:
                        print(f"[kag_compat_patch] 过滤后: {len(filtered_chunks)}/{len(rerank_chunks)}")
                        rerank_chunks = filtered_chunks
                    else:
                        # 过滤后为空，使用教材专用向量索引补充
                        print(f"[kag_compat_patch] 过滤后为空, 使用教材专用索引补充...")
                        rerank_chunks = []  # 先清空原来的 pharmacology chunks
                        try:
                            import requests
                            
                            # SiliconFlow 向量化配置（与 neo4j_provenance.py 一致）
                            VECTORIZE_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
                            VECTORIZE_BASE_URL = "https://api.siliconflow.cn/v1"
                            VECTORIZE_MODEL = "Qwen/Qwen3-Embedding-8B"
                            
                            # 获取问题向量
                            headers = {
                                "Authorization": f"Bearer {VECTORIZE_API_KEY}",
                                "Content-Type": "application/json"
                            }
                            payload = {"model": VECTORIZE_MODEL, "input": query[:500]}
                            resp = requests.post(f"{VECTORIZE_BASE_URL}/embeddings", headers=headers, json=payload, timeout=30)
                            resp.raise_for_status()
                            query_vector = resp.json()["data"][0]["embedding"]
                            
                            # 使用教材专用索引检索
                            index_name = f"_{target_namespace}_chunk_content_vector_index"
                            print(f"[kag_compat_patch] 使用索引: {index_name}")
                            
                            with neo4j_driver.session(database=NEO4J_DEFAULT_DATABASE) as session:
                                result = session.run(f"""
                                    CALL db.index.vector.queryNodes('{index_name}', 8, $vector)
                                    YIELD node, score
                                    RETURN node.id AS chunk_id, node.content AS content, 
                                           node.name AS name, score
                                    ORDER BY score DESC
                                """, vector=query_vector)
                                
                                # 构建 ChunkData 对象
                                from kag.interface.solver.retriever_abc import ChunkData
                                for record in result:
                                    chunk = ChunkData(
                                        chunk_id=record["chunk_id"],
                                        content=record["content"] or "",
                                        title=record["name"] or "",
                                        score=record["score"]
                                    )
                                    rerank_chunks.append(chunk)
                                    print(f"[kag_compat_patch] 补充 chunk: {record['chunk_id'][:20]}... score={record['score']:.3f}")
                            
                            print(f"[kag_compat_patch] 补充完成: {len(rerank_chunks)} 条")
                        except Exception as e2:
                            print(f"[kag_compat_patch] 补充失败: {e2}")
                            import traceback
                            traceback.print_exc()
            except Exception as e:
                print(f"[kag_compat_patch] 过滤失败: {e}")
                import traceback
                traceback.print_exc()
            # ============================================================
            
            # ============================================================
            # 关键词加权排序：优先选择包含问题关键词的 chunks
            # ============================================================
            try:
                import jieba
                # 提取问题中的关键词（2字以上的词）
                query_keywords = [w for w in jieba.cut(query) if len(w) >= 2]
                print(f"[kag_compat_patch] 问题关键词: {query_keywords}")
                
                def count_keyword_matches(chunk) -> int:
                    """计算 chunk 中包含多少个问题关键词"""
                    content = ""
                    if hasattr(chunk, "content"):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict):
                        content = chunk.get("content", "")
                    
                    count = 0
                    for kw in query_keywords:
                        if kw in content:
                            count += 1
                    return count
                
                # 按关键词匹配数排序（多的在前）
                rerank_chunks_with_score = [(chunk, count_keyword_matches(chunk)) for chunk in rerank_chunks]
                rerank_chunks_with_score.sort(key=lambda x: x[1], reverse=True)
                # 仅保留前 8 条，减少干扰
                rerank_chunks = [item[0] for item in rerank_chunks_with_score][:8]
                
                # 打印排序后的结果
                print(f"[kag_compat_patch] 关键词加权排序并截断(max 8)后:")
                for i, (chunk, score) in enumerate(rerank_chunks_with_score[:5]):
                    content = ""
                    if hasattr(chunk, "content"):
                        content = chunk.content[:50] if chunk.content else ""
                    elif isinstance(chunk, dict):
                        content = chunk.get("content", "")[:50]
                    print(f"  [{i}] score={score}: {content}...")
            except Exception as e:
                print(f"[kag_compat_patch] 关键词排序失败: {e}")
            
            # 生成参考资料列表（必须在排序后！）
            refer_retrieved_data = to_reference_list(
                prefix_id=0, retrieved_datas=rerank_chunks
            )
            
            # ============================================================
            # 显示传给 LLM 的 chunk 内容（验证用）
            # ============================================================
            print(f"[kag_compat_patch] === 传给 LLM 的 {len(rerank_chunks)} 个 Chunk ===")
            for i, chunk in enumerate(rerank_chunks[:5]):  # 只显示前5个
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content[:100] if chunk.content else ""
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")[:100]
                print(f"[kag_compat_patch] [{i}] {content}...")
            print(f"[kag_compat_patch] ========================================")
            
            # 保存到全局变量，供溯源模块使用
            global _last_rerank_chunks
            _last_rerank_chunks = rerank_chunks
            
            content_json = {"step": results}
            if reporter:
                reporter.add_report_line(
                    "generator", "final_generator_input", content_json, "FINISH"
                )
                reporter.add_report_line(
                    "generator_reference", "reference_chunk", rerank_chunks, "FINISH"
                )
                reporter.add_report_line(
                    "generator_reference_all",
                    "reference_ref_format",
                    refer_retrieved_data,
                    "FINISH",
                )
                reporter.add_report_line(
                    "generator_reference_graphs", "reference_graph", graph_data, "FINISH"
                )
            # ============================================================
            # 内容净化、预溯源与系统编号生成
            # ============================================================
            refer_data = extra_reference(refer_retrieved_data)
            
            import re
            
            # 内联轻量级溯源函数（避免导入 neo4j_provenance 触发 KAG 初始化报错）
            def _fetch_provenance_inline(chunk_id: str) -> dict:
                """内联版本的溯源查询，不依赖外部模块"""
                try:
                    from neo4j import GraphDatabase
                    uri = "bolt://localhost:7687"
                    auth = ("neo4j", "neo4j@openspg")  # 正确的密码
                    driver = GraphDatabase.driver(uri, auth=auth)
                    
                    # 直接查询 Medicine.Chunk 节点的层级信息
                    query = """
                    MATCH (c:`Medicine.Chunk`)
                    WHERE c.id = $chunk_id
                    WITH c, 
                         c.section_title AS direct_section_title,
                         c.chapter_title AS direct_chapter_title,
                         replace(replace(c.section_id, '"', ''), "'", '') AS clean_section_id
                    OPTIONAL MATCH (s:`Medicine.Section`)
                    WHERE s.id = clean_section_id
                    OPTIONAL MATCH (ch:`Medicine.Chapter`)-[:hasSections]->(s)
                    OPTIONAL MATCH (b:`Medicine.Textbook`)-[:hasChapters]->(ch)
                    RETURN 
                        coalesce(b.name, '') AS textbook,
                        coalesce(ch.name, direct_chapter_title, '') AS chapter,
                        coalesce(s.name, direct_section_title, '') AS section
                    LIMIT 1
                    """
                    with driver.session(database="medicine") as session:  # 正确的数据库名
                        result = session.run(query, chunk_id=chunk_id)
                        record = result.single()
                        if record:
                            out = {}
                            textbook = (record["textbook"] or "").strip().strip('"').strip("'")
                            chapter = (record["chapter"] or "").strip().strip('"').strip("'")
                            section = (record["section"] or "").strip().strip('"').strip("'")
                            if textbook: out["textbook"] = textbook
                            if chapter: out["chapter"] = chapter
                            if section: out["section"] = section
                            if out:
                                return out
                    driver.close()
                except Exception as e:
                    print(f"[kag_compat_patch] 内联溯源失败: {e}")
                return None

            global _chunk_to_ref_number
            _chunk_to_ref_number = {}
            refer_data_with_numbers = []
            
            print(f"[kag_compat_patch] 开始对 {len(refer_data)} 条资料进行预溯源...")
            
            for i, ref in enumerate(refer_data):
                ref_number = i + 1
                doc_id = ref.get('id', '') or ref.get('document_id', '')
                if doc_id:
                    _chunk_to_ref_number[doc_id] = ref_number
                
                # 1. 深度预溯源：获取教材-章节-小节
                source_label = "Medicine"
                if doc_id:
                    prov = _fetch_provenance_inline(str(doc_id))
                    if prov:
                        # 注入元数据供后续使用（重要！）
                        ref['provenance'] = prov
                        # 构建精准来源标签
                        parts = []
                        if prov.get('textbook'): parts.append(prov['textbook'])
                        if prov.get('chapter'): parts.append(prov['chapter'])
                        if prov.get('section'): parts.append(prov['section'])
                        if parts:
                            source_label = " - ".join(parts)
                            # 双重保险：同时更新 document_name
                            ref['document_name'] = source_label
                            print(f"[kag_compat_patch] 成功获取来源 [{doc_id[:30]}...]: {source_label}")

                # 2. 核心净化：直接修改原始对象的 content
                raw_content = ref.get('content', '')
                sanitized_content = re.sub(r'^\s*([0-9]+\s*[\.\\、\:]|[\(\（][0-9]+\s*[\)\）]|[①-⑳]|[一二三四五六七八九十]+\s*[\、\：\:]|[A-Za-z]\s*[\.\\、\:]|[\(\（][A-Za-z]\s*[\)\）])\s*', '', raw_content)
                ref['content'] = sanitized_content
                
                # 3. 提取核心概念（取前30个字符作为预览）
                # 尝试提取更有意义的标题
                concept_preview = sanitized_content[:60].split('。')[0] if sanitized_content else "资料"
                if len(concept_preview) > 50:
                    concept_preview = concept_preview[:50] + "..."
                
                # 4. 构建参考资料字符串传给 AI（更醒目的格式）
                refer_data_with_numbers.append(
                    f"===== 参考资料 [{ref_number}] =====\n来源: {source_label}\n主题: {concept_preview}\n内容:\n{sanitized_content}"
                )

            # 保存已净化的数据
            global _last_refer_data
            _last_refer_data = refer_data
            print(f"[kag_compat_patch] 保存已净化的 refer_data: {len(refer_data)} 条")
            
            if len(refer_data) and (not self.enable_ref):
                content_json["reference"] = refer_data

            content = json.dumps(content_json, ensure_ascii=False, indent=2)
            if not self.enable_ref:
                refer_data_str = "\n\n".join(refer_data_with_numbers)
                thoughts = "\n\n".join(results)
                content = f"""
Docs:
{refer_data_str}

Step by Step Analysis:
{thoughts}

            """
            return self.generate_answer(
                query=query, content=content, refer_data=refer_data, **kwargs
            )
        
        # 替换方法
        LLMGenerator.invoke = patched_invoke
        print("[kag_compat_patch] LLMGenerator.invoke 已修补，支持 RetrieverOutput")
        return True
    except Exception as e:
        print(f"[kag_compat_patch] 补丁失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 模块导入时自动执行补丁
_patch_success = patch_llm_generator()


# ============================================================
# 调试补丁：追踪 OpenSPGSearchAPI.search_vector 调用
# ============================================================
def patch_openspg_search_api():
    """给 OpenSPGSearchAPI.search_vector 添加调试日志"""
    try:
        from kag.common.tools.search_api.impl.openspg_search_api import OpenSPGSearchAPI
        
        original_search_vector = OpenSPGSearchAPI.search_vector
        
        def patched_search_vector(self, label, property_key, query_vector, topk=10, ef_search=None, params=None):
            print(f"[OpenSPGSearchAPI] search_vector: label={label}, property={property_key}, topk={topk}")
            result = original_search_vector(self, label, property_key, query_vector, topk, ef_search, params)
            print(f"[OpenSPGSearchAPI] 返回 {len(result) if result else 0} 个结果")
            if result and len(result) > 0:
                for i, r in enumerate(result[:2]):
                    score = r.get("score", 0)
                    name = r.get("node", {}).get("name", "")[:30]
                    print(f"[OpenSPGSearchAPI]   [{i}] score={score:.3f}, name={name}")
            return result
        
        OpenSPGSearchAPI.search_vector = patched_search_vector
        print("[kag_compat_patch] OpenSPGSearchAPI.search_vector 调试补丁已安装")
        return True
    except Exception as e:
        print(f"[kag_compat_patch] OpenSPGSearchAPI 补丁失败: {e}")
        return False

# 安装调试补丁
_search_patch_success = patch_openspg_search_api()


# ============================================================
# 调试补丁：追踪 VectorChunkRetriever.invoke 调用
# ============================================================
def patch_vector_chunk_retriever():
    """给 VectorChunkRetriever.invoke 添加调试日志"""
    try:
        from kag.common.tools.algorithm_tool.chunk_retriever.vector_chunk_retriever import VectorChunkRetriever
        
        original_invoke = VectorChunkRetriever.invoke
        
        def patched_invoke(self, task, **kwargs):
            query = task.arguments.get("query", "N/A") if hasattr(task, "arguments") else str(task)
            print(f"[VectorChunkRetriever] invoke: query={query[:30]}, score_threshold={self.score_threshold}")
            
            result = original_invoke(self, task, **kwargs)
            
            chunk_count = len(result.chunks) if result.chunks else 0
            print(f"[VectorChunkRetriever] 返回 {chunk_count} 个 chunks")
            if result.chunks:
                for i, chunk in enumerate(result.chunks[:3]):
                    print(f"[VectorChunkRetriever]   [{i}] score={chunk.score:.3f}, id={chunk.chunk_id[:30]}")
            if result.err_msg:
                print(f"[VectorChunkRetriever] 错误: {result.err_msg}")
            
            return result
        
        VectorChunkRetriever.invoke = patched_invoke
        print("[kag_compat_patch] VectorChunkRetriever.invoke 调试补丁已安装")
        return True
    except Exception as e:
        print(f"[kag_compat_patch] VectorChunkRetriever 补丁失败: {e}")
        return False

# 安装 VectorChunkRetriever 调试补丁
_vcr_patch_success = patch_vector_chunk_retriever()
