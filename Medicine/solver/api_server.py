"""
Medical QA API Server

简洁版本：直接调用 qa_with_provenance.py 实现问答和溯源。

启动命令：
  cd "E:/AIIA Project/MedExamEasy/DevOps/kag/kag-service/src/Pharmaco8/solver"
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ============ 必须在导入 qa_with_provenance 之前初始化 KAG 配置 ============
# 否则 kag_compat_patch.py 中的 KAG 模块导入会失败
def _get_config_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here.parent / "kag_config.yaml")

from kag.common.conf import init_env
init_env(config_file=_get_config_path())

from qa_with_provenance import qa_with_provenance, qa_with_provenance_stream


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# 禁用 uvicorn 的重复访问日志
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ============ Neo4j 配置（用于查询真实知识点数量）============
try:
    from neo4j import GraphDatabase
    NEO4J_URI = "bolt://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neo4j@openspg"
    NEO4J_DATABASE = "medicine"
    _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    HAS_NEO4J = True
    logger.info("Neo4j 连接已初始化")
except Exception as e:
    HAS_NEO4J = False
    _neo4j_driver = None
    logger.warning(f"Neo4j 连接初始化失败: {e}")

def get_knowledge_unit_count(namespace: str) -> int:
    """从 Neo4j 查询指定命名空间的 KnowledgeUnit 数量"""
    if not HAS_NEO4J or not _neo4j_driver:
        return 0
    try:
        with _neo4j_driver.session(database=NEO4J_DATABASE) as session:
            # 使用反引号转义点号
            label = f"`{namespace}.KnowledgeUnit`"
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            record = result.single()
            return record["count"] if record else 0
    except Exception as e:
        logger.warning(f"查询 KnowledgeUnit 数量失败: {e}")
        return 0


# ============ FastAPI 应用 ============

app = FastAPI(title="Medical QA Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SolveRequest(BaseModel):
    question: str
    project_id: Optional[str] = None

# ============ API 端点 ============

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "medical-qa"}


@app.post("/solve_stream")
async def solve_stream(body: SolveRequest, request: Request):
    """流式问答接口（SSE）- 真正的流式输出"""
    config_path = _get_config_path()
    
    async def event_generator():
        try:
            # 使用真正的流式函数
            async for event in qa_with_provenance_stream(body.question, config_path):
                event_type = event.get("type", "")
                
                if event_type == "thinking":
                    # 发送 thinking 状态（可选，前端可以显示加载提示）
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                elif event_type == "delta":
                    # 流式发送答案片段
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                elif event_type == "references":
                    # 格式化引用后发送
                    refs = event.get("references", [])
                    formatted_refs = []
                    logger.info("[api_server 调试] 收到 %d 个 refs", len(refs))
                    for i, ref in enumerate(refs[:8]):
                        info = ref.get("info", [])
                        logger.info("[api_server 调试] ref[%d] keys: %s", i, list(ref.keys()))
                        if info and isinstance(info, list):
                            item = info[0]
                            provenance = item.get("provenance") or {}
                            logger.info("[api_server 调试] ref[%d] provenance: %s", i, provenance)
                            
                            # 构建层级标题：教材 - 章节 - 小节
                            parts = []
                            # 优先级：provenance.textbook > item.document_name (通常此时已被补丁改为精细标题) > "Medicine"
                            textbook = provenance.get("textbook") or item.get("document_name")
                            if textbook and textbook != "":
                                parts.append(textbook)
                            
                            if provenance.get("chapter"):
                                parts.append(provenance["chapter"])
                            if provenance.get("section"):
                                parts.append(provenance["section"])
                            
                            title = " - ".join(parts) if parts else (item.get("document_name") or "参考教材")
                            logger.info("[api_server 调试] ref[%d] 最终 title: %s", i, title)
                            
                            formatted_refs.append({
                                "title": title,
                                "content": item.get("content", ""),
                                "chapter": provenance.get("chapter", ""),
                                "section": provenance.get("section", ""),
                                "ref_number": ref.get("ref_number", 0),
                            })
                    if formatted_refs:
                        yield f"data: {json.dumps({'type': 'references', 'references': formatted_refs}, ensure_ascii=False)}\n\n"
                
                elif event_type == "error":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                elif event_type == "done":
                    yield f"data: {json.dumps({'type': 'done', 'mode': 'kag'})}\n\n"
            
        except Exception as e:
            logger.error("问答出错: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'mode': 'error'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )



@app.post("/solve")
async def solve(body: SolveRequest):
    """非流式问答接口"""
    config_path = _get_config_path()
    result = await qa_with_provenance(body.question, config_path)
    return result


# ============ 知识图谱抽取 API ============

import time
import threading
import subprocess
from typing import Dict

# 抽取任务状态存储（生产环境应使用 Redis）
extraction_tasks: Dict[str, dict] = {}

# MinIO 客户端导入（可选）
try:
    from minio_client import download_from_minio, parse_minio_path
    HAS_MINIO = True
    logger.info("MinIO 客户端模块已加载")
except ImportError:
    HAS_MINIO = False
    logger.warning("MinIO 客户端模块未找到，将使用本地文件模式")

class ExtractionRequest(BaseModel):
    textbook_id: int
    file_path: str = ""  # 本地文件路径（兼容旧模式）
    textbook_name: str = ""
    namespace: Optional[str] = None
    # MinIO 路径（新模式）
    content_path: Optional[str] = None  # MinIO 路径如 "textbooks/1/content.md"
    catalog_path: Optional[str] = None  # MinIO 路径如 "textbooks/1/catalog.md"
    task_id: Optional[str] = None  # Java 端传递的任务 ID

class ExtractionResponse(BaseModel):
    task_id: str
    status: str
    message: str

@app.post("/api/extract", response_model=ExtractionResponse)
async def start_extraction(body: ExtractionRequest):
    """启动知识图谱抽取任务（支持本地文件和 MinIO）"""
    
    # 确定命名空间
    namespace = body.namespace or body.textbook_name
    
    # 确定内容文件路径
    content_file_path = body.file_path
    catalog_file_path = None
    
    # 如果提供了 MinIO 路径，优先从 MinIO 下载
    if HAS_MINIO and body.content_path:
        logger.info(f"检测到 MinIO 路径，开始下载文件: {body.content_path}")
        here = Path(__file__).resolve().parent
        temp_dir = here.parent / "builder" / "data" / "minio_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 解析并下载内容文件
        content_object = parse_minio_path(body.content_path)
        if content_object:
            # 清理教材名称：去除序号、书名号等，只保留核心名称
            raw_name = body.textbook_name or f"{body.textbook_id}"
            # 去除开头的序号（如 "16." 或 "13. "）
            import re as regex
            clean_name = regex.sub(r'^\d+\.?\s*', '', raw_name)
            # 提取书名号内的内容（如果有）
            match = regex.search(r'《(.+?)》', clean_name)
            if match:
                clean_name = match.group(1)
            # 处理不完整的书名号（只有《没有》）
            clean_name = clean_name.replace('《', '').replace('》', '')
            # 替换路径非法字符
            clean_name = clean_name.replace('/', '_').replace('\\', '_')
            # 如果清理后为空，使用 textbook_id
            if not clean_name.strip():
                clean_name = f"{body.textbook_id}"
            
            content_local = str(temp_dir / f"《{clean_name}》_content.md")
            if download_from_minio(content_object, content_local):
                content_file_path = content_local
                logger.info(f"内容文件下载成功: {content_local}")
            else:
                return ExtractionResponse(
                    task_id="",
                    status="FAILED",
                    message=f"从 MinIO 下载内容文件失败: {body.content_path}"
                )
        
        # 解析并下载目录文件（可选）
        if body.catalog_path:
            catalog_object = parse_minio_path(body.catalog_path)
            if catalog_object:
                catalog_local = str(temp_dir / f"《{clean_name}》_catalog.md")
                if download_from_minio(catalog_object, catalog_local):
                    catalog_file_path = catalog_local
                    logger.info(f"目录文件下载成功: {catalog_local}")
    
    # 验证教材配置
    if not content_file_path:
        validation_error = _validate_textbook_config(namespace, "")
        if validation_error:
            return ExtractionResponse(
                task_id="",
                status="FAILED",
                message=validation_error
            )
    
    # 生成或使用传入的任务 ID
    task_id = body.task_id or f"extract_{body.textbook_id}_{int(time.time())}"
    
    extraction_tasks[task_id] = {
        "task_id": task_id,
        "textbook_id": body.textbook_id,
        "namespace": namespace,
        "status": "RUNNING",
        "progress": 0,
        "current_step": "初始化抽取任务...",
        "total_chapters": 0,
        "completed_chapters": 0,
        "extracted_knowledge_points": 0,
        "extracted_sub_knowledge_points": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "start_timestamp": time.time(),
        "session_start_chunk": None,
        "estimated_remaining": None,
        "completed_at": None,
        "error_message": None,
    }
    
    # 异步执行抽取
    thread = threading.Thread(
        target=_run_extraction,
        args=(task_id, content_file_path, namespace, catalog_file_path),
        daemon=True
    )
    thread.start()
    
    logger.info(f"抽取任务 {task_id} 已启动: namespace={namespace}, file={content_file_path}")
    
    return ExtractionResponse(
        task_id=task_id,
        status="RUNNING",
        message="抽取任务已启动"
    )

def _validate_textbook_config(namespace: str, file_path: str) -> str:
    """验证教材配置是否存在，返回错误信息或 None"""
    here = Path(__file__).resolve().parent
    medicine_dir = here.parent
    textbooks_yaml = medicine_dir / "textbooks.yaml"
    
    if not textbooks_yaml.exists():
        return "textbooks.yaml 配置文件不存在"
    
    # 检查是否有匹配的教材配置
    import yaml
    with open(textbooks_yaml, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)
    
    def normalize_name(name):
        if not name:
            return ""
        return name.replace('（', '(').replace('）', ')').replace(' ', '').lower()
    
    normalized_namespace = normalize_name(namespace)
    textbooks = manifest.get('textbooks', [])
    
    for tb in textbooks:
        tb_name = tb.get('name', '')
        normalized_tb_name = normalize_name(tb_name)
        
        # 精确匹配或包含匹配
        if (normalized_namespace == normalized_tb_name or
            normalized_namespace in normalized_tb_name or
            normalized_tb_name in normalized_namespace):
            content_file = tb.get('content_file')
            if content_file:
                full_path = medicine_dir / content_file.lstrip('./')
                if full_path.exists():
                    return None  # 配置有效
                else:
                    return f"教材内容文件不存在: {content_file}"
    
    # 没有找到匹配的配置
    if file_path:
        return None  # 允许使用外部文件
    
    return f"教材 '{namespace}' 未在 textbooks.yaml 中配置，无法抽取"

@app.get("/api/extract/{task_id}/progress")
async def get_extraction_progress(task_id: str):
    """查询抽取进度"""
    task = extraction_tasks.get(task_id)
    if not task:
        # 返回一个默认的"未找到"状态，而不是 404（避免 Java 端报错）
        return {
            "task_id": task_id,
            "status": "NOT_FOUND",
            "error_message": "任务不存在或已过期",
            "progress": 0
        }
    return task


@app.get("/api/extract/by-textbook/{textbook_id}/progress")
async def get_extraction_progress_by_textbook(textbook_id: int):
    """按教材 ID 查询抽取进度（供 Java 后端调用）"""
    # 遍历所有任务，找到匹配的教材 ID
    for task_id, task in extraction_tasks.items():
        if task.get("textbook_id") == textbook_id:
            return task
    
    # 未找到
    return {
        "textbook_id": textbook_id,
        "status": "NOT_FOUND",
        "error_message": "未找到该教材的抽取任务",
        "progress": 0
    }

@app.get("/api/extract/ckpt-status")
async def get_ckpt_status(namespace: str = None):
    """检测是否有 ckpt 缓存，用于判断是否可以断点续抽"""
    ckpt_dir = Path(__file__).resolve().parent.parent / "builder" / "ckpt"
    
    has_cache = False
    cache_info = {}
    namespaces_with_cache = []
    
    if ckpt_dir.exists():
        # 检查各个 namespace 子目录
        for ns_dir in ckpt_dir.iterdir():
            if ns_dir.is_dir():
                ns_name = ns_dir.name
                ns_cache = {}
                ns_has_files = False
                
                # 检查该 namespace 下的各个组件缓存
                for component_dir in ns_dir.iterdir():
                    if component_dir.is_dir():
                        files = list(component_dir.glob("*"))
                        if files:
                            ns_has_files = True
                            ns_cache[component_dir.name] = len(files)
                
                if ns_has_files:
                    has_cache = True
                    cache_info[ns_name] = ns_cache
                    namespaces_with_cache.append(ns_name)
    
    return {
        "has_cache": has_cache,
        "cache_dir": str(ckpt_dir),
        "cache_info": cache_info,
        "namespaces": namespaces_with_cache,
        "message": f"发现断点缓存: {', '.join(namespaces_with_cache)}" if has_cache else "无缓存"
    }

# 存储运行中的进程，用于暂停
extraction_processes = {}

class PauseRequest(BaseModel):
    textbook_id: int

@app.post("/api/extract/pause")
async def pause_extraction(body: PauseRequest):
    """暂停抽取任务（终止进程，缓存会保留）"""
    # 查找对应的任务
    target_task_id = None
    for task_id, task in extraction_tasks.items():
        if task.get("textbook_id") == body.textbook_id and task.get("status") == "RUNNING":
            target_task_id = task_id
            break
    
    if not target_task_id:
        return {"success": False, "error": "未找到运行中的任务"}
    
    # 终止进程
    process = extraction_processes.get(target_task_id)
    if process and process.poll() is None:  # 进程仍在运行
        process.terminate()
        logger.info(f"已终止任务 {target_task_id} 的进程")
    
    # 更新任务状态
    extraction_tasks[target_task_id]["status"] = "PAUSED"
    extraction_tasks[target_task_id]["current_step"] = "已暂停（可继续）"
    
    return {"success": True, "message": "抽取已暂停，缓存已保留"}

def _run_extraction(task_id: str, file_path: str, namespace: str, catalog_path: str = None):
    """执行抽取（在后台线程中运行）
    
    参数说明：
    - file_path: 内容文件路径（已由 start_extraction 从 MinIO 下载到本地）
    - namespace: KAG 命名空间（由 Java 后端从数据库读取或自动生成）
    - catalog_path: 目录文件路径（可选，已由 start_extraction 从 MinIO 下载到本地）
    
    文件获取流程：
    1. Java 后端从数据库读取 content_md_path 和 catalog_md_path（MinIO 路径）
    2. Java 后端调用 KAG API `/api/extract`，传入这些路径
    3. start_extraction 函数从 MinIO 下载文件到本地临时目录
    4. 本函数接收已下载的本地文件路径，直接调用 indexer.py
    """
    task = extraction_tasks[task_id]
    
    try:
        # 获取路径
        here = Path(__file__).resolve().parent
        medicine_dir = here.parent  # Medicine 目录
        builder_dir = medicine_dir / "builder"
        indexer_path = builder_dir / "indexer.py"
        
        if not indexer_path.exists():
            raise FileNotFoundError(f"indexer.py not found at {indexer_path}")
        
        # 验证文件存在
        if not file_path or not Path(file_path).exists():
            raise ValueError(f"内容文件不存在: {file_path}")
        
        logger.info(f"使用内容文件: {file_path}")
        logger.info(f"使用命名空间: {namespace}")
        if catalog_path:
            logger.info(f"使用目录文件: {catalog_path}")
        
        # 构建命令
        cmd = ["python", str(indexer_path)]
        cmd.extend(["--content", file_path])
        if catalog_path and Path(catalog_path).exists():
            cmd.extend(["--catalog", catalog_path])
        cmd.extend(["--namespace", namespace])
        
        # 添加配置文件路径
        config_path = medicine_dir / "kag_config.yaml"
        if config_path.exists():
            cmd.extend(["--config", str(config_path)])
        
        cmd.extend(["--host", "http://127.0.0.1:8887"])
        
        logger.info(f"执行抽取命令: {' '.join(cmd)}")
        task["current_step"] = "正在启动抽取进程..."
        task["progress"] = 5
        
        # 执行命令（使用 -u 禁用输出缓冲，确保实时输出）
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # 禁用 Python 输出缓冲
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(builder_dir),
            encoding='utf-8',
            errors='replace',
            env=env,
            bufsize=1  # 行缓冲
        )
        
        # 保存进程引用，用于暂停
        extraction_processes[task_id] = process
        
        chunk_count = 0
        total_chunks = 0
        knowledge_count = 0
        
        # 解析进度输出
        all_output = []
        import re
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            all_output.append(line)
            logger.info(f"[indexer] {line}")
            
            # 解析 Chunk 处理进度: "Processing Chunk 1/100: chunk_id"
            chunk_match = re.search(r'Processing Chunk\s+(\d+)/(\d+)', line)
            if chunk_match:
                chunk_count = int(chunk_match.group(1))
                total_chunks = int(chunk_match.group(2))
                task["completed_chapters"] = chunk_count  # 复用字段显示 chunk 数
                task["total_chapters"] = total_chunks
                task["current_step"] = f"处理 Chunk {chunk_count}/{total_chunks}"
                task["progress"] = min(5 + int(90 * chunk_count / max(total_chunks, 1)), 95)
                
                # 记录本次 session 的起点
                current_time = time.time()
                if task.get("session_start_time") is None:
                    task["session_start_time"] = current_time
                    task["last_chunk_time"] = current_time
                    task["real_processed_count"] = 0  # 真正处理的 chunk 数（排除跳过的）
                
                # 判断本次 chunk 是真正处理还是跳过的（跳过的 chunk 处理很快）
                time_since_last = current_time - task.get("last_chunk_time", current_time)
                task["last_chunk_time"] = current_time
                
                # 如果处理时间超过 5 秒，认为是真正处理了（实际每个 chunk 约 1 分钟）
                if time_since_last > 5:
                    task["real_processed_count"] = task.get("real_processed_count", 0) + 1
                
                real_processed = task.get("real_processed_count", 0)
                
                # 计算预计剩余时间（真正处理 4 个 chunk 后才显示）
                if real_processed >= 4:
                    elapsed = current_time - task["session_start_time"]
                    avg_per_chunk = elapsed / real_processed  # 每个真正处理的 chunk 的平均时间
                    remaining_chunks = total_chunks - chunk_count
                    remaining_seconds = int(avg_per_chunk * remaining_chunks)
                    
                    if remaining_seconds < 60:
                        task["estimated_remaining"] = f"{remaining_seconds} 秒"
                    elif remaining_seconds < 3600:
                        task["estimated_remaining"] = f"{remaining_seconds // 60} 分 {remaining_seconds % 60} 秒"
                    else:
                        hours = remaining_seconds // 3600
                        minutes = (remaining_seconds % 3600) // 60
                        task["estimated_remaining"] = f"{hours} 小时 {minutes} 分"
                else:
                    task["estimated_remaining"] = "计算中..."
                
                # 解析日志中的知识点数量（indexer.py 已去重）
                ku_match = re.search(r'\[KnowledgeUnit\].*?(\d+)', line)
                if ku_match:
                    task["extracted_knowledge_points"] = int(ku_match.group(1))
                
                # 解析日志中的子知识点数量
                subku_match = re.search(r'\[SubKnowledgeUnit\].*?(\d+)', line)
                if subku_match:
                    task["extracted_sub_knowledge_points"] = int(subku_match.group(1))
                
                continue
            
            # 解析独立的知识点日志行（不在 Processing Chunk 行内）
            ku_only_match = re.search(r'\[KnowledgeUnit\].*?(\d+)', line)
            if ku_only_match:
                task["extracted_knowledge_points"] = int(ku_only_match.group(1))
                continue
            
            # 解析独立的子知识点日志行
            subku_only_match = re.search(r'\[SubKnowledgeUnit\].*?(\d+)', line)
            if subku_only_match:
                task["extracted_sub_knowledge_points"] = int(subku_only_match.group(1))
                continue
            
            # 解析教材处理开始
            if "处理教材" in line or "Processing textbook" in line.lower():
                task["current_step"] = line[:100]
                task["progress"] = 5
                continue
            
            # 解析处理完成
            if "处理完成" in line:
                task["current_step"] = line[:100]
                continue
            
            # 检测错误
            if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
                task["current_step"] = f"错误: {line[:80]}"
        
        process.wait()
        
        if process.returncode == 0:
            task["status"] = "COMPLETED"
            task["progress"] = 100
            task["current_step"] = "抽取完成"
            task["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            logger.info(f"抽取任务 {task_id} 完成")
        elif task.get("status") == "PAUSED":
            # 暂停导致的进程终止，不是真正的失败
            logger.info(f"抽取任务 {task_id} 已被用户暂停")
        else:
            task["status"] = "FAILED"
            # 获取最后几行输出作为错误信息
            last_lines = '\n'.join(all_output[-5:]) if all_output else "无输出"
            task["error_message"] = f"退出码 {process.returncode}: {last_lines}"
            logger.error(f"抽取任务 {task_id} 失败: {task['error_message']}")
            
    except Exception as e:
        logger.error(f"抽取任务 {task_id} 异常: {e}", exc_info=True)
        task["status"] = "FAILED"
        task["error_message"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

