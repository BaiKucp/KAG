import os
import sys
import re
import json
import traceback
import copy
import argparse
import yaml
import numpy as np
from typing import List, Dict, Optional

# When running as a package module, relative import works.
# When running as a script (python indexer.py), __package__ is None and
# relative imports fail, so we fall back to importing from the same directory.
try:
    from .subku_engine import SubKUEngine
except Exception:
    _builder_dir = os.path.dirname(os.path.abspath(__file__))
    if _builder_dir not in sys.path:
        sys.path.insert(0, _builder_dir)
    from subku_engine import SubKUEngine

# Setup paths
sys.path.append(os.getcwd())

from knext.graph.client import GraphClient
from knext.schema.client import BASIC_TYPES

from kag.builder.model.chunk import Chunk
from kag.common.conf import init_env, KAG_CONFIG, KAG_PROJECT_CONF
from kag.interface import LLMClient, PromptABC, VectorizeModelABC
from kag.builder.component.extractor.knowledge_unit_extractor import KnowledgeUnitSchemaFreeExtractor
from kag.builder.model.sub_graph import SubGraph
from kag.common.utils import generate_hash_id

# ============================================================================
# 猴子补丁：修复 LaTeX 公式中的反斜杠导致 JSON 解析失败的问题
# ============================================================================
try:
    import kag.builder.prompt.default.util as kag_util
    
    _original_check_data = kag_util.check_data
    
    def _patched_check_data(data_str, *args, **kwargs):
        """
        在调用原始 check_data 之前，预处理 LaTeX 公式中的反斜杠。
        将单反斜杠（如 \\lim, \\alpha）转换为双反斜杠以符合 JSON 规范。
        """
        if isinstance(data_str, str):
            # 修复：将非转义的单反斜杠变成双反斜杠
            # (?<!\\) 确保不匹配已经是双反斜杠的情况
            # (?![\\\"]) 确保不影响 JSON 标准转义序列如 \\ 和 \"
            import re
            data_str = re.sub(r'(?<!\\)\\(?![\\"])', r'\\\\', data_str)
        return _original_check_data(data_str, *args, **kwargs)
    
    kag_util.check_data = _patched_check_data
    
    # 修复 load_NER_data：处理 LLM 返回不规范 JSON 的情况
    _original_load_NER_data = kag_util.load_NER_data
    
    def _patched_load_NER_data(respond):
        """
        增强的 NER 数据解析，处理 LLM 返回不规范 JSON 的情况。
        包括：
        - [diagnosis] 应该是 ["diagnosis"]
        - 被截断的 JSON 对象数组（如 [{"name": "xxx", ...}, {"name": 未完成）
        """
        import re
        import json
        
        if isinstance(respond, str):
            # 尝试修复没有引号的字符串列表
            # 例如: [\n    diagnosis\n] -> ["diagnosis"]
            try:
                # 先尝试原始解析
                return _original_load_NER_data(respond)
            except (ValueError, json.JSONDecodeError) as e:
                # 原始解析失败，尝试修复格式
                fixed = respond.strip()
                
                # 移除 markdown 代码块
                if "```json" in fixed:
                    fixed = fixed.split("```json")[1]
                if "```" in fixed:
                    fixed = fixed.split("```")[0]
                fixed = fixed.strip()
                
                # === 修复被截断的 JSON 对象数组 ===
                # 如果是对象数组格式但被截断，尝试提取完整的对象
                if fixed.startswith('[') and not fixed.endswith(']'):
                    # 找到所有完整的 JSON 对象 {...}
                    complete_objects = []
                    depth = 0
                    obj_start = -1
                    for i, char in enumerate(fixed):
                        if char == '{':
                            if depth == 0:
                                obj_start = i
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0 and obj_start >= 0:
                                obj_str = fixed[obj_start:i+1]
                                try:
                                    obj = json.loads(obj_str)
                                    complete_objects.append(obj)
                                except:
                                    pass  # 跳过无法解析的对象
                                obj_start = -1
                    
                    if complete_objects:
                        print(f"[indexer.py] 修复截断的 JSON: 提取了 {len(complete_objects)} 个完整对象")
                        return complete_objects
                
                # 尝试修复缺少结尾括号的情况
                if fixed.startswith('['):
                    # 尝试添加缺失的括号
                    if not fixed.endswith(']'):
                        # 移除最后一个不完整的对象
                        last_comma = fixed.rfind('},')
                        if last_comma > 0:
                            fixed = fixed[:last_comma+1] + ']'
                        else:
                            last_brace = fixed.rfind('}')
                            if last_brace > 0:
                                fixed = fixed[:last_brace+1] + ']'
                    
                    try:
                        return json.loads(fixed)
                    except:
                        pass
                
                # 尝试修复没有引号的列表项
                # 匹配 [\n    word\n] 这种格式并添加引号
                if fixed.startswith('[') and fixed.endswith(']'):
                    inner = fixed[1:-1].strip()
                    if inner and not inner.startswith('"') and not inner.startswith('{'):
                        # 按逗号或换行分割
                        items = re.split(r'[,\n]+', inner)
                        items = [item.strip() for item in items if item.strip()]
                        # 给每个项添加引号（如果不是已有引号或对象）
                        quoted_items = []
                        for item in items:
                            item = item.strip().strip('"').strip("'")
                            if item:
                                quoted_items.append(f'"{item}"')
                        if quoted_items:
                            fixed = '[' + ', '.join(quoted_items) + ']'
                            try:
                                return json.loads(fixed)
                            except:
                                pass
                
                # 如果仍然失败，返回空列表而不是抛出异常
                print(f"[indexer.py] 无法解析 NER 响应，返回空列表")
                return []
        
        return _original_load_NER_data(respond)
    
    kag_util.load_NER_data = _patched_load_NER_data
    
    # 修复 load_knowIE_data：处理 LaTeX 反斜杠导致的 JSON 解析错误
    _original_load_knowIE_data = kag_util.load_knowIE_data
    
    def _patched_load_knowIE_data(respond, lang="en"):
        """
        增强的 KnowIE 数据解析，修复 LaTeX 反斜杠导致的 JSON 解析错误。
        在调用原始函数之前，先转义单独的反斜杠。
        """
        import re
        
        if isinstance(respond, str):
            # 修复：将非转义的单反斜杠变成双反斜杠
            # 匹配 \ 后面不是 " \ / b f n r t u 的情况
            fixed = re.sub(r'\\(?!["\\\/bfnrtu])', r'\\\\', respond)
            try:
                return _original_load_knowIE_data(fixed, lang)
            except Exception:
                # 如果还是失败，尝试更激进的修复
                try:
                    # 将所有单斜杠替换为双斜杠
                    fixed2 = respond.replace('\\', '\\\\')
                    return _original_load_knowIE_data(fixed2, lang)
                except Exception:
                    # 返回空字典而不是抛出异常
                    return {}
        
        return _original_load_knowIE_data(respond, lang)
    
    kag_util.load_knowIE_data = _patched_load_knowIE_data
    
    print("[indexer.py] 已应用 LaTeX 反斜杠修复补丁 + NER/KnowIE 解析容错补丁")
except Exception as e:
    print(f"[indexer.py] 应用补丁失败: {e}")

# Input/Output type aliases for extractor methods
Input = Chunk
Output = SubGraph

# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='多教材知识图谱抽取工具')
    parser.add_argument('--manifest', type=str, default=None,
                        help='教材清单文件路径 (textbooks.yaml)')
    parser.add_argument('--books', type=str, default=None,
                        help='要处理的教材ID列表，逗号分隔 (如: pharmacology,pathology)')
    parser.add_argument('--namespace', type=str, default=None,
                        help='覆盖目标 Namespace')
    parser.add_argument('--config', type=str, default=None,
                        help='覆盖 kag_config.yaml 路径')
    parser.add_argument('--content', type=str, default=None,
                        help='直接指定教材内容文件 (单文件模式)')
    parser.add_argument('--catalog', type=str, default=None,
                        help='直接指定知识点清单文件 (单文件模式)')
    parser.add_argument('--host', type=str, default='http://127.0.0.1:8887',
                        help='OpenSPG Server 地址')
    parser.add_argument('--auto-namespace', action='store_true',
                        help='自动从文件名提取中文教材名作为 namespace (如: 《药理学》-> 药理学)')
    return parser.parse_args()


def load_textbooks_manifest(manifest_path: str) -> dict:
    """加载教材清单文件"""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"清单文件不存在: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)
    
    return manifest


def extract_textbook_name(content_file: str) -> str:
    """从内容文件路径中提取教材名称。
    
    尝试从文件名中提取《...》或其他模式的教材名称。
    例如：'13.《药理学（第10版）》绪论到第六章.md' -> '药理学（第10版）'
          '《药理学（第10版）》_content.md' -> '药理学（第10版）'
    
    Args:
        content_file: 教材内容文件的路径
        
    Returns:
        提取的教材名称（不含序号和书名号），如果无法提取则返回文件basename
    """
    if not content_file:
        return "未知教材"
    
    # 获取文件名（不含路径）
    basename = os.path.basename(content_file)
    
    # 尝试匹配《...》模式，提取书名号内的内容
    match = re.search(r'《(.+?)》', basename)
    if match:
        # 直接返回书名号内的内容，不含书名号
        return match.group(1)
    
    # 如果没有匹配，尝试去除序号和扩展名
    name = re.sub(r'^\d+\.', '', basename)  # 去除开头的数字序号（如 13.）
    name = re.sub(r'\.md$', '', name, flags=re.IGNORECASE)  # 去除 .md 扩展名
    name = re.sub(r'_content$', '', name, flags=re.IGNORECASE)  # 去除 _content 后缀
    
    return name.strip() if name.strip() else "未知教材"


def get_textbooks_to_process(manifest: dict, book_ids: Optional[str]) -> List[dict]:
    """根据参数筛选要处理的教材列表"""
    all_books = manifest.get('textbooks', [])
    
    if book_ids:
        selected_ids = [b.strip() for b in book_ids.split(',')]
        books = [b for b in all_books if b.get('id') in selected_ids]
    else:
        # 默认处理所有已启用的教材
        books = [b for b in all_books if b.get('enabled', True)]
    
    return books


# ============================================================================
# 全局变量 (将在运行时初始化)
# ============================================================================

# 这些变量将在 initialize_environment() 中设置
HOST_ADDR = None
NAMESPACE = None
client = None
llm = None
vectorizer = None
batch_vectorizer = None
ner_prompt = None
triple_prompt = None
kn_prompt = None
extractor = None  # FixKnowledgeUnitExtractor instance

# 教材相关变量
KB_MD_FILE = None
KNOWLEDGEPOINTS_FILE = None

# ============================================================================
# 环境初始化函数
# ============================================================================

def initialize_environment(
    config_path: str,
    host_addr: str,
    namespace_override: Optional[str] = None,
    content_file: Optional[str] = None,
    catalog_file: Optional[str] = None,
):
    """
    初始化运行时环境：加载配置、创建客户端、设置教材路径。
    
    所有教材都写入同一个项目 (Medicine)，通过 namespace 前缀区分。
    
    Args:
        config_path: kag_config.yaml 路径
        host_addr: OpenSPG Server 地址
        namespace_override: 覆盖 Namespace (作为标签前缀，如 Medici.xxx)
        content_file: 教材内容文件路径
        catalog_file: 知识点清单文件路径 (可选, None 表示自由抽取)
    """
    global HOST_ADDR, NAMESPACE, client, llm, vectorizer, batch_vectorizer
    global ner_prompt, triple_prompt, kn_prompt
    global KB_MD_FILE, KNOWLEDGEPOINTS_FILE
    
    HOST_ADDR = host_addr
    
    # 初始化 KAG 环境
    print(f"Initializing environment with config: {config_path}")
    init_env(config_file=config_path)
    
    # 设置 Namespace (作为标签前缀，如 Medici.Chapter)
    if namespace_override:
        NAMESPACE = namespace_override
    else:
        NAMESPACE = KAG_PROJECT_CONF.namespace
    print(f"Target Namespace (label prefix): {NAMESPACE}")
    
    # 按 namespace 设置独立的 ckpt 目录（避免不同教材缓存冲突）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_base_dir = os.path.join(script_dir, "ckpt", NAMESPACE)
    os.makedirs(ckpt_base_dir, exist_ok=True)
    KAG_PROJECT_CONF.ckpt_dir = ckpt_base_dir
    print(f"Checkpoint directory: {ckpt_base_dir}")
    
    # 始终使用配置文件中的 Project ID (所有教材共用 Medicine)
    print(f"Target Project ID: {KAG_PROJECT_CONF.project_id}")
    
    # 初始化 GraphClient
    client = GraphClient(host_addr=HOST_ADDR, project_id=KAG_PROJECT_CONF.project_id)
    
    # 初始化 LLM
    print("Initializing LLM...")
    llm_config = KAG_CONFIG.all_config["openie_llm"]
    llm = LLMClient.from_config(llm_config)
    
    # 初始化 Prompts
    ner_prompt = PromptABC.from_config({"type": "knowledge_unit_ner"})
    # 确保 NER prompt 的 schema 包含 Drug 类型（与 Disease 一样）
    if hasattr(ner_prompt, 'schema'):
        original_schema = list(ner_prompt.schema) if isinstance(ner_prompt.schema, list) else []
        # 确保 Drug/药品 在 schema 中
        has_drug = any('药品' in str(s) or 'Drug' in str(s) for s in original_schema)
        if not has_drug:
            ner_prompt.schema.append('药品')
            print(f"  [Schema] 添加了 '药品' 类型到 NER schema")
        print(f"  [Schema] NER 实体类型: {ner_prompt.schema}")
    triple_prompt = PromptABC.from_config({"type": "knowledge_unit_triple"})
    kn_prompt = PromptABC.from_config({"type": "knowledge_unit"})
    kn_prompt.template_zh = """你是一名专业的医学知识点标注专家。

### 提取要求：
- 请特别注意提取 **药物/药品名称**（如：青霉素、阿司匹林、地西泮、吗啡等）
- 请提取疾病名称、症状、医学概念等

### 输入：
$input
"""
    
    # 初始化 Vectorizer
    print("Initializing Vectorizer...")
    vectorize_config = KAG_CONFIG.all_config["vectorize_model"]
    vectorizer = VectorizeModelABC.from_config(vectorize_config)
    
    # 初始化 BatchVectorizer
    from kag.builder.component.vectorizer.batch_vectorizer import BatchVectorizer
    batch_vectorizer = BatchVectorizer(vectorize_model=vectorizer, batch_size=32)
    
    # 设置教材文件路径
    KB_MD_FILE = content_file
    KNOWLEDGEPOINTS_FILE = catalog_file
    
    print(f"Content file: {KB_MD_FILE}")
    print(f"Catalog file: {KNOWLEDGEPOINTS_FILE or '(None - 自由抽取模式)'}")
    
    # 初始化知识点清单 (必须在设置 KNOWLEDGEPOINTS_FILE 之后)
    initialize_catalog()


# ============================================================================
# 配置常量
# ============================================================================


# Strict matching: final KnowledgeUnit must be from catalog; otherwise dropped.
STRICT_KU_MATCH = True
STRICT_KU_SIM_THRESHOLD = 0.65

# Encourage many-to-many: allow multiple KUs per chunk (upper bound in prompt/output)
MAX_KU_PER_CHUNK = 6

# Feature toggles
ENABLE_SUB_KU = True
ENABLE_ATOMIC_QUERY = True  # 是否启用 AtomicQuery 功能
MAX_ATOMIC_QUERY_PER_CHUNK = 5  # 每个 Chunk 最多生成的原子问数量
DEBUG_ATOMIC_QUERY = False  # 是否打印原子问调试日志

# ============ 测试模式配置 ============
TEST_MODE = False  # 禁用测试模式，进行完整抽取
TEST_MODE_CHUNK_LIMIT = 3  # 测试模式下最多处理的 chunk 数量

# ============ LLM 实体分类配置 ============
ENABLE_LLM_ENTITY_CLASSIFICATION = True  # 启用 LLM 智能分类实体类型 (Disease/Symptom/Event/Concept)

# 通用实体名黑名单 (过滤过于泛化的词)
BLACKLIST_NAMES = {
    "化学物质", "结构", "功能", "机制", "作用", "性质",
    "实验", "研究", "发展", "分类", "定义", "概念", "药理", "临床",
    "不良反应", "适应症", "禁忌症", "过程", "影响", "结果", "意义",
    "feature", "structure", "function", "mechanism"
}

# Sub knowledge points (free extraction; parented by core KnowledgeUnit)
MAX_SUB_KU_PER_CHUNK = 20
# If the model returns too few SubKUs, do one best-effort top-up call.
MIN_SUB_KU_PER_CHUNK = 8
SUB_KU_TOPUP_ROUNDS = 1
# Deterministic extraction: promote intra-chunk titles to SubKUs.
ENABLE_SUB_KU_TITLES = True
# Re-parent SubKUs by global catalog similarity if local (chunk) core KUs are missing/weak.
ENABLE_SUB_KU_GLOBAL_PARENT = True
SUB_KU_PARENT_SIM_THRESHOLD = 0.72
SUB_KU_PARENT_SIM_MARGIN = 0.05
# Hard constraint to reduce near-duplicates among SubKnowledgeUnit candidates.
# Cosine similarity above this threshold will be treated as redundant.
SUB_KU_EMBED_DEDUPE_THRESHOLD = 0.85

# If a SubKN is almost the same as an extracted core KnowledgeUnit (KN), drop it.
# This is stronger than exact-string match and catches paraphrases.
SUB_KU_DROP_SIM_TO_CORE_KU_THRESHOLD = 0.90

# If a SubKN mentions a strong topic keyword but does not match the topic of this chunk's
# core KUs (and is also semantically far), treat it as off-topic hallucination and drop.
SUB_KU_OFFTOPIC_MIN_SIM_TO_CORE = 0.55

# Quality guardrails for SubKnowledgeUnit names.
# We want SubKUs to be noun-phrase-like, specific, and graph-node-friendly.
SUB_KU_SENTENCE_MARKERS = ["见于", "多见于", "常见于", "主要见于", "发生于", "可见于", "表现为", "可见", "导致", "因此", "由于", "从而", "包括"]
SUB_KU_SPLIT_PUNCT = ["。", "；", ";", "，", ",", "：", ":", "!", "！", "?", "？"]
# If a SubKU candidate contains a whitespace-separated tail, and the tail looks like
# an explanatory sentence/description, drop the tail and keep only the noun-phrase prefix.
SUB_KU_SPACE_TAIL_MARKERS = ["主要", "多", "常", "可", "为", "是", "属于", "见", "发生", "沉积", "引起", "导致", "表现"]
SUB_KU_MIN_CJK_CHARS = 3
SUB_KU_MAX_NAME_CHARS = 40


# ============================================================================
# Chunk 内容清理函数
# ============================================================================

def clean_chunk_content(text: str) -> str:
    """
    清理 chunk 内容中的特殊符号，保留原文有意义的内容。
    
    清理项目：
    - Markdown 标题符号 (#)
    - 字面换行符 (\\n)
    - HTML 标签 (<br>, <p>, <table> 等)
    - Markdown 高亮标记 (==文字==)
    - 多余空白字符
    
    保留：
    - LaTeX 公式 ($...$)
    - 实际文字内容
    """
    if not text or not isinstance(text, str):
        return text
    
    cleaned = text
    
    # 1. 移除 HTML 表格及其内容
    cleaned = re.sub(r'<table[\s\S]*?</table>', ' ', cleaned, flags=re.IGNORECASE)
    
    # 2. 移除所有 HTML 标签 (<br>, <br/>, <p>, </p> 等)
    cleaned = re.sub(r'<[^>]*>', ' ', cleaned)
    
    # 3. 移除 Markdown 标题符号 (# ## ### 等)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'#{2,}', '', cleaned)  # 移除行内多个 #
    
    # 4. 移除字面字符串 \n (反斜杠+n，不是真正换行符)
    cleaned = cleaned.replace('\\n', ' ')
    
    # 5. 移除 Markdown 高亮标记 (==文字==)
    cleaned = re.sub(r'==([^=]+)==', r'\1', cleaned)
    cleaned = cleaned.replace('==', '')  # 移除剩余的 ==
    
    # 6. 移除 Markdown 加粗/斜体标记
    cleaned = cleaned.replace('**', '').replace('__', '')
    
    # 7. 规范化空白字符
    cleaned = re.sub(r'[\r\n]+', ' ', cleaned)  # 多个换行变单个空格
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)   # 多个空格变单个
    
    # 8. 清理 LaTeX 公式格式问题（OCR/转换产生的问题）
    # 8.1 修复 \lim  _ {x} 变成 \lim_{x} (移除多余空格)
    cleaned = re.sub(r'\\(lim|sum|prod|int|frac|sqrt|sin|cos|tan|log|ln|exp|max|min|sup|inf)\s+_\s*', r'\\\1_', cleaned)
    # 8.2 修复 \lim  ^ {x} 变成 \lim^{x}
    cleaned = re.sub(r'\\(lim|sum|prod|int|frac|sqrt|sin|cos|tan|log|ln|exp|max|min|sup|inf)\s+\^\s*', r'\\\1^', cleaned)
    # 8.3 修复 a _ {n} 变成 a_{n} (下标前的空格)
    cleaned = re.sub(r'(\w)\s+_\s*\{', r'\1_{', cleaned)
    # 8.4 修复 a ^ {n} 变成 a^{n} (上标前的空格)
    cleaned = re.sub(r'(\w)\s+\^\s*\{', r'\1^{', cleaned)
    # 8.5 修复 {\text {和}} 变成 \text{和} (移除多余大括号和空格)
    cleaned = re.sub(r'\{\\text\s*\{([^}]*)\}\}', r'\\text{\1}', cleaned)
    # 8.6 修复 \\text {xxx} 变成 \\text{xxx}
    cleaned = re.sub(r'\\text\s+\{', r'\\text{', cleaned)
    # 8.7 修复 \\mathrm {xxx} 变成 \\mathrm{xxx}
    cleaned = re.sub(r'\\mathrm\s+\{', r'\\mathrm{', cleaned)
    # 8.8 修复 \\left\{  变成 \\left\{
    cleaned = re.sub(r'\\left\s*\\\{', r'\\left\\{', cleaned)
    cleaned = re.sub(r'\\right\s*\\\}', r'\\right\\}', cleaned)
    
    # 8.9 修复不完整的 $$ 分隔符（chunking 切分导致）
    # 注意：只处理块级公式（$$...$$），不要影响行内公式（$...$）
    double_dollar_count = cleaned.count('$$')
    single_dollar_count = cleaned.count('$') - double_dollar_count * 2
    
    # 只有当有奇数个 $$ 且没有单独的 $ 行内公式时才可能需要补全
    # 如果有单独的 $ 符号，说明内容包含行内公式，不应该随意添加 $$
    if double_dollar_count % 2 != 0 and single_dollar_count == 0:
        first_pos = cleaned.find('$$')
        last_pos = cleaned.rfind('$$')
        
        # 只有当第一个 $$ 前的内容是纯 LaTeX 公式（以 \ 命令开头，没有中文）时才添加开头 $$
        before_first = cleaned[:first_pos].strip() if first_pos > 0 else ''
        latex_block_pattern = r'^\\[a-zA-Z]'  # 以 LaTeX 命令开头
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in before_first)
        
        if before_first and re.match(latex_block_pattern, before_first) and not has_chinese:
            cleaned = '$$ ' + cleaned
        else:
            # 检查最后一个 $$ 后是否是纯 LaTeX 块
            after_last = cleaned[last_pos+2:].strip() if last_pos >= 0 else ''
            has_chinese_after = any('\u4e00' <= c <= '\u9fff' for c in after_last)
            if after_last and re.match(latex_block_pattern, after_last) and not has_chinese_after:
                cleaned = cleaned + ' $$'
    
    # 9. 移除首尾空白
    cleaned = cleaned.strip()
    
    # 10. 不做额外的反斜杠处理，保持原文内容
    # 数据存储和传输过程中反斜杠会被正确保留
    
    return cleaned


def fix_latex_backslashes_in_neo4j(namespace: str):
    """
    修复 Neo4j 中 LaTeX 公式的双斜杠问题。
    
    由于 JSON 序列化时反斜杠会被转义，OpenSPG 服务器存储后可能导致
    单斜杠（如 \\lim）变成双斜杠（如 \\\\lim）。此函数在抽取完成后
    自动修复 Neo4j 中的双斜杠。
    
    Args:
        namespace: 要修复的命名空间（如 MedicineHigherMath）
    """
    import os
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
    
    print(f"[LaTeX修复] 检查 {namespace}.Chunk 中的双斜杠问题...")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session(database=neo4j_database) as session:
            # 检查是否有需要修复的节点
            count_query = f"""
                MATCH (n:`{namespace}.Chunk`)
                WHERE n.content CONTAINS '\\\\\\\\'
                RETURN count(n) as count
            """
            result = session.run(count_query)
            count = result.single()["count"]
            
            if count == 0:
                print(f"[LaTeX修复] 未发现双斜杠问题，跳过修复。")
                driver.close()
                return
            
            print(f"[LaTeX修复] 发现 {count} 个 Chunk 节点包含双斜杠，正在修复...")
            
            # 执行修复：将双斜杠替换为单斜杠
            update_query = f"""
                MATCH (n:`{namespace}.Chunk`)
                WHERE n.content CONTAINS '\\\\\\\\'
                SET n.content = REPLACE(n.content, '\\\\\\\\', '\\\\')
                RETURN count(n) as updated
            """
            result = session.run(update_query)
            updated = result.single()["updated"]
            print(f"[LaTeX修复] 已修复 {updated} 个 Chunk 节点。")
        
        driver.close()
    except Exception as e:
        print(f"[LaTeX修复] 修复失败: {e}")


def vectorize_chunks_for_namespace(namespace: str):
    """
    为指定命名空间的 Chunk 节点自动生成向量。
    
    使用 SiliconFlow API 调用 Qwen3-Embedding-8B 模型生成 4096 维向量，
    然后更新到 Neo4j 中每个 Chunk 节点的 contentVector 属性。
    
    Args:
        namespace: 要向量化的命名空间（如 Pathology、Pharmacology）
    """
    import os
    import requests
    import time
    from neo4j import GraphDatabase
    
    # 配置
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
    
    api_key = os.environ.get("VECTORIZE_API_KEY", "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz")
    base_url = os.environ.get("VECTORIZE_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.environ.get("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-8B")
    
    chunk_label = f"{namespace}.Chunk"
    
    def get_embedding(text: str, max_len: int = 2000) -> list:
        """调用 API 获取文本向量"""
        if not text:
            return None
        text = text[:max_len]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": model, "input": text}
        
        try:
            resp = requests.post(f"{base_url}/embeddings", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"  向量化失败: {e}")
            return None
    
    print(f"[向量化] 开始为 {namespace}.Chunk 生成向量...")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session(database=neo4j_database) as session:
            # 查询需要向量化的 Chunk
            result = session.run(f"""
                MATCH (c:`{chunk_label}`)
                WHERE c.contentVector IS NULL AND c.content IS NOT NULL
                RETURN c.id AS id, c.content AS content
            """)
            
            chunks = list(result)
            total = len(chunks)
            
            if total == 0:
                # 检查已有向量的数量
                count_result = session.run(f"""
                    MATCH (c:`{chunk_label}`) WHERE c.contentVector IS NOT NULL
                    RETURN count(c) AS count
                """)
                existing = count_result.single()["count"]
                print(f"[向量化] 所有 Chunk ({existing} 个) 都已有向量，跳过。")
                driver.close()
                return
            
            print(f"[向量化] 找到 {total} 个需要向量化的 Chunk")
            
            success = 0
            failed = 0
            
            for i, record in enumerate(chunks):
                chunk_id = record["id"]
                content = record["content"]
                
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[向量化] 处理进度: {i+1}/{total}")
                
                vector = get_embedding(content)
                
                if vector:
                    session.run(f"""
                        MATCH (c:`{chunk_label}` {{id: $id}})
                        SET c.contentVector = $vector
                    """, id=chunk_id, vector=vector)
                    success += 1
                else:
                    failed += 1
                
                # 避免 API 限流
                time.sleep(0.3)
            
            print(f"[向量化] 完成! 成功: {success}, 失败: {failed}")
        
        driver.close()
    except Exception as e:
        print(f"[向量化] 失败: {e}")


def _normalize_ku_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # drop markdown emphasis
    s = s.replace("**", "").replace("__", "")
    # normalize whitespace
    s = re.sub(r"\s+", "", s)
    # unify full-width parentheses
    s = s.replace("（", "(").replace("）", ")")
    # remove bracketed english/latin alias to improve matching
    s = re.sub(r"\([^)]*\)", "", s)
    return s.strip()


def _normalize_chapter_title(title: str) -> str:
    """Normalize chapter headings for matching between book and catalog."""
    if not isinstance(title, str):
        return ""
    s = title.strip().replace("**", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("：", ":")
    if s == "绪论":
        return s
    # "第一章细胞..." -> "细胞..."; if no suffix, keep original
    if s.startswith("第") and "章" in s:
        suffix = s.split("章", 1)[1]
        return suffix or s
    return s


def _normalize_section_title(title: str) -> str:
    """Normalize section headings for matching between book and catalog."""
    if not isinstance(title, str):
        return ""
    s = title.strip().replace("**", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("：", ":")
    # "第一节适应" -> "适应"; if no suffix, keep original
    if s.startswith("第") and "节" in s:
        suffix = s.split("节", 1)[1]
        return suffix or s
    return s


def load_catalog_heading_titles(md_path: str):
    """Load chapter/section heading titles from the knowledge-point catalog markdown.

    Returns:
      chapter_title_by_idx: Dict[str, str]  (key: chapter idx, e.g. "一", "四", "1", or "绪论")
      section_title_by_idx: Dict[Tuple[str, str], str] (key: (chapter idx, section idx))

    Notes:
    - Titles are returned without ** markdown emphasis.
    - Using idx-based mapping is more robust than suffix-text matching.
    """
    if not os.path.exists(md_path):
        return {}, {}

    chapter_title_by_idx: Dict[str, str] = {}
    section_title_by_idx: Dict[tuple, str] = {}
    current_chapter_idx = ""

    bullet_re = re.compile(r"^(?P<indent>\s*)\*\s+(?P<item>.+?)\s*$")
    with open(md_path, "r", encoding="utf-8") as f:
        for raw in f:
            m = bullet_re.match(raw.rstrip("\n"))
            if not m:
                continue
            item = m.group("item").strip()
            if item.startswith("**") and item.endswith("**"):
                item = item.replace("**", "").strip()

            if re.match(r"^绪论\b", item):
                current_chapter_idx = "绪论"
                chapter_title_by_idx[current_chapter_idx] = item
                continue

            m_ch = re.match(r"^第(?P<idx>.+?)章\b", item)
            if m_ch:
                current_chapter_idx = (m_ch.group("idx") or "").strip()
                if current_chapter_idx:
                    chapter_title_by_idx[current_chapter_idx] = item
                continue

            m_sec = re.match(r"^第(?P<idx>.+?)节\b", item)
            if m_sec:
                sec_idx = (m_sec.group("idx") or "").strip()
                if current_chapter_idx and sec_idx:
                    section_title_by_idx[(current_chapter_idx, sec_idx)] = item
                continue

    return chapter_title_by_idx, section_title_by_idx


def load_knowledge_points(md_path: str) -> List[str]:
    """Load leaf knowledge points from the catalog markdown.

    The file is a nested bullet list with chapter/section headings in ** **.
    We treat the deepest bullet items as candidate knowledge points.
    """

    if not os.path.exists(md_path):
        print(f"WARNING: knowledge points file not found: {md_path}")
        return []

    points: List[str] = []
    with open(md_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("*"):
                continue

            # Strip leading bullet markers like '*', '*   '
            item = line.lstrip("*").strip()

            # Headings are wrapped in **...**; keep them only if they look like leaf points
            # Example leaf: '**第一节 适应**' is a section heading; we don't want it as KU.
            item_clean = item
            if item_clean.startswith("**") and item_clean.endswith("**"):
                # unwrap markdown emphasis
                item_clean = item_clean.replace("**", "").strip()

            # Skip chapter/section headings
            if re.match(r"^(绪论|第.+章|第.+节)\b", item_clean):
                continue

            item_clean = item_clean.strip()
            if not item_clean:
                continue

            points.append(item_clean)

    # de-dup while preserving order
    seen = set()
    ordered: List[str] = []
    for p in points:
        key = _normalize_ku_name(p)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(p)
    return ordered


def load_knowledge_points_outline(md_path: str):
    """Load hierarchical knowledge points from the catalog markdown.

    Returns:
      all_points: List[str]
      by_chapter: Dict[str, List[str]]   (key: normalized chapter title)
      by_section: Dict[Tuple[str, str], List[str]] (key: (norm_chapter, norm_section))
    """

    if not os.path.exists(md_path):
        print(f"WARNING: knowledge points file not found: {md_path}")
        return [], {}, {}

    all_points: List[str] = []
    by_chapter: Dict[str, List[str]] = {}
    by_section: Dict[tuple, List[str]] = {}

    current_chapter_key = ""
    current_section_key = ""

    bullet_re = re.compile(r"^(?P<indent>\s*)\*\s+(?P<item>.+?)\s*$")
    with open(md_path, "r", encoding="utf-8") as f:
        for raw in f:
            m = bullet_re.match(raw.rstrip("\n"))
            if not m:
                continue
            item = m.group("item").strip()
            if item.startswith("**") and item.endswith("**"):
                item = item.replace("**", "").strip()

            # Chapter heading
            if re.match(r"^(绪论|第.+章)\b", item):
                current_chapter_key = _normalize_chapter_title(item)
                current_section_key = ""
                continue

            # Section heading
            if re.match(r"^第.+节\b", item):
                current_section_key = _normalize_section_title(item)
                continue

            # Leaf point
            if not item:
                continue
            all_points.append(item)
            if current_chapter_key:
                by_chapter.setdefault(current_chapter_key, []).append(item)
                if current_section_key:
                    by_section.setdefault(
                        (current_chapter_key, current_section_key), []
                    ).append(item)

    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for p in items:
            k = _normalize_ku_name(p)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out

    all_points = _dedup_keep_order(all_points)
    by_chapter = {k: _dedup_keep_order(v) for k, v in by_chapter.items()}
    by_section = {k: _dedup_keep_order(v) for k, v in by_section.items()}
    return all_points, by_chapter, by_section


# ============================================================================
# 延迟初始化的全局变量 (将在 initialize_catalog() 中设置)
# ============================================================================
ALLOWED_KNOWLEDGE_POINTS = []
CATALOG_ALL_POINTS = []
CATALOG_BY_CHAPTER = {}
CATALOG_BY_SECTION = {}
CATALOG_CHAPTER_TITLES = {}
CATALOG_SECTION_TITLES = {}
ALLOWED_KU_MAP = {}
ALLOWED_KU_INDEX = {}
ALLOWED_KU_MATRIX = None
TOPIC_KEYWORDS = []
ALLOWED_KU_KEYWORD_IDXS = {}
SUBKU_ENGINE = None


def initialize_catalog():
    """初始化知识点清单和相关数据结构。
    
    必须在 initialize_environment() 之后调用，因为需要 KNOWLEDGEPOINTS_FILE 和 vectorizer。
    """
    global ALLOWED_KNOWLEDGE_POINTS, CATALOG_ALL_POINTS, CATALOG_BY_CHAPTER, CATALOG_BY_SECTION
    global CATALOG_CHAPTER_TITLES, CATALOG_SECTION_TITLES, ALLOWED_KU_MAP, ALLOWED_KU_INDEX
    global ALLOWED_KU_MATRIX, TOPIC_KEYWORDS, ALLOWED_KU_KEYWORD_IDXS, SUBKU_ENGINE
    
    # 加载知识点清单
    ALLOWED_KNOWLEDGE_POINTS = load_knowledge_points(KNOWLEDGEPOINTS_FILE) if KNOWLEDGEPOINTS_FILE else []
    
    if KNOWLEDGEPOINTS_FILE:
        CATALOG_ALL_POINTS, CATALOG_BY_CHAPTER, CATALOG_BY_SECTION = load_knowledge_points_outline(
            KNOWLEDGEPOINTS_FILE
        )
        CATALOG_CHAPTER_TITLES, CATALOG_SECTION_TITLES = load_catalog_heading_titles(KNOWLEDGEPOINTS_FILE)
    else:
        CATALOG_ALL_POINTS, CATALOG_BY_CHAPTER, CATALOG_BY_SECTION = [], {}, {}
        CATALOG_CHAPTER_TITLES, CATALOG_SECTION_TITLES = {}, {}
    
    ALLOWED_KU_MAP = {_normalize_ku_name(p): p for p in ALLOWED_KNOWLEDGE_POINTS}
    
    # Index for slicing embedding matrix quickly
    ALLOWED_KU_INDEX = {
        _normalize_ku_name(p): i for i, p in enumerate(ALLOWED_KNOWLEDGE_POINTS)
    }
    
    # Precompute catalog embeddings (best-effort) for fast similarity mapping
    ALLOWED_KU_MATRIX = None
    if ALLOWED_KNOWLEDGE_POINTS and vectorizer:
        try:
            _vecs = vectorizer.vectorize(ALLOWED_KNOWLEDGE_POINTS)
            _M = np.array(_vecs)
            _norms = np.linalg.norm(_M, axis=1, keepdims=True)
            _norms[_norms == 0] = 1
            ALLOWED_KU_MATRIX = _M / _norms
        except Exception as e:
            print(f"WARNING: failed to precompute catalog embeddings: {e}")
    
    # Topic keywords used to keep SubKnowledgeUnit parenting consistent.
    TOPIC_KEYWORDS = []
    
    # Fast lookup of catalog KU indices by keyword (best-effort).
    ALLOWED_KU_KEYWORD_IDXS = {k: [] for k in TOPIC_KEYWORDS}
    if ALLOWED_KNOWLEDGE_POINTS:
        try:
            for i, p in enumerate(ALLOWED_KNOWLEDGE_POINTS):
                pn = _normalize_ku_name(p)
                for k in TOPIC_KEYWORDS:
                    if k in pn:
                        ALLOWED_KU_KEYWORD_IDXS[k].append(i)
        except Exception:
            pass
    
    if ALLOWED_KNOWLEDGE_POINTS:
        print(f"Loaded {len(ALLOWED_KNOWLEDGE_POINTS)} knowledge points from catalog.")
    else:
        print("WARNING: No knowledge points loaded; extraction will be unconstrained.")
    
    # 初始化 SubKU 引擎
    SUBKU_ENGINE = SubKUEngine(
        vectorizer=vectorizer,
        normalize_ku_name=_normalize_ku_name,
        topic_keywords=TOPIC_KEYWORDS,
        allowed_knowledge_points=ALLOWED_KNOWLEDGE_POINTS,
        allowed_ku_matrix=ALLOWED_KU_MATRIX,
        allowed_ku_keyword_idxs=ALLOWED_KU_KEYWORD_IDXS,
        enable_sub_ku_global_parent=ENABLE_SUB_KU_GLOBAL_PARENT,
        sub_ku_parent_sim_threshold=SUB_KU_PARENT_SIM_THRESHOLD,
        sub_ku_parent_sim_margin=SUB_KU_PARENT_SIM_MARGIN,
        sub_ku_embed_dedupe_threshold=SUB_KU_EMBED_DEDUPE_THRESHOLD,
        sub_ku_drop_sim_to_core_threshold=SUB_KU_DROP_SIM_TO_CORE_KU_THRESHOLD,
        sub_ku_offtopic_min_sim_to_core=SUB_KU_OFFTOPIC_MIN_SIM_TO_CORE,
        sub_ku_drop_exact=set(),
        sub_ku_sentence_markers=SUB_KU_SENTENCE_MARKERS,
        sub_ku_split_punct=SUB_KU_SPLIT_PUNCT,
        sub_ku_space_tail_markers=SUB_KU_SPACE_TAIL_MARKERS,
        sub_ku_min_cjk_chars=SUB_KU_MIN_CJK_CHARS,
        sub_ku_max_name_chars=SUB_KU_MAX_NAME_CHARS,
        generate_hash_id=generate_hash_id,
    )
    
    # 设置默认的 kn_prompt 模板
    if kn_prompt:
        kn_prompt.template_zh = _build_kn_template(ALLOWED_KNOWLEDGE_POINTS)
    
    # 初始化 Extractor (需要放在函数末尾，确保 FixKnowledgeUnitExtractor 类已定义)
    # 注意：由于 FixKnowledgeUnitExtractor 定义在后面，这里使用延迟初始化
    global extractor
    # extractor 将在 run_extraction 中按需创建，因为类定义在文件后面


def _candidate_points_for_context(chapter_title: str, section_title: str) -> List[str]:
    """Return candidates scoped to the current chunk (section > chapter > global)."""
    if not CATALOG_ALL_POINTS:
        return ALLOWED_KNOWLEDGE_POINTS

    ckey = _normalize_chapter_title(chapter_title or "")
    skey = _normalize_section_title(section_title or "")
    if ckey and skey and (ckey, skey) in CATALOG_BY_SECTION:
        return CATALOG_BY_SECTION[(ckey, skey)]
    if ckey and ckey in CATALOG_BY_CHAPTER:
        return CATALOG_BY_CHAPTER[ckey]
    return CATALOG_ALL_POINTS


def _build_kn_template(candidate_points: List[str], chapter_title: str = "", section_title: str = "") -> str:
    lines = "\n".join([f"- {p}" for p in candidate_points]) if candidate_points else "- （候选为空）"
    ctx = ""
    if chapter_title or section_title:
        ctx = f"\n### 本 chunk 所在位置：\n- 章节：{chapter_title or '未知'}\n- 小节：{section_title or '未知'}\n"
    
    # 从候选列表中取前3个作为动态示例（如果有的话）
    example_items = candidate_points[:3] if candidate_points else ["知识点A", "知识点B", "知识点C"]
    example_json = json.dumps(example_items, ensure_ascii=False)
    
    return f"""你是一名专业的医学知识点标注专家。

你的任务：从输入的文档片段（chunk）中，**只选择**与其内容相关、且出现在下面《知识点清单（本章/本节子集）》中的知识点。
请尽可能完整列出该 chunk 涉及的所有知识点（可能为 0~{MAX_KU_PER_CHUNK} 个），不要只输出一个。

### 关键要求（必须遵守）：
1. **候选约束**：输出的每一个知识点名称，必须来自《知识点清单（本章/本节子集）》。如果文本描述与某个知识点"相近/同义/包含关系"，也要映射到清单中的最合适条目。
2. **多对多**：同一个 chunk 可以对应多个知识点；同一个知识点也可以对应多个 chunk（系统会为每次出现建立关联）。
3. **只做识别**：无需生成长段解释；每个知识点的内容字段填知识点名称即可。
4. **强制包含**：如果 chunk 中出现了明确的标题/小标题，则必须在输出数组中包含对应的知识点。
{ctx}
### 《知识点清单（本章/本节子集）》：
{lines}

### 输出格式（严格 JSON，仅输出 JSON，不要附加任何解释文字）：
- 输出必须是一个 JSON 数组（list），元素是知识点名称字符串。
- 数组长度最多 {MAX_KU_PER_CHUNK}，按相关性从高到低排序。
- 每个元素必须来自《知识点清单（本章/本节子集）》。

示例：
{example_json}

### 输入：
$input
"""


# Default template will be set inside initialize_catalog() after kn_prompt is initialized.


def _extract_forced_kus_from_text(text: str, candidate_points: List[str]) -> List[str]:
    """Force-add obvious KUs explicitly mentioned in the chunk.

    Scope is limited to candidate_points (chapter/section scoped) to reduce noise.
    - Body match: normalized candidate term appears in normalized text
    """
    if not text or not candidate_points:
        return []

    # Normalize full text for substring match (remove whitespace, normalize parentheses)
    normalized_text = re.sub(r"\s+", "", text)
    normalized_text = normalized_text.replace("（", "(").replace("）", ")")
    normalized_text_for_match = _normalize_ku_name(normalized_text)

    # Extract markdown headings
    headings = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("#"):
            continue
        h = line.lstrip("#").strip()
        # strip common numbering prefixes: 一、 (一) 1. （1） etc.
        h = re.sub(r"^[（(]?[一二三四五六七八九十0-9]+[)）]?\s*[、.．]?\s*", "", h)
        h = re.sub(r"\s+", "", h)
        if h:
            headings.append(h)

    forced: List[str] = []
    seen = set()
    for p in candidate_points:
        p_norm = _normalize_ku_name(p)
        if not p_norm or len(p_norm) < 2:
            continue
        # Heading containment (robust for "萎缩的类型")
        if any((p_norm in hh) or (hh in p_norm) for hh in headings):
            if p_norm not in seen:
                forced.append(p)
                seen.add(p_norm)
            continue
        # Body match
        if p_norm in normalized_text_for_match:
            if p_norm not in seen:
                forced.append(p)
                seen.add(p_norm)
    return forced


def _strip_code_fences(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
        s = s.strip()
    return s


def _parse_llm_json(raw_value):
    """Best-effort parse for LLM outputs.

    Accepts list/dict directly, or parses JSON from a string (with/without code fences).
    Falls back to extracting the outermost [] or {} block.
    """
    if isinstance(raw_value, (list, dict)):
        return raw_value
    if not isinstance(raw_value, str):
        return None

    txt = _strip_code_fences(raw_value)
    try:
        return json.loads(txt)
    except Exception:
        pass

    lb = txt.find("[")
    rb = txt.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        try:
            return json.loads(txt[lb : rb + 1])
        except Exception:
            pass

    ob = txt.find("{")
    cb = txt.rfind("}")
    if ob != -1 and cb != -1 and cb > ob:
        try:
            return json.loads(txt[ob : cb + 1])
        except Exception:
            pass

    return None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items or []:
        s = (x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_llm_string_list(
    raw_value,
    *,
    dict_list_keys: tuple = (),
    allow_dict_keys_as_items: bool = False,
    question_first: bool = False,
) -> List[str]:
    """Parse a list[str] from an LLM response.

    - If JSON list: returns list of strings
    - If JSON dict: optionally reads list from specific keys, or uses dict keys as items
    - If plain text: falls back to newline/bullet list
    """
    parsed0 = _parse_llm_json(raw_value)

    if isinstance(parsed0, list):
        out = [str(x).strip() for x in parsed0 if str(x).strip()]
        return _dedupe_keep_order(out)

    if isinstance(parsed0, dict):
        for k in dict_list_keys or ():
            inner = parsed0.get(k)
            if isinstance(inner, list):
                out = [str(x).strip() for x in inner if str(x).strip()]
                return _dedupe_keep_order(out)
        if allow_dict_keys_as_items:
            out = [str(k).strip() for k in parsed0.keys() if str(k).strip()]
            return _dedupe_keep_order(out)
        return []

    if not isinstance(raw_value, str):
        return []

    txt = _strip_code_fences(raw_value)
    lines: List[str] = []
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^[0-9]+[.、）)]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line:
            lines.append(line)

    if question_first:
        questions = [l for l in lines if ("？" in l or "?" in l)]
        others = [l for l in lines if l not in questions]
        lines = questions + others

    return _dedupe_keep_order(lines)


def _parse_ku_name_list(raw_value) -> List[str]:
    """Parse a KU name list from an LLM response.

    Handles:
    """
    parsed = raw_value
    if isinstance(raw_value, str):
        parsed = _parse_llm_json(raw_value)

    if isinstance(parsed, list):
        names: List[str] = []
        for item in parsed:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    names.append(s)
                continue
            if isinstance(item, dict):
                k = (
                    item.get("知识点名称")
                    or item.get("name")
                    or item.get("名称")
                    or item.get("title")
                )
                if k and str(k).strip():
                    names.append(str(k).strip())
        return _dedupe_keep_order(names)

    if isinstance(parsed, dict):
        if "SingleExtractedKnowledge" in parsed and isinstance(parsed.get("SingleExtractedKnowledge"), dict):
            inner_obj = parsed.get("SingleExtractedKnowledge") or {}
            title = (
                inner_obj.get("知识点名称")
                or inner_obj.get("name")
                or inner_obj.get("名称")
                or inner_obj.get("title")
            )
            return [str(title).strip()] if title and str(title).strip() else []

        if any(k in parsed for k in ("内容", "Content", "知识类型", "Knowledge Type")) and not any(
            k in parsed for k in ("knowledge_units", "知识点", "result", "data")
        ):
            return []

        for key in ("knowledge_units", "知识点", "result", "data"):
            inner = parsed.get(key)
            if isinstance(inner, dict):
                return _dedupe_keep_order([str(k).strip() for k in inner.keys() if str(k).strip()])

        return _dedupe_keep_order([str(k).strip() for k in parsed.keys() if str(k).strip()])

    # Fallback: treat as newline/bullet list
    return _parse_llm_string_list(raw_value)

class FixKnowledgeUnitExtractor(KnowledgeUnitSchemaFreeExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _map_to_allowed_knowledge_point(self, name: str, candidate_points: List[str] = None):
        """Map a raw extracted name to the closest catalog knowledge point.

        Priority:
        1) normalized exact match in catalog
        2) (optional) embedding similarity match
        3) fallback to cleaned name
        """

        raw = (name or "").strip()
        norm = _normalize_ku_name(raw)

        # If catalog is empty, cannot enforce strictness
        if not ALLOWED_KNOWLEDGE_POINTS:
            return None if STRICT_KU_MATCH else (raw or (name or ""))

        # Prefer scoped candidates if provided
        scoped_points = candidate_points or []
        if scoped_points:
            scoped_map = {_normalize_ku_name(p): p for p in scoped_points}
            if norm in scoped_map:
                return scoped_map[norm]
        else:
            if norm in ALLOWED_KU_MAP:
                return ALLOWED_KU_MAP[norm]

        # Heuristic containment mapping (robust for headings like "萎缩的类型", "一、萎缩", "萎缩(atrophy)")
        # Prefer scoped candidates to avoid cross-chapter noise.
        def _best_containment_match(raw_norm: str, options: List[str]):
            best = ""
            best_len = 0
            for opt in options:
                opt_norm = _normalize_ku_name(opt)
                if not opt_norm:
                    continue
                if (opt_norm in raw_norm) or (raw_norm in opt_norm):
                    if len(opt_norm) > best_len:
                        best = opt
                        best_len = len(opt_norm)
            return best

        if scoped_points:
            c = _best_containment_match(norm, scoped_points)
            if c:
                return c
        else:
            c = _best_containment_match(norm, ALLOWED_KNOWLEDGE_POINTS)
            if c:
                return c

        # Embedding-based fallback (best-effort), scoped first then global
        try:
            vec = vectorizer.vectorize([raw])[0]
            v = np.array(vec)
            n = np.linalg.norm(v)
            if n == 0 or ALLOWED_KU_MATRIX is None:
                return ALLOWED_KU_MAP.get(norm, raw) or raw
            v = v / n

            def _best_from_points(points: List[str]):
                idxs = []
                for p in points:
                    k = _normalize_ku_name(p)
                    if k in ALLOWED_KU_INDEX:
                        idxs.append(ALLOWED_KU_INDEX[k])
                if not idxs:
                    return None
                sims_local = np.dot(ALLOWED_KU_MATRIX[idxs], v)
                local_best = int(np.argmax(sims_local))
                best_idx = idxs[local_best]
                return float(sims_local[local_best]), ALLOWED_KNOWLEDGE_POINTS[best_idx]

            if scoped_points:
                r = _best_from_points(scoped_points)
                if r and r[0] >= STRICT_KU_SIM_THRESHOLD:
                    return r[1]

            sims = np.dot(ALLOWED_KU_MATRIX, v)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= STRICT_KU_SIM_THRESHOLD:
                return ALLOWED_KNOWLEDGE_POINTS[best_idx]
        except Exception:
            pass

        # Strict mode: anything not confidently mapped is dropped
        if STRICT_KU_MATCH:
            return None

        return ALLOWED_KU_MAP.get(norm, raw) or raw

    def triples_extraction(self, passage, entities):
        # 调用父类的关系抽取方法
        return super().triples_extraction(passage, entities)

    def _build_sub_ku_prompt(self) -> str:
        return SUBKU_ENGINE.build_sub_ku_prompt(MAX_SUB_KU_PER_CHUNK)

    def _extract_subku_titles_from_chunk(self, passage: str) -> List[str]:
        return SUBKU_ENGINE.extract_subku_titles_from_chunk(passage)

    def _filter_sub_ku_names(self, names: List[str]) -> List[str]:
        return SUBKU_ENGINE.filter_sub_ku_names(names)

    def _filter_sub_ku_not_equal_knowledge_units(self, names: List[str], core_kus: List[dict]) -> List[str]:
        return SUBKU_ENGINE.filter_sub_ku_not_equal_knowledge_units(names, core_kus)

    def _filter_sub_ku_not_too_similar_to_core_kus(
        self,
        names: List[str],
        core_kus: List[dict],
        threshold: float = SUB_KU_DROP_SIM_TO_CORE_KU_THRESHOLD,
    ) -> List[str]:
        return SUBKU_ENGINE.filter_sub_ku_not_too_similar_to_core_kus(names, core_kus, threshold)

    def _filter_sub_ku_off_topic_by_core_kus(
        self,
        names: List[str],
        core_kus: List[dict],
        min_sim: float = SUB_KU_OFFTOPIC_MIN_SIM_TO_CORE,
    ) -> List[str]:
        return SUBKU_ENGINE.filter_sub_ku_off_topic_by_core_kus(names, core_kus, min_sim)

    def _dedupe_sub_kus_by_containment(self, names: List[str]) -> List[str]:
        return SUBKU_ENGINE.dedupe_sub_kus_by_containment(names)

    def _extract_topic_keywords(self, text: str) -> set:
        return SUBKU_ENGINE.extract_topic_keywords(text)

    def _subku_soft_key(self, name: str) -> str:
        return SUBKU_ENGINE._subku_soft_key(name)

    def _dedupe_sub_kus_by_soft_key(self, names: List[str]) -> List[str]:
        return SUBKU_ENGINE.dedupe_sub_kus_by_soft_key(names)

    def _pick_parent_ku_for_sub(self, sub_name: str, core_kus: List[dict]):
        return SUBKU_ENGINE.pick_parent_ku_for_sub(sub_name, core_kus)

    def _dedupe_sub_kus_by_embedding(
        self,
        names: List[str],
        threshold: float = SUB_KU_EMBED_DEDUPE_THRESHOLD,
    ) -> List[str]:
        return SUBKU_ENGINE.dedupe_sub_kus_by_embedding(names, threshold)

    def assemble_sub_knowledge_unit(
        self,
        sub_graph: SubGraph,
        sub_names: List[str],
        core_kus: List[dict],
        section_id: str,
        chunk_id: str,
    ):
        return SUBKU_ENGINE.assemble_sub_knowledge_unit(
            sub_graph=sub_graph,
            sub_names=sub_names,
            core_kus=core_kus,
            section_id=section_id,
            chunk_id=chunk_id,
            max_sub_ku_per_chunk=MAX_SUB_KU_PER_CHUNK,
            assemble_node_cb=self.assemble_sub_graph_with_spg_properties,
        )

    def named_entity_recognition(self, passage: str):
        # Call parent's NER
        entities = super().named_entity_recognition(passage)
        
        # 中文类型 -> 英文 schema 类型映射
        TYPE_MAPPING = {
            "疾病": "Disease",
            "概念": "Concept",
            "症状": "Symptom",
            "学科": "Concept",  # 学科归类为概念
            "人物": "Others",
            "技术": "Concept",
            "活动": "Event",
            "对象": "Concept",
            "术语": "Concept",
            "药物": "Drug",
            "其他": "Others",
            "Medicine": "Drug",
        }
        
        cleaned_entities = []
        entities_to_reclassify = []  # 需要 LLM 重新分类的实体
        
        for ent in entities:
            name = ent.get("name", "").strip()
            category = ent.get("category", "Others")
            
            # 0. 中文类型映射到英文
            if category in TYPE_MAPPING:
                category = TYPE_MAPPING[category]
                ent["category"] = category
            
            # 1. Medicine -> Drug (不是 Concept)
            if category == "Medicine":
                category = "Drug"
                ent["category"] = "Drug"
            
            # 1.5 药物名称模式匹配 - 确保药物被正确识别
            DRUG_SUFFIXES = ("素", "平", "宁", "唑", "酮", "因", "啉", "胺", "酸", "肽", "定", "沙星", "西林", "霉素", "他汀", "普利", "洛尔", "沙坦", "地平", "拉唑", "司琼")
            DRUG_KEYWORDS = ("阿司匹林", "吗啡", "胰岛素", "青霉素", "头孢", "阿托品", "地塞米松", "氨苄", "红霉素", "利多卡因", "奥美拉唑", "氯丙嗪", "氟西汀", "阿莫西林")
            if category in ("Concept", "Others"):
                # 检查是否为药物
                if any(name.endswith(suffix) for suffix in DRUG_SUFFIXES) or any(kw in name for kw in DRUG_KEYWORDS):
                    category = "Drug"
                    ent["category"] = "Drug"
            
            # 2. Name Filter (Block generic terms)
            if name in BLACKLIST_NAMES:
                continue
            if len(name) < 2:
                continue
            
            cleaned_entities.append(ent)
            
            # 3. 收集需要重新分类的实体（原类型为 Concept/Others 的）
            if category in ("Concept", "Others"):
                entities_to_reclassify.append(ent)
        
        # 4. 使用 LLM 批量重新分类实体
        if entities_to_reclassify and ENABLE_LLM_ENTITY_CLASSIFICATION:
            self._llm_classify_entities(entities_to_reclassify)
        
        # 统计分类结果（静默模式）
        # type_counts = {}
        # for e in cleaned_entities:
        #     c = e.get("category", "Unknown")
        #     type_counts[c] = type_counts.get(c, 0) + 1
            
        return cleaned_entities
    
    def _llm_classify_entities(self, entities: List[Dict]):
        """使用 LLM 批量分类实体类型"""
        if not entities:
            return
        
        entity_names = [e.get("name", "") for e in entities]
        names_str = "\n".join([f"- {n}" for n in entity_names])
        
        prompt = f"""你是医学术语分类专家。请将以下医学术语分类为以下类型之一：
- Disease（疾病）: 具体的疾病名称，如"肺炎"、"糖尿病"、"心肌梗死"
- Symptom（症状）: 疾病的表现，如"发热"、"头痛"、"咳嗽"
- Drug（药物）: 药品/药物名称，包括：
  * 西药：阿司匹林、青霉素、胰岛素、吗啡、阿托品、地西泮、氯丙嗪
  * 抗生素：头孢菌素、红霉素、氨苄西林、阿莫西林
  * 降压药：卡托普利、氨氯地平、洛沙坦
  * 以"素"、"平"、"宁"、"唑"、"因"、"胺"、"酮"、"定"结尾的药物名
- Event（事件）: 医疗活动，如"手术"、"治疗"、"检查"
- Concept（概念）: 医学概念、理论、因素、过程，如"病理学"、"药理"、"细胞"

【重要规则】：
- "XXX因素" 形式的术语（如"遗传因素"、"生理因素"、"病理因素"、"环境因素"）是 Concept，不是 Drug！
- "病理学"、"药理学"是学科，归类为 Concept
- 只有具体的药物名称才归类为 Drug，如"阿司匹林"、"青霉素"
- 含有"影响XXX的因素"、"决定因素"等描述性术语归类为 Concept

待分类术语：
{names_str}

请用 JSON 格式返回，例如：{{"肺炎": "Disease", "病理因素": "Concept", "阿司匹林": "Drug"}}
只返回 JSON，不要其他内容。"""

        try:
            # 创建临时 prompt 对象
            classify_prompt = copy.deepcopy(self.kn_prompt)
            classify_prompt.template_zh = prompt
            
            # 使用正确的 KAG LLMClient 调用方式
            response = self.llm.invoke(
                {"input": ""},  # 空输入，因为 prompt 已包含所有内容
                classify_prompt,
                with_except=False,
                with_json_parse=False,
            )
            result_text = str(response).strip()
            
            # 解析 JSON
            import json
            # 尝试提取 JSON
            if "{" in result_text and "}" in result_text:
                json_str = result_text[result_text.index("{"):result_text.rindex("}")+1]
                # 修复：LLM 可能返回单引号 JSON，替换为双引号
                json_str = json_str.replace("'", '"')
                classifications = json.loads(json_str)
                
                # 统计各类型分类结果（只处理字符串值）
                type_counts = {}
                for k, v in classifications.items():
                    if isinstance(v, str):
                        type_counts[v] = type_counts.get(v, 0) + 1
                if type_counts:
                    print(f"  [LLM分类] 结果: {type_counts}")
                
                # 应用分类结果
                for ent in entities:
                    name = ent.get("name", "")
                    if name in classifications:
                        new_category = classifications[name]
                        # 只接受字符串类型的分类
                        if isinstance(new_category, str) and new_category in ("Disease", "Symptom", "Event", "Concept", "Drug"):
                            # 【后处理校验】：检查 Drug 分类是否合理
                            if new_category == "Drug":
                                # "因素"类术语不应被归类为 Drug
                                if "因素" in name or "因子" in name:
                                    print(f"  [LLM分类修正] '{name}' 误判为 Drug，修正为 Concept")
                                    new_category = "Concept"
                            ent["category"] = new_category
        except Exception as e:
            print(f"  [LLM分类] 失败: {e}")  # 不再静默处理

    def assemble_sub_graph_with_spg_records(self, entities: List[Dict]):
        sub_graph = SubGraph([], [])
        # WHITELIST: Only allow these types
        # Others 类型已禁用以提高效率，不再抽取无意义的实体
        ALLOWED_TYPES = ["Concept", "Drug", "Disease", "Symptom", "Event"] 

        filtered_entities = []
        for record in entities:
            category = record.get("category", "")
            name = record.get("name", "").strip()
            
            # 1. Category Filter
            if category not in ALLOWED_TYPES and category != "KnowledgeUnit":
                continue
            
            # 2. Name Filter
            if len(name) < 2:
                continue
                
            if name in BLACKLIST_NAMES:
                continue

            s_name = name
            s_label = category
            
            self.assemble_sub_graph_with_spg_properties(
                sub_graph, s_name, s_name, s_label, record
            )
            filtered_entities.append(record)
            
        return sub_graph, filtered_entities

    def assemble_sub_graph_with_spg_properties(
        self, sub_graph: SubGraph, s_id, s_name, s_label, record
    ):
        properties = record
        tmp_properties = copy.deepcopy(record)
        original_label = s_label
        s_label = self.get_stand_schema(s_label)
        spg_type = self.schema.get(s_label)

        # --- SPECIAL HANDLING FOR SubKnowledgeUnit ---
        # 如果 schema 中没有 SubKnowledgeUnit 类型定义，直接保留所有属性
        if spg_type is None and "SubKnowledgeUnit" in s_label:
            # 确保 name 属性存在
            if "name" not in tmp_properties and s_name:
                tmp_properties["name"] = s_name
            sub_graph.add_node(
                id=s_id, name=s_name, label=s_label, properties=tmp_properties
            )
            return

        # --- 无法归类的实体直接跳过 ---
        # If the type is not defined in the schema, skip it
        if spg_type is None:
            return
        
        # Safety check: skip if still None
        if spg_type is None:
            # print(f"Warning: Schema missing type '{s_label}' (and 'Others') for entity '{s_name}'. Skipping.")
            return

        record["category"] = s_label

        for prop_name, prop_value in properties.items():
            if prop_value == "NAN":
                tmp_properties.pop(prop_name)
                continue
            if prop_name in spg_type.properties:
                from knext.schema.model.property import Property

                prop: Property = spg_type.properties.get(prop_name)
                o_label = prop.object_type_name_en
                if o_label not in BASIC_TYPES:
                    if isinstance(prop_value, str):
                        prop_value = [prop_value]
                    for o_name in prop_value:
                        sub_graph.add_node(id=o_name, name=o_name, label=o_label)
                        if "relatedQuery" in prop_name:
                            sub_graph.add_edge(
                                s_id=o_name,
                                s_label=o_label,
                                p=prop_name.replace("relatedQuery", "relatedTo"),
                                o_id=s_id,
                                o_label=s_label,
                            )
                        else:
                            sub_graph.add_edge(
                                s_id=s_id,
                                s_label=s_label,
                                p=prop_name,
                                o_id=o_name,
                                o_label=o_label,
                            )
                    tmp_properties.pop(prop_name)
        
        # 确保 name 属性始终保留（防止被意外移除）
        if "name" not in tmp_properties and s_name:
            tmp_properties["name"] = s_name
        
        record["properties"] = tmp_properties
        
        sub_graph.add_node(
            id=s_id, name=s_name, label=s_label, properties=record["properties"]
        )

    def _invoke(self, input: Input, **kwargs) -> List[Output]:
        """
        Overridden _invoke to pass section_id to assemble_knowledge_unit
        """
        title = input.name
        passage = title.split("_split_")[0] + "\n" + input.content
        out = []
        
        # 1. NER
        entities = self.named_entity_recognition(passage)
        
        # 2. Base SubGraph Assembly (Chunk + Entities)
        sub_graph, entities = self.assemble_sub_graph_with_spg_records(entities)
        
        # 3. Knowledge Unit Extraction
        filtered_entities = [
            {k: v for k, v in ent.items() if k in ["name", "category"]}
            for ent in entities
        ]
        chapter_title = input.kwargs.get("chapter_title", "")
        section_title = input.kwargs.get("section_title", "")
        candidate_points = _candidate_points_for_context(chapter_title, section_title)
        local_kn_prompt = copy.deepcopy(self.kn_prompt)
        local_kn_prompt.template_zh = _build_kn_template(
            candidate_points, chapter_title, section_title
        )
        knowledge_unit_entities = self.llm.invoke(
            {"input": passage, "named_entities": filtered_entities},
            local_kn_prompt,
            with_except=False,
            with_json_parse=False,
        )

        extracted_names = _parse_ku_name_list(knowledge_unit_entities)
        # Merge forced KUs (from headings/body) + extracted KUs, and cap.
        forced_names = _extract_forced_kus_from_text(passage, candidate_points)
        merged: List[str] = []
        seen = set()
        for n in forced_names + extracted_names:
            if not isinstance(n, str):
                continue
            nn = n.strip()
            if not nn or nn in seen:
                continue
            seen.add(nn)
            merged.append(nn)
            if len(merged) >= MAX_KU_PER_CHUNK:
                break
        knowledge_unit_entities = merged
        triples = self.triples_extraction(passage, filtered_entities)

        # 4. Extract section_id from input chunk kwargs
        section_id = input.kwargs.get("section_id", "Global")

        # 5. Assemble KUs (global-dedup) + build KU<->Chunk edges here
        knowledge_unit_nodes = self.assemble_knowledge_unit(
            sub_graph,
            entities,
            knowledge_unit_entities,
            triples,
            section_id=section_id,
            chunk_id=input.id,
            candidate_points=candidate_points,
        )

        # 5.5 SubKnowledgeUnit extraction (free). If core KUs exist, prefer parenting under them.
        if ENABLE_SUB_KU and input.id:
            def _post_filter_sub_names(names: List[str]) -> List[str]:
                names = self._filter_sub_ku_names(names)
                names = self._filter_sub_ku_not_equal_knowledge_units(names, knowledge_unit_nodes or [])
                names = self._dedupe_sub_kus_by_containment(names)
                names = self._dedupe_sub_kus_by_soft_key(names)
                return names

            sub_prompt = copy.deepcopy(self.kn_prompt)
            sub_prompt.template_zh = self._build_sub_ku_prompt()
            core_ku_names = []
            for ku in (knowledge_unit_nodes or []):
                n = (ku.get("content") or "").strip()
                if n:
                    core_ku_names.append(n)
            core_ku_text = json.dumps(core_ku_names, ensure_ascii=False)
            sub_raw = self.llm.invoke(
                {"input": passage, "core_knowledge_units": core_ku_text},
                sub_prompt,
                with_except=False,
                with_json_parse=False,
            )

            sub_names = _parse_llm_string_list(sub_raw, allow_dict_keys_as_items=True)

            # Deterministic boost: include subsection/item titles from this chunk.
            if ENABLE_SUB_KU_TITLES:
                title_subkus = self._extract_subku_titles_from_chunk(passage)
                if title_subkus:
                    sub_names = title_subkus + (sub_names or [])

            # Quality filter: drop generic single terms / sentence-like snippets.
            sub_names = _post_filter_sub_names(sub_names)

            # Top-up once if too few; ask for additional distinct items excluding existing.
            if len(sub_names) < MIN_SUB_KU_PER_CHUNK and SUB_KU_TOPUP_ROUNDS > 0:
                topup_prompt = copy.deepcopy(self.kn_prompt)
                topup_prompt.template_zh = f"""你是一名医学知识点抽取专家，目标是为知识图谱抽取子知识点（SubKnowledgeUnit）。

任务：基于给定 chunk 内容，在"不重复、不近义改写"的前提下，补充更多子知识点。

硬性要求：
1) 只从 chunk 内容抽取，不要引入外部知识。
2) 输出必须与【已抽取列表】中的任何一项都明显不同；近义/高度相似/包含关系强的视为重复，禁止输出。
3) 输出最多 {MAX_SUB_KU_PER_CHUNK} 个；只输出严格 JSON 数组（list）。
4) 不要输出过于泛化的单词（如仅一个通用概念名称）。
5) 不要输出句子型短句（例如包含：见于/多见于/发生于/表现为/可见/导致/因此/包括，或含逗号句号冒号等）。如果 chunk 中出现这类表达，请提炼为可作为节点的名词短语。
6) 不要输出与 core_knowledge_units 或已抽取列表中任何一项"完全相同"的名称。
7) 不要输出彼此存在"包含关系"的子知识点。

【已抽取列表】（禁止重复/改写）：
$existing

【core_knowledge_units（可选）】：
$core_knowledge_units

chunk：
$input
"""
                existing_text = json.dumps(sub_names, ensure_ascii=False)

                topup_raw = self.llm.invoke(
                    {"input": passage, "core_knowledge_units": core_ku_text, "existing": existing_text},
                    topup_prompt,
                    with_except=False,
                    with_json_parse=False,
                )
                more = _parse_llm_string_list(topup_raw, allow_dict_keys_as_items=True)
                if more:
                    sub_names = sub_names + more

            # Filter again after top-up merge.
            sub_names = _post_filter_sub_names(sub_names)

            if sub_names:
                self.assemble_sub_knowledge_unit(
                    sub_graph=sub_graph,
                    sub_names=sub_names,
                    core_kus=(knowledge_unit_nodes or []),
                    section_id=section_id,
                    chunk_id=input.id,
                )

        # 6. AtomicQuery extraction (optional) - TEMPORARILY DISABLED
        # 说明：原子问当前不需要，先通过 ENABLE_ATOMIC_QUERY 统一关闭。
        if ENABLE_ATOMIC_QUERY and input.id:
            aq_prompt = copy.deepcopy(self.kn_prompt)
            aq_prompt.template_zh = f"""你是一名医学知识检索问题生成助手。

任务：基于下面给出的 chunk 内容，生成最多 {MAX_ATOMIC_QUERY_PER_CHUNK} 个"原子问"（AtomicQuery）。

要求：
1) 问题必须能直接从该 chunk 内容中得到答案（不要跨章节、不要引入外部知识）。
2) 问题要具体、短、可检索（不要太泛）。
3) 只输出严格 JSON 数组（list），元素是问题字符串；不要输出任何解释文字。

示例：
["某概念的定义是什么？", "某机制有哪些常见类型？"]

### 输入：
$input
"""

            atomic_raw = self.llm.invoke(
                {"input": passage},
                aq_prompt,
                with_except=False,
                with_json_parse=False,
            )

            if DEBUG_ATOMIC_QUERY:
                print(f"[AtomicQuery] raw for {input.id}: {repr(atomic_raw)[:500]}")

            atomic_questions = _parse_llm_string_list(
                atomic_raw,
                dict_list_keys=("questions", "原子问", "atomic_queries", "data", "result"),
                question_first=True,
            )[:MAX_ATOMIC_QUERY_PER_CHUNK]

            # Deterministic fallback: if model didn't return usable JSON, generate minimal queries from KUs
            if not atomic_questions and knowledge_unit_nodes:
                for ku in knowledge_unit_nodes:
                    ku_name = (ku.get("content") or "").strip()
                    if not ku_name:
                        continue
                    atomic_questions.append(f"{ku_name}是什么？")
                    if len(atomic_questions) >= MAX_ATOMIC_QUERY_PER_CHUNK:
                        break
            for q in atomic_questions:
                aq_id = generate_hash_id(q)
                # create AtomicQuery node
                self.assemble_sub_graph_with_spg_properties(
                    sub_graph,
                    aq_id,
                    q,
                    "AtomicQuery",
                    {"name": q, "title": q, "description": q},
                )
                sub_graph.add_edge(
                    s_id=aq_id,
                    s_label="AtomicQuery",
                    p="sourceChunk",
                    o_id=input.id,
                    o_label="Chunk",
                )

                # Link AtomicQuery to all KUs extracted for this chunk
                for ku in (knowledge_unit_nodes or []):
                    ku_id = ku.get("name")
                    if not ku_id:
                        continue
                    sub_graph.add_edge(
                        s_id=aq_id,
                        s_label="AtomicQuery",
                        p="relatedTo",
                        o_id=ku_id,
                        o_label="KnowledgeUnit",
                    )

        # Keep chunk linking for extracted entities only.
        # KnowledgeUnit -> Chunk is linked explicitly via `sourceChunk` in assemble_knowledge_unit.
        self.assemble_sub_graph(sub_graph, input, entities, triples)
        out.append(sub_graph)
        return out

    def assemble_knowledge_unit(
        self,
        sub_graph: SubGraph,
        source_entities: List[dict],
        input_knowledge_units: dict, # Dict[str, Dict]
        triples: List[list],
        section_id: str = "Global",
        chunk_id: str = None,
        candidate_points: List[str] = None,
    ):
        knowledge_unit_nodes = []
        # Accept both dict and list (after parsing); normalize to dict[str, dict]
        if isinstance(input_knowledge_units, list):
            normalized = {}
            for item in input_knowledge_units:
                if isinstance(item, str) and item.strip():
                    normalized[item.strip()] = {}
                elif isinstance(item, dict):
                    k = (
                        item.get("知识点名称")
                        or item.get("name")
                        or item.get("名称")
                        or item.get("title")
                    )
                    if k:
                        normalized[str(k)] = item
            input_knowledge_units = normalized

        # Ensure input_knowledge_units is a dict
        if not isinstance(input_knowledge_units, dict):
            return []
            
        knowledge_units = dict(input_knowledge_units)

        # Avoid duplicating KU nodes/edges within the same chunk
        seen_knowledge_ids = set()
        
        for knowledge_name, knowledge_value in knowledge_units.items():
            if not isinstance(knowledge_value, dict):
                continue
            
            # DEBUG: Print what LLM found
            # print(f"DEBUG: LLM Extracted Raw Name: {knowledge_name}")

            # Map to catalog knowledge point (enforces requirements file)
            mapped_name = self._map_to_allowed_knowledge_point(
                knowledge_name,
                candidate_points=candidate_points,
            )

            # Strict: if not in catalog (and not confidently mapped), drop it
            if not mapped_name:
                continue

            # Knowledge type normalization
            k_type = (
                knowledge_value.get("知识类型")
                or knowledge_value.get("knowledgetype")
                or knowledge_value.get("knowledgeType")
                or "概念"
            )

            # Keep properties minimal for stable graph + linking
            ku_props = {
                "knowledgeType": k_type,
                "content": mapped_name,
                "name": mapped_name,
                "ontology": knowledge_value.get("领域本体")
                or knowledge_value.get("ontology")
                or "病理学",
                "relatedQuery": knowledge_value.get("关联问")
                or knowledge_value.get("relatedQuery")
                or [],
            }

            # Global-dedup ID: only by mapped knowledge name
            knowledge_id = generate_hash_id(mapped_name)

            if knowledge_id in seen_knowledge_ids:
                continue
            seen_knowledge_ids.add(knowledge_id)
            
            self.assemble_sub_graph_with_spg_properties(
                sub_graph,
                knowledge_id,
                mapped_name,
                "KnowledgeUnit",
                ku_props,
            )
            knowledge_unit_nodes.append(
                {"name": knowledge_id, "category": "KnowledgeUnit", "content": mapped_name}
            )

            # Many-to-many: KU -> sourceChunk -> Chunk (always add if chunk_id present)
            if chunk_id:
                sub_graph.add_edge(
                    s_id=knowledge_id,
                    s_label="KnowledgeUnit",
                    p="sourceChunk",
                    o_id=chunk_id,
                    o_label="Chunk",
                )

                # Also add the forward edge for easier navigation: Chunk -> hasKnowledgeUnit -> KU
                sub_graph.add_edge(
                    s_id=chunk_id,
                    s_label="Chunk",
                    p="hasKnowledgeUnit",
                    o_id=knowledge_id,
                    o_label="KnowledgeUnit",
                )

            # Optional: Section -> hasKnowledgeUnit -> KU for navigation
            if section_id and section_id != "Global":
                sub_graph.add_edge(
                    s_id=section_id,
                    s_label="Section",
                    p="hasKnowledgeUnit",
                    o_id=knowledge_id,
                    o_label="KnowledgeUnit",
                )
            
            # No core entities processing for KnowledgeUnit anymore as requested.
            # Just the node itself acting as a tag/anchor.

        return knowledge_unit_nodes

extractor = FixKnowledgeUnitExtractor(
    llm=llm,
    ner_prompt=ner_prompt,
    triple_prompt=None,
    kn_prompt=kn_prompt,
)

def parse_md_structure(file_path):
    """
    Parses Markdown to create Textbook -> Chapter -> Section -> Chunk structure.
    Returns: SubGraph containing all structural nodes/edges.
    """
    print(f"Parsing structure from {file_path}...")
    with open(file_path, 'r', encoding='utf_8') as f:
        lines = f.readlines()
    
    sub_graph = SubGraph([], [])
    
    # Textbook Node - 从文件名提取教材名称
    textbook_name = extract_textbook_name(file_path)
    book_id = generate_hash_id(f"Textbook_{textbook_name}")
    book_label = f"{NAMESPACE}.Textbook"
    sub_graph.add_node(id=book_id, name=textbook_name, label=book_label, properties={"name": textbook_name})
    
    current_chapter_node = None
    current_section_node = None
    
    # Tracking for linking
    prev_chapter_node = None
    prev_section_node = None
    
    buffer = []
    chunk_sequence = 0 
    
    def flush_chunk(section_id, text):
        nonlocal chunk_sequence
        if not text.strip():
            return
        
        from kag.builder.component.splitter.length_splitter import LengthSplitter
        from kag.builder.model.chunk import Chunk as KagChunk
        import re as regex
        
        chunk_size = 600  # chunk 最大长度
        chunk_overlap = 100
        min_chunk_size = 100  # 过短的段落合并到下一个
        
        # ===== Step 1: 按编号/标题拆分成逻辑段落 =====
        # 匹配：1. 2. 3. 或 （一）（二）或 (1) (2) 等开头
        paragraph_pattern = regex.compile(
            r'(?=\n\s*(?:\d+\.\s|[（(][一二三四五六七八九十\d]+[）)]\s*))'
        )
        
        # 先按编号拆分
        paragraphs = paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 如果没有编号，保持原样
        if len(paragraphs) <= 1:
            paragraphs = [text.strip()]
        
        # ===== Step 2: 合并过短的段落，切分过长的段落 =====
        splitter = LengthSplitter(split_length=chunk_size, window_length=chunk_overlap)
        
        final_chunks = []
        buffer_text = ""
        
        for para in paragraphs:
            combined = (buffer_text + "\n" + para).strip() if buffer_text else para
            
            if len(combined) < min_chunk_size:
                # 太短，继续积累
                buffer_text = combined
            elif len(combined) <= chunk_size:
                # 合适长度，直接作为一个 chunk
                final_chunks.append(combined)
                buffer_text = ""
            else:
                # 超长，使用 LengthSplitter 切分
                if buffer_text and len(buffer_text) >= min_chunk_size:
                    final_chunks.append(buffer_text)
                    buffer_text = ""
                
                # 对当前段落进行切分
                temp_chunk = KagChunk(
                    id=generate_hash_id(f"{section_id}_temp"),
                    name=f"temp_{section_id}",
                    content=para,
                )
                split_results = splitter.slide_window_chunk(temp_chunk, chunk_size, chunk_overlap)
                for sc in split_results:
                    final_chunks.append(sc.content)
        
        # 处理剩余的 buffer
        if buffer_text:
            final_chunks.append(buffer_text)
        
        # ===== Step 3: 清理并写入图数据库 =====
        for chunk_content in final_chunks:
            # 清理 chunk 内容中的特殊符号
            cleaned_content = clean_chunk_content(chunk_content)
            if not cleaned_content or len(cleaned_content) < 10:
                continue  # 跳过过短或空的 chunk
            
            # 生成唯一的 chunk ID，包含命名空间前缀以避免跨教材同名问题
            # 格式: namespace_hash，确保不同教材的相同内容生成不同的 ID
            content_hash = generate_hash_id(f"{section_id}_{chunk_sequence}")
            seg_id = f"{NAMESPACE}_{content_hash}"  # 包含命名空间前缀
            seg_label = f"{NAMESPACE}.Chunk"
            sub_graph.add_node(id=seg_id, name=seg_id, label=seg_label, properties={"content": cleaned_content, "sequence": chunk_sequence, "section_id": section_id})
            chunk_sequence += 1

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line: continue

        # Intro (绪论) as a dedicated Chapter + Section to capture正文内容
        # Supports "# 绪 论" or "# 绪论"
        intro_norm = line.replace(" ", "")
        if intro_norm.startswith("#绪论"):
            if current_section_node and buffer:
                flush_chunk(current_section_node["id"], "\n".join(buffer))
                buffer = []

            cid = "Chapter_Intro_v2"
            clabel = f"{NAMESPACE}.Chapter"
            intro_title = CATALOG_CHAPTER_TITLES.get("绪论", "绪论")
            current_chapter_node = {"id": cid, "label": clabel, "title": intro_title, "idx": "绪论"}
            sub_graph.add_node(id=cid, name=intro_title, label=clabel, properties={"name": intro_title, "title": intro_title})
            sub_graph.add_edge(s_id=book_id, s_label=book_label, p="hasChapters", o_id=cid, o_label=clabel)

            # Reset section tracking for intro
            prev_section_node = None
            sid = f"{cid}_Section_Intro_v2"
            slabel = f"{NAMESPACE}.Section"
            current_section_node = {"id": sid, "label": slabel}
            sub_graph.add_node(id=sid, name=intro_title, label=slabel, properties={"name": intro_title, "title": intro_title})
            sub_graph.add_edge(s_id=cid, s_label=clabel, p="hasSections", o_id=sid, o_label=slabel)

            continue
            
        # Chapter
        chapter_match = re.match(r'^#\s+第(.*?)章', line)
        if chapter_match:
            if current_section_node and buffer:
                flush_chunk(current_section_node["id"], "\n".join(buffer))
                buffer = []
            
            idx_str = chapter_match.group(1)
            rest = line[chapter_match.end():].strip()
            title = rest if rest else f"第{idx_str}章"
            if not rest:
                # Lookahead
                for j in range(i, min(i+5, len(lines))):
                    next_l = lines[j].strip()
                    if not next_l: continue
                    if next_l.startswith('# '):
                        title = next_l.strip('# ').strip()
                    break
            
            # Prefer catalog chapter display title if available (by chapter index)
            idx_key = (idx_str or "").strip()
            if idx_key and idx_key in CATALOG_CHAPTER_TITLES:
                title = CATALOG_CHAPTER_TITLES[idx_key]

            cid = f"Chapter_{idx_str}_v2"
            clabel = f"{NAMESPACE}.Chapter"
            current_chapter_node = {"id": cid, "label": clabel, "title": title, "idx": idx_str}
            sub_graph.add_node(id=cid, name=title, label=clabel, properties={"name": title, "title": title})
            
            # Textbook -> Chapter
            sub_graph.add_edge(s_id=book_id, s_label=book_label, p="hasChapters", o_id=cid, o_label=clabel)
            
            # Chapter Linking
            if prev_chapter_node:
                # nextChapter
                sub_graph.add_edge(s_id=prev_chapter_node["id"], s_label=clabel, p="nextChapter", o_id=cid, o_label=clabel)
            
            prev_chapter_node = current_chapter_node
            # Reset Section tracking
            current_section_node = None
            prev_section_node = None

            # Create an overview section to capture chapter intro text before the first explicit "第X节"
            sid0 = f"{cid}_Section_0_v2"
            slabel0 = f"{NAMESPACE}.Section"
            current_section_node = {"id": sid0, "label": slabel0}
            sub_graph.add_node(id=sid0, name="概述", label=slabel0, properties={"name": "概述", "title": "概述"})
            sub_graph.add_edge(s_id=cid, s_label=clabel, p="hasSections", o_id=sid0, o_label=slabel0)
            prev_section_node = current_section_node
            
            continue
            
        # Section
        # Updated Regex: Allow optional spaces around "第" and "节", and be more robust
        section_match = re.match(r'^#+\s*第\s*(\S+)\s*节\s*(.*)', line)
        if section_match:
            if current_section_node and buffer:
                flush_chunk(current_section_node["id"], "\n".join(buffer))
                buffer = []
            
            idx_str = section_match.group(1)
            title_text = section_match.group(2).strip() # Ensure no trailing spaces
            
            # If title is empty, maybe the line ends at "节"?
            if not title_text:
                title = f"第{idx_str}节"
            else:
                title = title_text # User requested only content, no prefix

            # Prefer catalog section display title if available (scoped by current chapter)
            chapter_idx = (current_chapter_node.get("idx") if current_chapter_node else "") or ""
            chapter_idx = str(chapter_idx).strip()
            sec_idx = str(idx_str or "").strip()
            if chapter_idx and sec_idx and (chapter_idx, sec_idx) in CATALOG_SECTION_TITLES:
                title = CATALOG_SECTION_TITLES[(chapter_idx, sec_idx)]

            sid = f"{current_chapter_node['id']}_Section_{idx_str}_v2" if current_chapter_node else f"Section_{idx_str}_v2"
            slabel = f"{NAMESPACE}.Section"
            current_section_node = {"id": sid, "label": slabel}
            
            sub_graph.add_node(id=sid, name=title, label=slabel, properties={"name": title, "title": title})
            
            if current_chapter_node:
                # Chapter -> Section (hasSections)
                sub_graph.add_edge(s_id=current_chapter_node["id"], s_label=current_chapter_node["label"], p="hasSections", o_id=sid, o_label=slabel)
            
            # Section Linking
            if prev_section_node:
                 sub_graph.add_edge(s_id=prev_section_node["id"], s_label=slabel, p="nextSection", o_id=sid, o_label=slabel)
            
            prev_section_node = current_section_node
            
            continue
            
        # Content
        if current_section_node:
            buffer.append(line)
            
    if current_section_node and buffer:
        flush_chunk(current_section_node["id"], "\n".join(buffer))
        
    return sub_graph


def add_medicine_labels_for_namespace(namespace: str):
    """
    给指定命名空间的节点自动添加 Medicine.* 标签
    
    在抽取完成后调用此函数，确保所有节点都能被 Medicine 命名空间的向量索引检索到。
    
    Args:
        namespace: 教材命名空间，如 "Pharmacology", "Pathology", "Biology"
    """
    if namespace == "Medicine":
        print(f"[标签同步] 跳过 Medicine 命名空间（无需添加）")
        return
    
    print(f"\n[标签同步] 正在为 {namespace} 命名空间添加 Medicine 标签...")
    
    # Neo4j 连接配置
    import os
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
    
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    except Exception as e:
        print(f"[标签同步] 连接 Neo4j 失败: {e}")
        return
    
    # 需要添加 Medicine 标签的节点类型
    node_types = [
        "Chunk", "AtomicQuery", "SubKnowledgeUnit", "KnowledgeUnit",
        "Summary", "Outline", "Chapter", "Section", "Textbook",
        "Table", "Concept", "Drug", "Disease", "Symptom", "Event"
    ]
    
    total_updated = 0
    
    with driver.session(database=neo4j_database) as session:
        for node_type in node_types:
            source_label = f"{namespace}.{node_type}"
            target_label = f"Medicine.{node_type}"
            
            try:
                # 添加标签
                result = session.run(f"""
                    MATCH (n:`{source_label}`)
                    WHERE NOT n:`{target_label}`
                    SET n:`{target_label}`
                    RETURN count(n) as updated
                """)
                updated = result.single()["updated"]
                
                if updated > 0:
                    total_updated += updated
                    print(f"  {source_label} -> {target_label}: {updated} 个节点")
            except Exception as e:
                # 可能标签不存在，跳过
                pass
    
        # ===== 清理 Medicine.Chunk 节点的 content 中的 ID 前缀 =====
        # 问题：ID 和实际内容之间是字面字符串 '\n' (反斜杠+n)，不是真正的换行符
        print("  正在清理 Chunk content 中的 ID 前缀...")
        try:
            cleanup_result = session.run(r"""
                MATCH (c:`Medicine.Chunk`) 
                WHERE c.content CONTAINS '\\n' 
                WITH c, split(c.content, '\\n') AS parts 
                WHERE size(parts[0]) >= 32
                SET c.content = substring(c.content, size(parts[0]) + 2) 
                RETURN count(c) as cleaned
            """)
            cleaned = cleanup_result.single()["cleaned"]
            if cleaned > 0:
                print(f"  清理了 {cleaned} 个 Chunk 的 content")
        except Exception as e:
            print(f"  清理失败: {e}")
        # ===== 清理结束 =====
    
    driver.close()
    
    print(f"[标签同步] 完成，共更新 {total_updated} 个节点")


# 缓存的 Neo4j driver（用于即时标签添加）
_label_driver = None

def _add_medicine_labels_for_nodes(nodes):
    """
    为一批节点立即添加 Medicine.* 标签
    
    在每个 Chunk 抽取完成后调用，实现边抽取边添加标签
    """
    global _label_driver
    import os
    
    if not nodes:
        return
    
    # 初始化 driver
    if _label_driver is None:
        try:
            from neo4j import GraphDatabase
            neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
            neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
            _label_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        except:
            return
    
    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
    
    with _label_driver.session(database=neo4j_database) as session:
        for node in nodes:
            if not hasattr(node, 'label') or not node.label:
                continue
            
            # 解析原始标签（如 Pharmacology.Chunk）
            if '.' in node.label:
                ns, node_type = node.label.split('.', 1)
                if ns != "Medicine":
                    target_label = f"Medicine.{node_type}"
                    try:
                        session.run(f"""
                            MATCH (n)
                            WHERE n.id = $node_id AND n:`{node.label}`
                            SET n:`{target_label}`
                        """, node_id=node.id)
                        
                        # 立即清理 Chunk 节点的 content 中的 ID 前缀
                        if node_type == "Chunk":
                            session.run(r"""
                                MATCH (c)
                                WHERE c.id = $node_id AND c.content CONTAINS '\\n'
                                WITH c, split(c.content, '\\n') AS parts
                                WHERE size(parts[0]) >= 32
                                SET c.content = substring(c.content, size(parts[0]) + 2)
                            """, node_id=node.id)
                    except:
                        pass


def run_extraction(structural_graph: SubGraph):
    """
    Iterates over chunks in structural_graph, calls Extractor, and writes results.
    """
    print("Starting Knowledge Extraction & Import...")
    
    # 1. Write Structure First
    print("Writing Structural Skeleton...")
    client.write_graph(sub_graph=structural_graph.to_dict(), operation="UPSERT", lead_to_builder=False)
    
    # 2. Extract from Chunks
    chunk_nodes = [n for n in structural_graph.nodes if n.label.endswith("Chunk")]
    total = len(chunk_nodes)
    
    # TEST_MODE: 限制处理的 chunk 数量
    if TEST_MODE:
        chunk_nodes = chunk_nodes[:TEST_MODE_CHUNK_LIMIT]
        print(f"[TEST MODE] 只处理 {len(chunk_nodes)}/{total} 个 chunks")
    
    print(f"Found {len(chunk_nodes)} chunks to process.")
    
    # 初始化 Extractor (在函数内创建，因为需要 llm 等已初始化)
    global extractor
    extractor = FixKnowledgeUnitExtractor(
        llm=llm,
        ner_prompt=ner_prompt,
        triple_prompt=triple_prompt,
        kn_prompt=kn_prompt,
    )
    print("Extractor initialized.")

    # Build Section/Chapter lookup for scoped knowledge-point candidates
    section_id_to_title = {
        n.id: (n.properties.get("title") or n.name)
        for n in structural_graph.nodes
        if n.label.endswith("Section")
    }
    chapter_id_to_title = {
        n.id: (n.properties.get("title") or n.name)
        for n in structural_graph.nodes
        if n.label.endswith("Chapter")
    }
    # Map Section -> Chapter using structural edges (Chapter -[hasSections]-> Section)
    section_id_to_chapter_id = {
        e.to_id: e.from_id
        for e in structural_graph.edges
        if e.label == "hasSections" and e.from_type.endswith("Chapter") and e.to_type.endswith("Section")
    }
    
    from concurrent.futures import ThreadPoolExecutor, as_completed

    processed_count = 0
    
    # Collection for Exercise Linking
    collected_kus = [] # List of KU dicts
    
    def process_single_chunk(node, idx, total_count):
        print(f"Processing Chunk {idx+1}/{total_count}: {node.id}")
        local_kus = []
        try:
            content = node.properties.get("content", "")
            
            # ===== 清理 content 中可能被 Server 添加的 ID 前缀 =====
            # 检测格式: "hash_id\n实际内容" 
            if '\n' in content and len(content) > 32:
                first_line_end = content.index('\n')
                first_line = content[:first_line_end]
                # 如果第一行看起来像 hash ID（只包含十六进制字符）
                if len(first_line) >= 32 and all(c in '0123456789abcdef' for c in first_line[:32]):
                    content = content[first_line_end + 1:]  # 移除 ID 前缀
            # ===== 清理结束 =====
            
            if len(content) < 10: 
                return None
            
            # --- PASS SECTION ID VIA KWARGS ---
            section_id = node.properties.get("section_id", "Global")
            section_title = section_id_to_title.get(section_id, "")
            chapter_id = section_id_to_chapter_id.get(section_id, "")
            chapter_title = chapter_id_to_title.get(chapter_id, "")
            input_obj = Chunk(
                id=node.id,
                name=node.name,
                content=content,
                kwargs={
                    "section_id": section_id,
                    "section_title": section_title,
                    "chapter_title": chapter_title,
                },
            )
            # ----------------------------------

            results = extractor.invoke(input_obj)
            
            
            for res in results:
                # 'res' is BuilderComponentData, accessing the actual SubGraph via 'res.data'
                res_sg = res.data
                
                for n in res_sg.nodes:
                    if not n.label.startswith(NAMESPACE) and "." not in n.label:
                        n.label = f"{NAMESPACE}.{n.label}"
                    
                    # Schema Optimization: Map Medicine -> Concept
                    type_suffix = n.label.split(".")[-1]
                    if type_suffix == "Medicine":
                        n.label = f"{NAMESPACE}.Concept"
                        type_suffix = "Concept"

                    # --- TRACEABILITY FEATURE START ---
                    # Link every extracted Entity back to the Chunk
                    valid_mentions_types = ["Concept", "Disease", "Symptom", "Drug"]
                    
                    if type_suffix in valid_mentions_types:
                        res_sg.add_edge(
                            s_id=input_obj.id, s_label=f"{NAMESPACE}.Chunk",
                            p="mentions",
                            o_id=n.id, o_label=n.label
                        )
                    
                    # Special handling for KnowledgeUnit
                    if type_suffix == "KnowledgeUnit":
                         # Collect KU for linking
                         local_kus.append({
                             "name": n.id,
                             "category": "KnowledgeUnit",
                             "content": n.properties.get("content", "")
                         })
                    
                    # Special handling for SubKnowledgeUnit
                    if type_suffix == "SubKnowledgeUnit":
                         local_kus.append({
                             "name": n.id,
                             "category": "SubKnowledgeUnit",
                             "content": n.properties.get("content", "")
                         })

                    # --- TRACEABILITY FEATURE END ---

                for e in res_sg.edges:
                    if not e.from_type.startswith(NAMESPACE) and "." not in e.from_type:
                        e.from_type = f"{NAMESPACE}.{e.from_type}"
                    if not e.to_type.startswith(NAMESPACE) and "." not in e.to_type:
                        e.to_type = f"{NAMESPACE}.{e.to_type}"

                # Generate vector embeddings for all nodes before writing
                try:
                    from kag.interface.builder.base import BuilderComponentData
                    input_data = BuilderComponentData(data=res_sg, hash_key=input_obj.id)
                    vectorized_results = batch_vectorizer.invoke(input_data)
                    if vectorized_results:
                        # BatchVectorizer returns list of BuilderComponentData
                        res_sg = vectorized_results[0].data
                except Exception as vec_err:
                    print(f"Warning: vectorization failed for {node.id}: {vec_err}")

                graph_data = res_sg.to_dict()
                client.write_graph(sub_graph=graph_data, operation="UPSERT", lead_to_builder=False)
                
                # 立即为新节点添加 Medicine 标签（边抽取边添加）
                if NAMESPACE != "Medicine":
                    try:
                        _add_medicine_labels_for_nodes(res_sg.nodes)
                    except Exception as label_err:
                        pass  # 静默处理，不影响主流程
            
            return local_kus
        except Exception as e:
            print(f"Error processing {node.id}: {e}")
            traceback.print_exc()
            return None

    # Parallel Execution
    max_workers = 5
    print(f"Starting parallel extraction with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_chunk, node, idx, total): node for idx, node in enumerate(chunk_nodes)}
        
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                processed_count += 1
                collected_kus.extend(res)
                # 分别统计 KnowledgeUnit 和 SubKnowledgeUnit（使用唯一名称，避免重复计数）
                unique_ku_names = set(ku.get("name", "") for ku in collected_kus if ku.get("name") and ku.get("category") == "KnowledgeUnit")
                unique_subku_names = set(ku.get("name", "") for ku in collected_kus if ku.get("name") and ku.get("category") == "SubKnowledgeUnit")
                print(f"[KnowledgeUnit] 已抽取 {len(unique_ku_names)} 个知识点")
                print(f"[SubKnowledgeUnit] 已抽取 {len(unique_subku_names)} 个子知识点")
                
    print(f"Extraction complete. Processed {processed_count}/{total} chunks.")

if __name__ == "__main__":
    args = parse_args()
    
    # 确定配置文件路径
    if args.config:
        config_path = args.config
    else:
        # 默认使用脚本同级目录的上一级 kag_config.yaml
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        config_path = os.path.join(project_root, "kag_config.yaml")
    
    # 模式判断：清单模式 vs 单文件模式
    if args.manifest:
        # ========== 清单驱动模式 ==========
        manifest_path = args.manifest
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.abspath(manifest_path)
        
        manifest = load_textbooks_manifest(manifest_path)
        manifest_dir = os.path.dirname(manifest_path)
        
        # 获取共享配置路径 (相对于清单文件)
        shared_config = manifest.get('shared_config')
        if shared_config and not args.config:
            config_path = os.path.join(manifest_dir, shared_config)
        
        default_host = manifest.get('default_host_addr', 'http://127.0.0.1:8887')
        default_namespace = manifest.get('default_namespace', 'Medici')
        
        books = get_textbooks_to_process(manifest, args.books)
        
        if not books:
            print("没有找到要处理的教材。请检查清单文件或 --books 参数。")
            sys.exit(1)
        
        print(f"=== 多教材知识抽取模式 ===")
        print(f"清单文件: {manifest_path}")
        print(f"待处理教材: {[b.get('name', b.get('id')) for b in books]}")
        print()
        
        for book in books:
            book_id = book.get('id', 'unknown')
            # 解析内容文件路径 (先解析，以便提取教材名称)
            content_file = book.get('content_file')
            if content_file and not os.path.isabs(content_file):
                content_file = os.path.abspath(os.path.join(manifest_dir, content_file))
            
            # 使用 manifest 中的 name，如果没有则从文件名提取
            book_name = book.get('name') or extract_textbook_name(content_file)
            namespace = args.namespace or book.get('namespace', default_namespace)
            host = args.host or default_host
            
            catalog_file = book.get('catalog_file')
            if catalog_file and not os.path.isabs(catalog_file):
                catalog_file = os.path.abspath(os.path.join(manifest_dir, catalog_file))
            
            print(f"\n{'='*60}")
            print(f"处理教材: {book_name}")
            print(f"{'='*60}")
            
            if not content_file or not os.path.exists(content_file):
                print(f"跳过: 内容文件不存在 - {content_file}")
                continue
            
            # 初始化环境 (所有教材都写入 Medicine 项目)
            initialize_environment(
                config_path=config_path,
                host_addr=host,
                namespace_override=namespace,
                content_file=content_file,
                catalog_file=catalog_file,
            )
            
            # 运行抽取
            sg = parse_md_structure(content_file)
            run_extraction(sg)
            
            # 自动添加 Medicine 标签（确保向量索引能检索到）
            add_medicine_labels_for_namespace(namespace)
            
            print(f"教材 [{book_name}] 处理完成。")
        
        print(f"\n=== 全部教材处理完成 ===")
    
    elif args.content:
        # ========== 单文件模式 ==========
        content_file = args.content
        if not os.path.isabs(content_file):
            content_file = os.path.abspath(content_file)
        
        catalog_file = args.catalog
        if catalog_file and not os.path.isabs(catalog_file):
            catalog_file = os.path.abspath(catalog_file)
        
        if not os.path.exists(content_file):
            print(f"内容文件不存在: {content_file}")
            sys.exit(1)
        
        # 确定 namespace：优先使用 --namespace，其次在 --auto-namespace 模式下自动提取
        namespace = args.namespace
        if not namespace and getattr(args, 'auto_namespace', False):
            # 从文件名自动提取中文教材名作为 namespace
            extracted_name = extract_textbook_name(content_file)
            # 只保留中文部分（去除版本号等）
            match = re.match(r'([\u4e00-\u9fa5]+)', extracted_name)
            if match:
                namespace = match.group(1)
            else:
                namespace = extracted_name
            print(f"[auto-namespace] 从文件名自动提取 namespace: {namespace}")
        
        print(f"=== 单文件模式 ===")
        print(f"教材文件: {content_file}")
        print(f"Namespace: {namespace or 'Medicine (默认)'}")
        
        initialize_environment(
            config_path=config_path,
            host_addr=args.host,
            namespace_override=namespace,
            content_file=content_file,
            catalog_file=catalog_file,
        )
        
        sg = parse_md_structure(content_file)
        run_extraction(sg)
        
        # 自动添加 Medicine 标签（确保向量索引能检索到）
        if namespace and namespace != "Medicine":
            add_medicine_labels_for_namespace(namespace)
        
        print("处理完成。")
    
    else:
        # ========== 默认模式：自动查找并使用 textbooks.yaml ==========
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        default_manifest = os.path.join(project_root, "textbooks.yaml")
        
        if os.path.exists(default_manifest):
            print(f"未指定参数，自动使用默认清单: {default_manifest}")
            manifest = load_textbooks_manifest(default_manifest)
            manifest_dir = os.path.dirname(default_manifest)
            
            # 获取共享配置路径
            shared_config = manifest.get('shared_config')
            if shared_config:
                config_path = os.path.join(manifest_dir, shared_config)
            
            default_host = manifest.get('default_host_addr', 'http://127.0.0.1:8887')
            default_namespace = manifest.get('default_namespace', 'Medici')
            
            books = get_textbooks_to_process(manifest, None)
            
            if not books:
                print("没有找到要处理的教材。请检查 textbooks.yaml 中的 enabled 配置。")
                sys.exit(1)
            
            print(f"=== 多教材知识抽取模式 ===")
            print(f"清单文件: {default_manifest}")
            print(f"待处理教材: {[b.get('name', b.get('id')) for b in books]}")
            print()
            
            for book in books:
                book_id = book.get('id', 'unknown')
                # 解析相对路径 (先解析，以便提取教材名称)
                content_file = book.get('content_file')
                if content_file and not os.path.isabs(content_file):
                    content_file = os.path.abspath(os.path.join(manifest_dir, content_file))
                
                # 使用 manifest 中的 name，如果没有则从文件名提取
                book_name = book.get('name') or extract_textbook_name(content_file)
                namespace = book.get('namespace', default_namespace)
                host = default_host
                
                catalog_file = book.get('catalog_file')
                if catalog_file and not os.path.isabs(catalog_file):
                    catalog_file = os.path.abspath(os.path.join(manifest_dir, catalog_file))
                
                print(f"\n{'='*60}")
                print(f"处理教材: {book_name}")
                print(f"{'='*60}")
                
                if not content_file or not os.path.exists(content_file):
                    print(f"跳过: 内容文件不存在 - {content_file}")
                    continue
                
                initialize_environment(
                    config_path=config_path,
                    host_addr=host,
                    namespace_override=namespace,
                    content_file=content_file,
                    catalog_file=catalog_file,
                )
                
                sg = parse_md_structure(content_file)
                run_extraction(sg)
                
                # 自动添加 Medicine 标签（确保向量索引能检索到）
                add_medicine_labels_for_namespace(namespace)
                
                # 自动修复 LaTeX 公式的双斜杠问题
                fix_latex_backslashes_in_neo4j(namespace)
                
                # 自动为 Chunk 节点生成向量
                vectorize_chunks_for_namespace(namespace)
                
                print(f"教材 [{book_name}] 处理完成。")
            
            print(f"\n=== 全部教材处理完成 ===")
        else:
            print("用法示例:")
            print("  清单模式: python indexer.py --manifest ../textbooks.yaml")
            print("  单文件模式: python indexer.py --content ./data/教材.md --catalog ./data/知识点.md")
            print("  指定教材: python indexer.py --manifest ../textbooks.yaml --books pharmacology,pathology")
            print("\n运行 python indexer.py --help 查看所有参数。")

