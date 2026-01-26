# -*- coding: utf-8 -*-
"""
医学领域专用的回答生成 Prompt

针对医学专业领域优化，强调：
1. 准确性和专业性
2. 结构化回答
3. 引用规范
4. 避免幻觉
"""

from typing import List
import logging

from kag.common.utils import get_now
from kag.interface import PromptABC

logger = logging.getLogger(__name__)


# ============================================================================
# 教材关键词映射表（用于自动识别问题应该检索哪个教材）
# ============================================================================
TEXTBOOK_KEYWORDS = {
    "Pharmacology": [
        # 通用术语
        "药物", "药品", "剂量", "给药", "药效", "副作用", "受体", "激动剂", 
        "拮抗剂", "代谢", "排泄", "半衰期", "生物利用度", "首过效应",
        "药理", "药代动力学", "药效学", "毒性", "耐药", "抗生素",
        "镇痛", "麻醉", "抗炎", "降压", "利尿", "抗凝", "用药",
        # 常见药名
        "阿司匹林", "布洛芬", "吗啡", "阿托品", "肾上腺素", "去甲肾上腺素",
        "乙酰胆碱", "多巴胺", "地西泮", "苯巴比妥", "华法林", "肝素",
        "青霉素", "头孢", "红霉素", "庆大霉素", "利福平", "异烟肼",
        "硝酸甘油", "普萘洛尔", "美托洛尔", "氨氯地平", "卡托普利",
        "胰岛素", "二甲双胍", "格列", "他汀", "地塞米松", "泼尼松"
    ],
    "Pathology": [
        # 通用术语
        "病变", "坏死", "炎症", "肿瘤", "水肿", "充血", "出血", "血栓", 
        "梗死", "萎缩", "增生", "化生", "发育异常", "细胞损伤",
        "病理", "癌", "瘤", "纤维化", "变性", "凋亡", "免疫",
        "感染", "溃疡", "肉芽肿", "栓塞", "损伤", "修复", "再生",
        # 病理过程
        "适应", "老化", "缺血", "缺氧", "中毒", "过敏", "自身免疫",
        "动脉粥样硬化", "血管炎", "肺炎", "肝硬化", "肾炎"
    ],
    "physiology": [
        # 细胞与基本功能
        "细胞膜", "细胞器", "离子通道", "载体", "主动转运", "被动转运",
        "静息电位", "动作电位", "兴奋性", "阈值", "不应期", "去极化", "复极化",
        # 血液
        "红细胞", "白细胞", "血小板", "血红蛋白", "血型", "血液凝固", "纤溶",
        "造血", "血浆", "渗透压", "血沉",
        # 循环系统
        "心脏", "心肌", "心电图", "心动周期", "心输出量", "血压", "心率",
        "心室", "心房", "瓣膜", "冠状动脉", "微循环", "淋巴循环",
        # 呼吸系统
        "呼吸", "肺通气", "肺换气", "呼吸运动", "呼吸中枢", "氧气", "二氧化碳",
        "血氧", "血红蛋白饱和度", "呼吸调节", "肺活量", "潮气量",
        # 消化系统
        "消化", "吸收", "胃液", "胰液", "胆汁", "肠液", "蠕动",
        "唾液", "食管", "胃", "小肠", "大肠", "肝脏功能",
        # 泌尿系统
        "肾脏", "肾小球", "肾小管", "尿液", "滤过", "重吸收", "分泌",
        "浓缩", "稀释", "酸碱平衡", "电解质", "水盐平衡",
        # 神经系统
        "神经元", "突触", "神经递质", "反射", "反射弧", "中枢神经",
        "周围神经", "自主神经", "交感", "副交感", "感觉", "运动神经",
        "大脑皮层", "脑干", "脊髓", "神经冲动", "传导",
        # 内分泌
        "激素", "内分泌", "垂体", "甲状腺", "肾上腺皮质", "胰岛",
        "反馈调节", "靶器官", "腺体",
        # 生殖与发育
        "生殖", "精子", "卵子", "受精", "妊娠", "泌乳",
        # 通用生理术语
        "生理", "调节", "稳态", "内环境", "反馈", "机制", "功能"
    ]
}


def classify_textbook_by_keywords(query: str) -> dict:
    """
    基于关键词匹配的教材分类（快速但可能不够准确）
    """
    scores = {}
    
    for textbook, keywords in TEXTBOOK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query)
        if score > 0:
            scores[textbook] = score
    
    if not scores:
        return {
            "primary": None,
            "secondary": list(TEXTBOOK_KEYWORDS.keys()),
            "confidence": 0.0,
            "method": "keywords"
        }
    
    sorted_textbooks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_textbooks[0][0]
    primary_score = sorted_textbooks[0][1]
    
    total_score = sum(scores.values())
    confidence = primary_score / total_score if total_score > 0 else 0
    
    secondary = [t for t, _ in sorted_textbooks[1:]] if len(sorted_textbooks) > 1 else []
    
    return {
        "primary": primary,
        "secondary": secondary,
        "confidence": round(confidence, 2),
        "scores": scores,
        "method": "keywords"
    }


def classify_textbook_by_llm(query: str) -> dict:
    """
    使用 LLM 判断问题属于哪个教材（更准确但较慢）
    如果检测到非医学问题，返回 primary="NON_MEDICAL"
    
    Returns:
        {"primary": "physiology", "secondary": ["Pathology"], "confidence": 0.9, "method": "llm"}
        或 {"primary": "NON_MEDICAL", "confidence": 1.0, "method": "llm"} 表示非医学问题
    """
    import requests
    
    # LLM 配置
    LLM_API_KEY = "sk-xmubwjeopdksjenuqmsjkvvipldehyacnmkvghpyoekwqdzz"
    LLM_BASE_URL = "https://api.siliconflow.cn/v1"
    LLM_MODEL = "Qwen/Qwen3-8B"
    
    # 构建分类 Prompt（包含非医学检测）
    textbook_list = "\n".join([
        "- Pharmacology: 药理学（药物作用机制、药效、副作用、给药方式等）",
        "- Pathology: 病理学（疾病机制、病变、炎症、肿瘤、坏死等）",
        "- physiology: 生理学（人体正常功能、器官系统、细胞生理、神经调节等）",
        "- NON_MEDICAL: 非医学问题（政治、历史、娱乐、科技、生活等与医学无关的问题）"
    ])
    
    prompt = f"""请判断以下问题的类别。

可选类别：
{textbook_list}

用户问题：{query}

判断规则：
1. 如果问题明显与医学、生理、病理、药理无关（如政治人物、历史事件、娱乐明星、科技产品等），请回答 NON_MEDICAL
2. 如果是医学相关问题，请回答最相关的教材名称（Pharmacology、Pathology 或 physiology）

只回答一个分类名称，不要解释。"""

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
            timeout=10
        )
        response.raise_for_status()
        
        answer = response.json()["choices"][0]["message"]["content"].strip()
        
        # 检测非医学问题
        if "NON_MEDICAL" in answer.upper() or "非医学" in answer:
            logger.info("检测到非医学问题: %s", query[:50])
            return {
                "primary": "NON_MEDICAL",
                "secondary": [],
                "confidence": 1.0,
                "method": "llm",
                "llm_raw_answer": answer
            }
        
        # 解析 LLM 回答
        valid_textbooks = ["Pharmacology", "Pathology", "physiology"]
        detected = []
        
        for tb in valid_textbooks:
            if tb.lower() in answer.lower():
                detected.append(tb)
        
        if not detected:
            logger.warning("LLM 分类结果无法解析: %s", answer)
            return None
        
        result = {
            "primary": detected[0],
            "secondary": detected[1:] if len(detected) > 1 else [],
            "confidence": 0.9 if len(detected) == 1 else 0.7,
            "method": "llm",
            "llm_raw_answer": answer
        }
        
        logger.info("LLM 教材分类结果: %s", result)
        return result
        
    except Exception as e:
        logger.warning("LLM 分类失败: %s", e)
        return None


def classify_textbook(query: str, use_llm: bool = True) -> dict:
    """
    智能分类问题所属教材
    
    策略：
    0. 先检测问题是否明确提到多个教材名称（如"病理学和药理学"）
    1. 先用关键词快速匹配
    2. 如果关键词匹配置信度低（<0.7）或无匹配，使用 LLM 判断
    3. 如果 LLM 失败，回退到关键词结果
    
    Args:
        query: 用户问题
        use_llm: 是否启用 LLM 分类（默认启用）
        
    Returns:
        {"primary": "physiology", "secondary": [], "confidence": 0.9}
    """
    # 步骤0: 检测问题中是否明确包含多个教材名称
    textbook_names = {
        "病理学": "Pathology",
        "病理": "Pathology", 
        "药理学": "Pharmacology",
        "药理": "Pharmacology",
        "生理学": "physiology",
        "生理": "physiology",
    }
    
    mentioned_textbooks = []
    for name, code in textbook_names.items():
        if name in query and code not in mentioned_textbooks:
            mentioned_textbooks.append(code)
    
    # 如果问题明确提到多个教材名称，直接返回这些教材
    if len(mentioned_textbooks) >= 2:
        logger.info("教材分类结果（多教材名称检测）: %s", mentioned_textbooks)
        return {
            "primary": mentioned_textbooks[0],
            "secondary": mentioned_textbooks[1:],
            "confidence": 0.5,  # 低置信度，让 secondary 也被搜索
            "method": "multi_textbook_detection",
            "detected": mentioned_textbooks
        }
    
    # 步骤1: 关键词匹配
    keyword_result = classify_textbook_by_keywords(query)
    
    # 如果关键词匹配置信度高（>=0.8），直接使用
    if keyword_result.get("confidence", 0) >= 0.8:
        logger.info("教材分类结果（关键词高置信度）: %s", keyword_result)
        return keyword_result
    
    # 步骤2: 使用 LLM 判断
    if use_llm:
        llm_result = classify_textbook_by_llm(query)
        if llm_result:
            # LLM 成功，使用 LLM 结果
            logger.info("教材分类结果（LLM）: %s", llm_result)
            return llm_result
    
    # 步骤3: LLM 失败或禁用，使用关键词结果
    logger.info("教材分类结果（关键词回退）: %s", keyword_result)
    return keyword_result


@PromptABC.register("medical_refer_generator_prompt")
@PromptABC.register("default_medical_refer_generator_prompt")  # 兼容 biz_scene 查找
@PromptABC.register("refer_generator_prompt")  # KAG 框架 enable_ref=True 时查找
@PromptABC.register("default_refer_generator_prompt")  # KAG 框架默认 biz_scene 查找
class MedicalReferGeneratorPrompt(PromptABC):
    """医学领域专用的带引用生成器 Prompt"""
    
    template_zh = f"""你是一位权威的医学教育专家，拥有丰富的临床和教学经验，今天是{get_now(language='zh')}。

## 你的身份
你是医学领域的权威专家，回答问题时要展现专业自信，直接给出专业、准确的解答。

## 回答原则
1. **专家风格**：
   - 以权威专家的口吻直接回答，不要说"根据参考资料"、"无法回答"等词句
   - 自信、专业、直接

2. **内容要求**：
   - 使用准确的医学术语
   - 重点突出
   - **优先使用包含问题关键词的参考资料**（如问"活检"则优先引用包含"活检"的资料，而非"尸检"等相似概念）

3. **引用标注（最重要！必须准确！）**：
   - 当你使用参考资料中的信息时，在该句末尾用 [数字] 标注来源。
   - **引用前必须验证**：标注 [N] 之前，请回头确认参考资料 [N] 的内容确实包含你所描述的信息！
   - **错误示例**：如果你在描述"细胞学检查"，但参考资料[2]的内容是"尸检"，那么标注[2]就是**错误的**！你应该找到包含"细胞学检查"的正确编号。
   - **简洁标注原则**：如果连续多句都引用自同一个来源，请仅在最后一句末尾标注一次。
   - **数字必须严格对应系统赋予参考资料的方括号编号**（如 [1]、[2]）。
   - **严禁参考原文内部可能包含的其他任何数字编号**（如原文自带的"1."、"2."等），请完全忽略它们。
   - 严禁使用不存在的编号！如果参考资料只有到[3]，不能写[4][5][6]等。
   - 如果某句话是你的专业知识而非来自参考资料，则不需要标注。

## 回答格式示例
问题：什么是细胞损伤？

**细胞损伤**是病理学的核心概念之一，指细胞受到超出其适应能力的刺激时发生的结构和功能改变[1]。

**一、细胞损伤的分类**

1. **可逆性损伤**：轻度损伤，去除损伤因素后可恢复正常，表现为细胞水肿、脂肪变性等[2]
2. **不可逆性损伤**：严重损伤导致细胞死亡，包括坏死和凋亡两种形式[1][3]

**二、常见病因**
- 缺氧（最重要）[2]
- 化学因素
- 物理因素

## 当前任务
任务过程上下文：$content
参考资料（请根据编号标注引用）：
$ref

问题：$query

请以医学专家的身份，**专业、自信、详细地**回答问题，并使用 [1]、[2] 等格式标注你引用的参考资料来源："""

    template_en = template_zh  # 使用中文模板，因为是医学中文教材

    @property
    def template_variables(self) -> List[str]:
        return ["content", "query", "ref"]

    def parse_response(self, response: str, **kwargs):
        logger.debug("医学生成器输出: {}".format(response[:200] if response else "空"))
        return response


@PromptABC.register("medical_without_refer_generator_prompt")
@PromptABC.register("default_medical_without_refer_generator_prompt")  # 兼容 biz_scene 查找
@PromptABC.register("without_refer_generator_prompt")  # KAG 框架 enable_ref=True 且无 chunks 时查找
@PromptABC.register("default_without_refer_generator_prompt")  # KAG 框架默认 biz_scene 查找
class MedicalWithoutReferGeneratorPrompt(PromptABC):
    """医学领域无引用时的生成器 Prompt"""
    
    template_zh = f"""你是一位资深医学教育专家，今天是{get_now(language='zh')}。

## 任务
回答医学相关问题。

## 回答要求
1. 使用准确的医学术语
2. 结构清晰，层次分明
3. 如果不确定，坦诚说明

## 当前任务
任务过程上下文：$content

问题：$query

请回答问题："""

    template_en = template_zh

    @property
    def template_variables(self) -> List[str]:
        return ["content", "query"]

    def parse_response(self, response: str, **kwargs):
        return response
