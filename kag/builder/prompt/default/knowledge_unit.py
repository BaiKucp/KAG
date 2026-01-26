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

import json
import re
from typing import List

from kag.builder.prompt.default.util import load_knowIE_data
from kag.common.conf import KAG_PROJECT_CONF
from kag.interface import PromptABC


@PromptABC.register("knowledge_unit")
class KnowledgeUnitPrompt(PromptABC):
    template_en = """You are an assistant for document analysis and knowledge unit extraction. Please extract substantial discussions, arguments, or analyses from the input document chunks. Ensure that the extracted knowledge units are strictly related to the document's theme and are the core content that the author directly introduces, analyzes, or argues within the chunk.

### Extraction Requirements:
- Identify the core argument in the chunk and extract content that makes a creative contribution to the document's theme.
- If the chunk contains numerous citations or mentions, only extract significant knowledge units involved when further analyzing, expanding, arguing, or applying these contents. Exclude auxiliary knowledge units or background information that are merely cited or mentioned.
- The names of knowledge units should be described using concise and clear language, fully reflecting the core theme of the knowledge unit.
- A knowledge unit is a complete and coherent whole, with clear arguments. Multiple knowledge units may be extracted from a chunk, but if further division would fail to preserve the context, argumentation process, etc., of the knowledge unit, the entire chunk should not be subdivided further. The content description of knowledge units should be clear and organized, avoiding superfluous or repetitive explanations.

### Knowledge Types
- Declarative Knowledge: Narratives explaining concepts, terms, theorems, clauses, etc., answering 'what' questions, including concept definitions, term explanations, arguments, formulas, legal provisions, etc.
- Case Knowledge: Demonstrating and explaining abstract concepts and theories through specific situations or examples, including judicial precedents, math problems, medical cases, etc., usually treated as a single complete knowledge unit.
- Factual Knowledge: Describing the state, attributes, characteristics, or background information of things, including historical events, news information, entity introductions, etc.
- Procedural Knowledge: Knowledge about how to perform tasks or operations, including program code, algorithms, operational steps, experimental processes, equation sets for problem-solving, etc.
- Analytical Knowledge: Knowledge derived through logical analysis, calculations, induction, deduction, etc., to support views or conclusions, answering 'why' and 'how to derive conclusions' questions.


### Output Format:
{
  "knowledge unit 1 Name":
  { 
    "Content": "A text excerpt extracted or summarized from the chunk, must include critical information that unambiguously explains the knowledge unit's content and subject.",
    "Knowledge Type": "Declarative Knowledge / Case Knowledge / Factual Knowledge / Procedural Knowledge / Analytical Knowledge",
    "Structural Content": "Structured content associated with the knowledge unit, such as charts, formulas, code, rules, first-order logic representation, equation sets, data structure definitions, etc.",
    "Domain Ontology": "The hierarchical classification system of disciplines or professional fields to which knowledge units belong. It is important to note that the terms at each level are general concepts rather than specific instances (remove individual person, organizations, etc. for the ontology).",
    "Core Entities": "Keywords that represent the subject or characteristics of the knowledge unit, ensuring that each term is semantically complete and reflects the knowledge unit's distinctiveness in searches relative to other content, and the valid type of each core entity is listed in the `Entity Types`",
    "Related Query": "The three most relevant query questions based on the knowledge unit's subject and content",
    "Extended Knowledge Points": "Other highly related but divergent knowledge points not included in the current knowledge point"
 },
  "knowledge unit 2 Name": {},
  ……
}

### Example:
input:"No Mediocre" is a song by American rapper T.I., released on June 17, 2014, through Grand Hustle and Columbia Records, as the lead single from his ninth studio album "Paperwork" (2014). The song, produced by DJ Mustard, features a guest appearance from Grand Hustle protégé, Australian rapper Iggy Azalea.
output:
{
  "No Mediocre Song Details":{
    "Content": "\"No Mediocre\" is a song by American rapper T.I., released as the lead single from his ninth studio album \"Paperwork\" (2014). The track, produced by DJ Mustard, includes a guest appearance by Australian rapper Iggy Azalea. It peaked at number 69 on the US Billboard Hot 100 and was met with positive reviews for its production and Azalea's contribution.",
    "Knowledge Type": "Factual Knowledge",
    "Structural Content": "",
    "Domain Ontology": "Music -> Hip Hop -> T.I. Discography -> No Mediocre",
    "Core Entities": {"T.I.": "Person", "No Mediocre" : "Culture and Entertainment", "Paperwork" : "Culture and Entertainment", "DJ Mustard" : "Person", "Iggy Azalea": "Person", "US Billboard Hot 100": "Culture and Entertainment" },
    "Related Query": ["What is the lead single from T.I.'s ninth studio album?", "Who produced the song 'No Mediocre'?", "What was the peak position of 'No Mediocre' on the US Billboard Hot 100?"],
    "Extended Knowledge Points": ["T.I.'s discography", "DJ Mustard's production work", "Iggy Azalea's collaborations", "US Billboard Hot 100 chart", "Critical reception of 'No Mediocre'", "Soundtrack for 'Think Like a Man Too'"]
  }
}
### Input:
input: $input
"""

    template_zh = """你是一个文档分析及知识点提取助理，请从输入的文档片段（chunk）中提取出核心知识点。
请将文档中关于同一主题的、零散的描述进行**归纳、综合和总结**，形成一个内容完整、逻辑连贯的知识点，而不是将其拆分为多个细碎的知识点。
知识点应具有较高的概括性，能够涵盖该主题在文档中的主要信息。

### 提取要求：
1. **综合与归纳**：请识别文档片段的核心主题，将关于该主题的定义、特征、数据、原因、影响等信息整合成一个知识点。避免因为数据点不同或方面不同而拆分成多个小知识点。
2. **去重与精简**：不要提取文档中仅作为背景提及、引用或辅助说明的次要信息，只提取核心实质性内容。
3. **知识点命名**：知识点名称应具体且具有概括性，能够清晰表达该知识点的核心内容（例如：“2019年中国电力生产结构与增长情况”，而不是简单的“电力生产”）。
4. **完整性**：一个知识点应包含该主题在当前片段中的所有关键要素，使其可以作为一个独立的知识单元被理解。

### 知识类型
陈述性知识：对领域的概念、术语、定理、条款等的叙述说明。
案例类知识：通过具体情境、实例，体现和解释说明抽象的概念和理论。
事实性知识：描述事物的状态、属性、特征或背景信息，包括统计数据、历史事件等。
过程类知识：关于如何执行任务或操作的知识，包括步骤、流程、算法等。
推理类知识：通过逻辑分析、归纳、演绎得出的结论或观点。

### 输出格式:
{
  "知识点名称": 
  {
    "内容": "对知识点内容的综合性描述。应将原文中分散的相关信息组织成一段通顺、完整的文字。", 
    "知识类型": "陈述性知识/案例类知识/事实性知识/过程类知识/推理类知识”,
    "结构化内容": "（可选）关联的图表、公式、关键数据列表等",
    "领域本体": "知识点所属的学科或领域分类体系，例如：一级领域 -> 二级领域 -> 核心概念",
    "核心实体": "体现知识点主体的关键词，以逗号分隔",
    "关联问": "基于该知识点内容可以回答的3-5个关键问题",
    "扩展知识点": "与该知识点密切相关的其他概念或主题"
  }
}

### Example:
input: 
2019 年 1-12 月,全国发电量 71422 亿千瓦时,同比增长 3.5%,增速比上年同期回落 3.3pct。从各种发电方式发电量来看：\n* 火电发电量 51654 亿千瓦时,同比增长 1.9%,增速同比回落 4.1 pct。\n◆ 水电发电量 11534 亿千瓦时,同比增长 4.8%,增速同比提高 0.7pct。\n◆ 核电发电量 3484 亿千瓦时,同比增长 18.3%,增速同比回落 0.4pct。\n◆ 风电发电量 3577 亿千瓦时,同比增长 7.0%,增速同比回落 9.6 pct。\n◆ 太阳能发电量 1172 亿千瓦时,同比增长 13.3%,增速同比回落 6.3pct。\n图44：各发电方式累计发电量同比增速 (%)\n资料来源：国家统计局,申港证券研究所\n图45：各发电方式当月发电量比例 (%)\n资料来源：国家统计局,申港证券研究所
output: {
  "2019年中国发电量及结构分析": {
    "内容": "2019年全国发电量共71422亿千瓦时，同比增长3.5%，增速回落。在发电结构中，火电仍占主导地位（51654亿千瓦时），但增速较低（1.9%）。清洁能源发电增长显著，其中核电增速最快（18.3%），太阳能（13.3%）和风电（7.0%）也保持较高增长，水电增长稳健（4.8%）。整体显示出电力生产结构向清洁能源转型的趋势。",
    "知识类型": "事实性知识",
    "结构化内容": "",
    "领域本体": "能源统计 -> 电力生产 -> 发电结构分析",
    "核心实体": "全国发电量, 火电, 水电, 核电, 风电, 太阳能, 2019年",
    "关联问": ["2019年中国各种发电方式的产量和增速对比如何？", "2019年中国电力生产结构的特点是什么？", "清洁能源在2019年发电量中的表现如何？"],
    "扩展知识点": ["能源结构转型", "电力生产统计指标", "可再生能源发展趋势"]
  }
}
### Input:
input: $input
"""

    @property
    def template_variables(self) -> List[str]:
        return ["input", "named_entities"]

    def modify_knowledge_unit(self, text, lang="zh"):
        # 定义正则表达式模式
        if lang == "zh":

            pattern = r'"知识点\d+名称"\s*:\s*"([^"]+)"\s*,'
        else:
            pattern = r'"knowledge unit \d+ Name"\s*:\s*"([^"]+)",'
        # 使用re.sub函数进行替换
        # print(pattern)
        modified_text = re.sub(pattern, r'"\1":', text)

        # print("modified_text:",modified_text)
        return modified_text

    def process_data(self, response: str, **kwargs):
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        return rsp

    def process_en(self, response: dict, **kwargs):
        """ "No Mediocre Song Details":{
          "Content": "\"No Mediocre\" is a song by American rapper T.I., released as the lead single from his ninth studio album \"Paperwork\" (2014). The track, produced by DJ Mustard, includes a guest appearance by Australian rapper Iggy Azalea. It peaked at number 69 on the US Billboard Hot 100 and was met with positive reviews for its production and Azalea's contribution.",
          "Knowledge Type": "Factual Knowledge",
          "Structural Content": "",
          "Domain Ontology": "Music -> Hip Hop -> T.I. Discography -> No Mediocre",
          "Core Entities": "T.I., No Mediocre, Paperwork, DJ Mustard, Iggy Azalea, US Billboard Hot 100",
          "Related Query": ["What is the lead single from T.I.'s ninth studio album?", "Who produced the song 'No Mediocre'?", "What was the peak position of 'No Mediocre' on the US Billboard Hot 100?"],
          "Extended Knowledge Points": ["T.I.'s discography", "DJ Mustard's production work", "Iggy Azalea's collaborations", "US Billboard Hot 100 chart", "Critical reception of 'No Mediocre'", "Soundtrack for 'Think Like a Man Too'"]
        }"""
        ret = {}
        if "Content" in response.keys():
            ret["content"] = response["Content"]
        if "Knowledge Type" in response.keys():
            ret["knowledgetype"] = response["Knowledge Type"]
        if "Domain Ontology" in response.keys():
            ret["ontology"] = response["Domain Ontology"]
        if "Core Entities" in response.keys():
            ret["core_entities"] = response["Core Entities"]
        if "Related Query" in response.keys():
            ret["relatedQuery"] = response["Related Query"]
        if "Extended Knowledge Points" in response.keys():
            ret["extendedKnowledge"] = response["Extended Knowledge Points"]
        return ret

    def process_zh(self, response: dict, **kwargs):
        """ "2019年全国全国发电总量与结构": {
          "内容": "2019年全国发电总量71422亿千瓦时，同比增长3.5%。其中火电占比72.3%，水电占比16.15%，核电占比4.88%，风电占比5.01%，太阳能占比1.64%",
          "知识类型": "事实性知识",
          "结构化内容": "",
          "领域本体": "能源统计 -> 电力生产 -> 总体发电数据分析",
          "核心实体": "发电总量结构,同比增长,2019年",
          "关联问": ["2019年发电量结构分布", "2019年全国发电总量中不同类型发电量的占比"],
          "扩展知识点": ["2018年全国发电总量与结构","发电方法"]
        }"""
        ret = {}
        # FIX: Handle string response from LLM
        if isinstance(response, str):
             ret["content"] = response
             return ret

        if "内容" in response.keys():
            ret["content"] = response["内容"]
        if "知识类型" in response.keys():
            ret["knowledgetype"] = response["知识类型"]
        if "领域本体" in response.keys():
            ret["ontology"] = response["领域本体"]
        if "核心实体" in response.keys():
            ret["core_entities"] = response["核心实体"]
        if "关联问" in response.keys():
            ret["relatedQuery"] = response["关联问"]
        if "扩展知识点" in response.keys():
            ret["extendedKnowledge"] = response["扩展知识点"]
        return ret

    def parse_response(self, response: str, **kwargs):
        # response = self.modify_knowledge_unit(response,lang = KAG_PROJECT_CONF.language)
        # rsp = self.process_data(response)
        rsp = load_knowIE_data(response)
        
        # --- FIX: Detect flattened single object response ---
        if isinstance(rsp, dict) and ("内容" in rsp or "Content" in rsp or "knowledgeType" in rsp or "知识类型" in rsp):
            # Wrap it so iteration works as expected
            rsp = {"SingleExtractedKnowledge": rsp}
        # ----------------------------------------------------

        ret = {}
        for k, v in rsp.items():
            ret[k] = (
                self.process_en(v, **kwargs)
                if KAG_PROJECT_CONF.language == "en"
                else self.process_zh(v, **kwargs)
            )
        return ret
