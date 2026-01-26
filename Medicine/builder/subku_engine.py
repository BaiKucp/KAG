from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np


class SubKUEngine:
    def __init__(
        self,
        *,
        vectorizer,
        normalize_ku_name: Callable[[str], str],
        topic_keywords: Sequence[str],
        allowed_knowledge_points: Sequence[str],
        allowed_ku_matrix,
        allowed_ku_keyword_idxs: Dict[str, List[int]],
        enable_sub_ku_global_parent: bool,
        sub_ku_parent_sim_threshold: float,
        sub_ku_parent_sim_margin: float,
        sub_ku_embed_dedupe_threshold: float,
        sub_ku_drop_sim_to_core_threshold: float,
        sub_ku_offtopic_min_sim_to_core: float,
        sub_ku_drop_exact: Set[str],
        sub_ku_sentence_markers: Sequence[str],
        sub_ku_split_punct: Sequence[str],
        sub_ku_space_tail_markers: Sequence[str],
        sub_ku_min_cjk_chars: int,
        sub_ku_max_name_chars: int,
        generate_hash_id: Callable[[str], str],
    ):
        self._vectorizer = vectorizer
        self._normalize_ku_name = normalize_ku_name
        self._topic_keywords = list(topic_keywords)
        self._allowed_knowledge_points = list(allowed_knowledge_points)
        self._allowed_ku_matrix = allowed_ku_matrix
        self._allowed_ku_keyword_idxs = allowed_ku_keyword_idxs or {}
        self._enable_sub_ku_global_parent = bool(enable_sub_ku_global_parent)
        self._sub_ku_parent_sim_threshold = float(sub_ku_parent_sim_threshold)
        self._sub_ku_parent_sim_margin = float(sub_ku_parent_sim_margin)
        self._sub_ku_embed_dedupe_threshold = float(sub_ku_embed_dedupe_threshold)
        self._sub_ku_drop_sim_to_core_threshold = float(sub_ku_drop_sim_to_core_threshold)
        self._sub_ku_offtopic_min_sim_to_core = float(sub_ku_offtopic_min_sim_to_core)
        self._sub_ku_drop_exact = set(sub_ku_drop_exact or set())
        self._sub_ku_sentence_markers = list(sub_ku_sentence_markers)
        self._sub_ku_split_punct = list(sub_ku_split_punct)
        self._sub_ku_space_tail_markers = list(sub_ku_space_tail_markers)
        self._sub_ku_min_cjk_chars = int(sub_ku_min_cjk_chars)
        self._sub_ku_max_name_chars = int(sub_ku_max_name_chars)
        self._generate_hash_id = generate_hash_id

    def build_sub_ku_prompt(self, max_sub_ku_per_chunk: int) -> str:
        return f"""你是一名医学知识点抽取专家，目标是为知识图谱抽取“子知识点（SubKnowledgeUnit）”。

    【输入】
    - chunk：一段医学教材正文（含小标题/编号/英文括号）
    （可选）- core_knowledge_units：本 chunk 已命中的“核心知识点”列表（如果给了，子知识点应尽量能归到这些核心知识点之下）

    【任务】
    从 chunk 中抽取子知识点（SubKnowledgeUnit），要求比核心知识点更细粒度、更可检索。

    【子知识点粒度要求（要更细）】
    子知识点必须是“短语/术术语级”，优先抽取以下类型（只从 chunk 内容来，不要引入外部知识）：
    1) 分类的子类/类型/分型
    2) 机制的具体环节/过程
    3) 条件/诱因/原因
    4) 形态学/功能学具体改变
    5) 关键对比点/边界
    6) 常见部位/对象（如果 chunk 明确提到才抽）

    【强约束：避免子知识点之间“相似度太高”】【必须遵守】
    - 你的输出列表中，任意两条子知识点都必须“明显不同”，不能只是换一种说法。
    - 如果两条子知识点语义几乎相同/近义/包含关系很强（你判断它们像同一个点），必须合并，只保留“更标准、更短、更可作为图谱节点的那条”。
    - 禁止同时输出：同义词、同一概念的中英文重复、仅多了“机制/原因/类型/表现”等尾巴的重复项。
      例（禁止）：["受体上调", "受体功能上调"]（应只保留一个）
      例（禁止）：["萎缩", "萎缩（atrophy）"]（只保留“萎缩”或更细的子类）
      例（禁止）：["病理性萎缩", "病理性萎缩的类型"]（只保留概念项）
    例（禁止）：["上皮组织的化生", "上皮组织化生及其生物学意义"]（视为同一主题的变体，只保留一个更节点化的表达）

        【强约束：过滤“泛化词/叙述句”】【必须遵守】
        - 不要输出过于泛化、单个概念层级太高的词（例如："萎缩"/"增生"/"肥大"/"坏死"/"炎症"/"肿瘤"）。
            需要输出时，必须提炼为更细的子类/分型/机制/条件（例如："病理性萎缩"/"失用性萎缩"/"压迫性萎缩"）。
        - 不要输出“像一句话”的短句（例如包含：见于/多见于/发生于/表现为/可见/导致/因此/包括，或含逗号句号冒号等）。
            如果 chunk 里出现这类表达，请提炼为可作为节点的名词短语（例如："生理性萎缩（physiological atrophy）"），不要把后面的叙述带上。
        - 不要输出“标题 + 空格 + 解释句”的形式（例如："淀粉样变的病理变化 淀粉样变物质主要沉积于细胞间"）。
            只保留前面的标题短语："淀粉样变的病理变化"。

        【强约束：SubKN 不能与核心知识点完全一致】
        - 如果 core_knowledge_units 中已经存在某个知识点名称，你不得再输出与其“完全相同”的子知识点名称（允许近似/更细，但不要一模一样）。

                【强约束：SubKN 之间不能出现“包含关系”】【必须遵守】
                - 你的输出列表中，不允许出现 A 完整包含 B（或 B 完整包含 A）这种“父子包含式重复”。
                    例：如果你输出了“萎缩的类型与机制”，就不要再输出“萎缩的类型”（它被前者包含）。
                - 遇到“总括性短语 + 子项短语”同时存在时，只保留信息量更高/更完整的那一个；不要两者都输出。

    【数量要求】
    - 输出 8~{max_sub_ku_per_chunk} 个（优先接近 {max_sub_ku_per_chunk}，除非 chunk 信息不足）
    - 如果 chunk 信息不足，允许少于 8 个，但不得为了凑数而编造

    【格式要求（严格）】
    只输出严格 JSON 数组（list），元素是子知识点名称字符串。
    不要输出任何解释、不要代码块标记、不要多余字段。

    【自检步骤（你必须在心里执行，但不要写出来）】
    1) 先列出候选子知识点（可以多一点）
    2) 合并近义/高度相似项，删除重复
    3) 检查每个子知识点是否“短语级、可检索、来自 chunk”
    4) 检查列表内任意两条是否过于相似；若是则继续合并/删减
    5) 输出最终 JSON 数组

    【现在开始】
    core_knowledge_units（可选）：
    $core_knowledge_units

    chunk：
    $input
    """

    def extract_subku_titles_from_chunk(self, passage: str) -> List[str]:
        if not isinstance(passage, str) or not passage.strip():
            return []

        def _clean_title(t: str) -> str:
            if not isinstance(t, str):
                return ""
            s = t.strip()
            s = s.replace("**", "").replace("__", "")
            s = re.sub(r"[：:。．、,，；;]+\s*$", "", s)
            s = re.sub(r"^[#\s]*", "", s)
            s = re.sub(r"^[（(]?[一二三四五六七八九十0-9]+[)）]?\s*[、.．：:)]?\s*", "", s)
            s = s.strip()
            if len(s) > 24:
                s = s[:24].strip()
            return s

        titles: List[str] = []
        for raw_line in passage.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            cand = None
            m = re.match(r"^#{1,6}\s*(.+)$", line)
            if m:
                cand = m.group(1)
            if cand is None:
                m = re.match(r"^[-*•]\s*\*\*(.+?)\*\*\s*[：:]", line)
                if m:
                    cand = m.group(1)
            if cand is None:
                m = re.match(r"^[0-9]+[.、）)]\s*\*\*(.+?)\*\*\s*[：:]", line)
                if m:
                    cand = m.group(1)
            if cand is None:
                m = re.match(r"^[（(]?[一二三四五六七八九十0-9]+[)）]?\s*[、.．：:)]\s*(.+)$", line)
                if m:
                    cand = m.group(1)

            if cand is None:
                continue
            cleaned = _clean_title(cand)
            if not cleaned or len(cleaned) < 2:
                continue
            if cleaned in {"概述", "概况", "小结", "总结"}:
                continue
            titles.append(cleaned)

        # De-dup by normalized key preserving order
        seen = set()
        out: List[str] = []
        for t in titles:
            k = self._normalize_ku_name(t)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out

    def extract_topic_keywords(self, text: str) -> Set[str]:
        t = (text or "").strip()
        if not t:
            return set()
        return {k for k in self._topic_keywords if k in t}

    def _cjk_len(self, s: str) -> int:
        return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")

    def filter_sub_ku_names(self, names: List[str]) -> List[str]:
        if not names:
            return []

        def _clean_one(raw: str) -> str:
            if not isinstance(raw, str):
                return ""
            s = raw.strip()
            if not s:
                return ""
            s = s.strip().strip('"').strip("'")
            s = s.replace("**", "").replace("__", "")
            s = re.sub(r"^[-*•]\s+", "", s)
            s = re.sub(r"^[0-9]+[.、）)]\s*", "", s)
            s = s.replace("（", "(").replace("）", ")")
            s = re.sub(r"\s+", " ", s).strip()

            # If the model appends a descriptive sentence after a space, keep only the title-like prefix.
            if " " in s:
                parts = [p for p in re.split(r"\s+", s) if p]
                if len(parts) >= 2:
                    first = parts[0].strip()
                    tail = " ".join(parts[1:]).strip()
                    tail_norm = self._normalize_ku_name(tail)
                    if any(m in tail_norm for m in self._sub_ku_sentence_markers) or any(
                        m in tail_norm for m in self._sub_ku_space_tail_markers
                    ):
                        s = first
                    else:
                        tail_cjk = len(re.findall(r"[\u4e00-\u9fff]", tail))
                        first_cjk = len(re.findall(r"[\u4e00-\u9fff]", first))
                        if first_cjk >= 3 and tail_cjk >= 6:
                            s = first

            cut_pos = None
            for token in (self._sub_ku_split_punct + self._sub_ku_sentence_markers):
                p = s.find(token)
                if p > 0:
                    cut_pos = p if cut_pos is None else min(cut_pos, p)
            if cut_pos is not None:
                s = s[:cut_pos].strip()

            s = re.sub(r"[：:。．、,，；;]+\s*$", "", s).strip()
            if len(s) > self._sub_ku_max_name_chars:
                s = s[: self._sub_ku_max_name_chars].strip()
            return s

        out: List[str] = []
        seen = set()
        for raw in names:
            name = _clean_one(raw)
            if not name:
                continue

            simple = re.sub(r"\([^)]*\)", "", name)
            simple = re.sub(r"\s+", "", simple).strip()
            if not simple:
                continue
            if simple in self._sub_ku_drop_exact:
                continue
            cjk_len = len(re.findall(r"[\u4e00-\u9fff]", simple))
            if cjk_len > 0 and cjk_len < self._sub_ku_min_cjk_chars:
                continue

            nk = self._normalize_ku_name(name)
            if not nk or nk in seen:
                continue
            seen.add(nk)
            out.append(name)
        return out

    def filter_sub_ku_not_equal_knowledge_units(self, names: List[str], core_kus: List[dict]) -> List[str]:
        if not names:
            return []
        core_set = set()
        for ku in core_kus or []:
            txt = (ku.get("content") or "").strip()
            k = self._normalize_ku_name(txt)
            if k:
                core_set.add(k)
        if not core_set:
            return names
        out: List[str] = []
        for n in names:
            if self._normalize_ku_name(n) in core_set:
                continue
            out.append(n)
        return out

    def _best_sim_to_texts(self, name: str, texts: List[str]) -> float:
        if not texts:
            return -1.0
        try:
            sub_vec = np.array(self._vectorizer.vectorize([name])[0], dtype=float)
            n0 = np.linalg.norm(sub_vec)
            if n0 == 0:
                return -1.0
            sub_vec = sub_vec / n0

            best = -1.0
            for t in texts:
                if not t:
                    continue
                v = np.array(self._vectorizer.vectorize([t])[0], dtype=float)
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                v = v / n
                sim = float(np.dot(v, sub_vec))
                if sim > best:
                    best = sim
            return best
        except Exception:
            return -1.0

    def filter_sub_ku_not_too_similar_to_core_kus(
        self,
        names: List[str],
        core_kus: List[dict],
        threshold: Optional[float] = None,
    ) -> List[str]:
        if not names:
            return []
        thr = self._sub_ku_drop_sim_to_core_threshold if threshold is None else float(threshold)
        core_texts: List[str] = []
        for ku in core_kus or []:
            t = (ku.get("content") or "").strip()
            if t:
                core_texts.append(t)
        if not core_texts:
            return names

        # Goal: normally drop subKUs that are too close to ANY core KU.
        # Exception: if a specific core KU would end up with zero subKUs, we allow
        # one near-duplicate subKU (the best-matching dropped candidate) to attach.
        try:
            core_vecs = np.array(self._vectorizer.vectorize(core_texts), dtype=float)
            core_norms = np.linalg.norm(core_vecs, axis=1, keepdims=True)
            core_norms[core_norms == 0] = 1.0
            core_vecs = core_vecs / core_norms
        except Exception:
            # Fallback to previous coarse behavior if vectorization fails.
            out: List[str] = []
            for n in names:
                sim = self._best_sim_to_texts(n, core_texts)
                if sim >= thr:
                    continue
                out.append(n)
            return out

        kept: List[str] = []
        kept_by_core = [0 for _ in range(len(core_texts))]
        dropped_best_by_core: Dict[int, tuple[str, float]] = {}

        for n in names:
            if not isinstance(n, str):
                continue
            cand = n.strip()
            if not cand:
                continue
            try:
                sub_vec = np.array(self._vectorizer.vectorize([cand])[0], dtype=float)
                n0 = float(np.linalg.norm(sub_vec))
                if n0 == 0:
                    # Can't compare; keep it and let later stages decide.
                    kept.append(cand)
                    continue
                sub_vec = sub_vec / n0
                sims = core_vecs @ sub_vec
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
            except Exception:
                kept.append(cand)
                continue

            if best_sim >= thr:
                prev = dropped_best_by_core.get(best_idx)
                if prev is None or best_sim > prev[1]:
                    dropped_best_by_core[best_idx] = (cand, best_sim)
                continue

            kept.append(cand)
            kept_by_core[best_idx] += 1

        # Re-add one high-sim candidate for any core KU that has none.
        seen = {self._normalize_ku_name(x) for x in kept if isinstance(x, str)}
        for core_idx, count in enumerate(kept_by_core):
            if count > 0:
                continue
            cand_sim = dropped_best_by_core.get(core_idx)
            if not cand_sim:
                continue
            cand, _sim = cand_sim
            nk = self._normalize_ku_name(cand)
            if not nk or nk in seen:
                continue
            kept.append(cand)
            seen.add(nk)

        return kept

    def filter_sub_ku_off_topic_by_core_kus(
        self,
        names: List[str],
        core_kus: List[dict],
        min_sim: Optional[float] = None,
    ) -> List[str]:
        if not names:
            return []
        ms = self._sub_ku_offtopic_min_sim_to_core if min_sim is None else float(min_sim)

        core_kw_union: Set[str] = set()
        core_texts: List[str] = []
        for ku in core_kus or []:
            txt = (ku.get("content") or "").strip()
            if not txt:
                continue
            core_texts.append(txt)
            core_kw_union |= self.extract_topic_keywords(txt)

        if not core_texts:
            return names

        out: List[str] = []
        for n in names:
            sub_kws = self.extract_topic_keywords(n)
            if not sub_kws:
                out.append(n)
                continue
            if core_kw_union and (sub_kws & core_kw_union):
                out.append(n)
                continue

            # keyword mismatch -> require semantic closeness to keep
            best_sim = self._best_sim_to_texts(n, core_texts)
            if best_sim < ms:
                continue
            out.append(n)
        return out

    def dedupe_sub_kus_by_containment(self, names: List[str]) -> List[str]:
        if not names:
            return []
        ordered: List[str] = []
        for raw in names:
            if not isinstance(raw, str):
                continue
            s = raw.strip()
            if s:
                ordered.append(s)
        if len(ordered) <= 1:
            return ordered

        keep = [True] * len(ordered)
        normed = [self._normalize_ku_name(s) for s in ordered]
        for i in range(len(ordered)):
            if not keep[i] or not normed[i]:
                continue
            for j in range(len(ordered)):
                if i == j or not keep[j] or not normed[j]:
                    continue
                a, b = normed[i], normed[j]
                if len(a) == len(b):
                    continue
                if a in b:
                    # i contained by j => drop i
                    if len(a) < len(b):
                        keep[i] = False
                        break
        return [ordered[i] for i in range(len(ordered)) if keep[i]]

    def _subku_soft_key(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        s = name.strip()
        if not s:
            return ""

        s = s.replace("（", "(").replace("）", ")")
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"\([^)]*\)", "", s)

        for token in ["及其", "以及", "及", "与", "和", "的"]:
            s = s.replace(token, "")

        suffixes = ["生物学意义","临床意义","病理意义","意义","概念","定义","特点","特征","表现","机制",
        ]
        changed = True
        while changed:
            changed = False
            for suf in suffixes:
                if s.endswith(suf) and len(s) > len(suf) + 1:
                    s = s[: -len(suf)]
                    changed = True
        return s.strip()

    def dedupe_sub_kus_by_soft_key(self, names: List[str]) -> List[str]:
        if not names:
            return []

        keys = [self._subku_soft_key(n) for n in names]

        kept_idx: List[int] = []
        seen = set()
        for i, k in enumerate(keys):
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            kept_idx.append(i)

        if len(kept_idx) <= 1:
            return [names[i] for i in kept_idx]

        keep_flag = [True] * len(kept_idx)
        for a_pos, i in enumerate(kept_idx):
            ka = keys[i]
            for b_pos, j in enumerate(kept_idx):
                if a_pos == b_pos:
                    continue
                kb = keys[j]
                if not kb or len(kb) <= len(ka):
                    continue
                if ka and (ka in kb):
                    keep_flag[a_pos] = False
                    break

        out: List[str] = []
        for pos, ok in enumerate(keep_flag):
            if ok:
                out.append(names[kept_idx[pos]])
        return out

    def dedupe_sub_kus_by_embedding(self, names: List[str], threshold: Optional[float] = None) -> List[str]:
        if not names:
            return []

        thr = self._sub_ku_embed_dedupe_threshold if threshold is None else float(threshold)

        ordered: List[str] = []
        seen = set()
        for raw in names:
            if not isinstance(raw, str):
                continue
            s = raw.strip()
            if not s:
                continue
            nk = self._normalize_ku_name(s)
            if not nk or nk in seen:
                continue
            seen.add(nk)
            ordered.append(s)

        if len(ordered) <= 1:
            return ordered

        try:
            vecs = self._vectorizer.vectorize(ordered)
            mat = np.array(vecs, dtype=float)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms

            kept: List[str] = []
            kept_vecs: List[np.ndarray] = []
            for i, name in enumerate(ordered):
                v = mat[i]
                if not kept_vecs:
                    kept.append(name)
                    kept_vecs.append(v)
                    continue
                sims = np.dot(np.stack(kept_vecs, axis=0), v)
                if float(np.max(sims)) >= thr:
                    continue
                kept.append(name)
                kept_vecs.append(v)
            return kept
        except Exception:
            return ordered

    def pick_parent_ku_for_sub(self, sub_name: str, core_kus: List[dict]) -> Optional[dict]:
        sub = (sub_name or "").strip()
        if not sub:
            return None

        sub_vec = None
        try:
            v0 = np.array(self._vectorizer.vectorize([sub])[0], dtype=float)
            n0 = np.linalg.norm(v0)
            if n0 != 0:
                sub_vec = v0 / n0
        except Exception:
            sub_vec = None

        best_core = None
        best_core_sim = -1.0
        if core_kus:
            if len(core_kus) == 1:
                best_core = core_kus[0]
                if sub_vec is not None:
                    txt = (best_core.get("content") or "").strip()
                    if txt:
                        try:
                            v = np.array(self._vectorizer.vectorize([txt])[0], dtype=float)
                            n = np.linalg.norm(v)
                            if n != 0:
                                v = v / n
                                best_core_sim = float(np.dot(v, sub_vec))
                        except Exception:
                            pass
            elif sub_vec is not None:
                for ku in core_kus:
                    txt = (ku.get("content") or "").strip()
                    if not txt:
                        continue
                    try:
                        v = np.array(self._vectorizer.vectorize([txt])[0], dtype=float)
                        n = np.linalg.norm(v)
                        if n == 0:
                            continue
                        v = v / n
                        sim = float(np.dot(v, sub_vec))
                        if sim > best_core_sim:
                            best_core_sim = sim
                            best_core = ku
                    except Exception:
                        continue

        best_global_name = None
        best_global_sim = -1.0
        kw_best_name = None
        kw_best_sim = -1.0
        sub_kws: Set[str] = set()
        if self._enable_sub_ku_global_parent and (sub_vec is not None) and (self._allowed_ku_matrix is not None):
            try:
                sims = np.dot(self._allowed_ku_matrix, sub_vec)
                best_idx = int(np.argmax(sims))
                best_global_sim = float(sims[best_idx])
                best_global_name = self._allowed_knowledge_points[best_idx]

                sub_kws = self.extract_topic_keywords(sub)
                if sub_kws and self._allowed_ku_keyword_idxs:
                    idxs: List[int] = []
                    for k in sub_kws:
                        idxs.extend(self._allowed_ku_keyword_idxs.get(k, []))
                    if idxs:
                        sims_local = np.dot(self._allowed_ku_matrix[idxs], sub_vec)
                        local_best = int(np.argmax(sims_local))
                        kw_best_idx = idxs[local_best]
                        kw_best_sim = float(sims_local[local_best])
                        kw_best_name = self._allowed_knowledge_points[kw_best_idx]
                        if kw_best_name and (kw_best_sim >= best_global_sim - 0.01):
                            best_global_sim = kw_best_sim
                            best_global_name = kw_best_name
            except Exception:
                best_global_name = None
                best_global_sim = -1.0
                kw_best_name = None
                kw_best_sim = -1.0
                sub_kws = set()

        core_ok = bool(best_core) and (best_core_sim >= self._sub_ku_parent_sim_threshold)

        relaxed_global_threshold = max(0.55, self._sub_ku_parent_sim_threshold - 0.10)
        global_ok = bool(best_global_name) and (
            (best_global_sim >= self._sub_ku_parent_sim_threshold)
            or (sub_kws and (best_global_sim >= relaxed_global_threshold))
        )

        sub_kws2 = sub_kws or self.extract_topic_keywords(sub)
        core_txt = (best_core.get("content") or "").strip() if isinstance(best_core, dict) else ""
        core_kws = self.extract_topic_keywords(core_txt)
        global_kws = self.extract_topic_keywords(best_global_name or "")
        if sub_kws2:
            core_has = bool(core_kws & sub_kws2)
            global_has = bool(global_kws & sub_kws2)
            if global_ok and global_has and (not core_has):
                return {
                    "name": self._generate_hash_id(best_global_name),
                    "category": "KnowledgeUnit",
                    "content": best_global_name,
                }
            if core_ok and core_has and (not global_has):
                return best_core

            if (not core_has) and kw_best_name:
                kw_kws = self.extract_topic_keywords(kw_best_name)
                if (kw_kws & sub_kws2) and (kw_best_sim >= relaxed_global_threshold):
                    return {
                        "name": self._generate_hash_id(kw_best_name),
                        "category": "KnowledgeUnit",
                        "content": kw_best_name,
                    }

        if global_ok and (not core_ok or (best_global_sim >= best_core_sim + self._sub_ku_parent_sim_margin)):
            return {
                "name": self._generate_hash_id(best_global_name),
                "category": "KnowledgeUnit",
                "content": best_global_name,
            }

        if core_ok:
            return best_core
        return None

    def assemble_sub_knowledge_unit(
        self,
        *,
        sub_graph,
        sub_names: List[str],
        core_kus: List[dict],
        section_id: str,
        chunk_id: str,
        max_sub_ku_per_chunk: int,
        assemble_node_cb: Callable,
    ) -> List[dict]:
        if not sub_names or not chunk_id:
            return []

        sub_names = self.filter_sub_ku_names(sub_names)
        sub_names = self.filter_sub_ku_not_equal_knowledge_units(sub_names, core_kus or [])
        sub_names = self.filter_sub_ku_not_too_similar_to_core_kus(sub_names, core_kus or [])
        sub_names = self.filter_sub_ku_off_topic_by_core_kus(sub_names, core_kus or [])
        sub_names = self.dedupe_sub_kus_by_containment(sub_names)
        sub_names = self.dedupe_sub_kus_by_soft_key(sub_names)
        if not sub_names:
            return []

        created: List[dict] = []
        deduped_names = self.dedupe_sub_kus_by_embedding(sub_names)

        for raw in deduped_names[:max_sub_ku_per_chunk]:
            name = (raw or "").strip()
            if not name:
                continue

            parent = self.pick_parent_ku_for_sub(name, core_kus or [])
            parent_id = parent.get("name") if isinstance(parent, dict) else None
            parent_content = (parent.get("content") or "").strip() if isinstance(parent, dict) else ""
            if parent_content and (self._normalize_ku_name(parent_content) == self._normalize_ku_name(name)):
                continue

            if parent_id and isinstance(parent, dict) and parent.get("category") == "KnowledgeUnit":
                if parent_content:
                    assemble_node_cb(
                        sub_graph,
                        parent_id,
                        parent_content,
                        "KnowledgeUnit",
                        {
                            "knowledgeType": "概念",
                            "content": parent_content,
                            "name": parent_content,
                            "ontology": "医学",
                            "relatedQuery": [],
                        },
                    )

            sub_id = self._generate_hash_id(f"sub::{name}")
            props = {
                "name": name,
                "content": name,
                "knowledgeType": "概念",
                "ontology": "医学",
            }
            assemble_node_cb(sub_graph, sub_id, name, "SubKnowledgeUnit", props)

            sub_graph.add_edge(
                s_id=sub_id,
                s_label="SubKnowledgeUnit",
                p="sourceChunk",
                o_id=chunk_id,
                o_label="Chunk",
            )
            sub_graph.add_edge(
                s_id=chunk_id,
                s_label="Chunk",
                p="hasSubKnowledgeUnit",
                o_id=sub_id,
                o_label="SubKnowledgeUnit",
            )
            if parent_id:
                sub_graph.add_edge(
                    s_id=sub_id,
                    s_label="SubKnowledgeUnit",
                    p="parentKnowledgeUnit",
                    o_id=parent_id,
                    o_label="KnowledgeUnit",
                )
                sub_graph.add_edge(
                    s_id=parent_id,
                    s_label="KnowledgeUnit",
                    p="hasSubKnowledgeUnit",
                    o_id=sub_id,
                    o_label="SubKnowledgeUnit",
                )

            created.append({"name": sub_id, "category": "SubKnowledgeUnit", "content": name})

        return created
