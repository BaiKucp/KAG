# 修改 KAG 框架的 llm_generator.py
# 让它同时支持 KAGRetrievedResponse (chunk_datas) 和 RetrieverOutput (chunks)

import os

target_file = r"C:\Users\51623\Kag\KAG\kag\solver\generator\llm_generator.py"

# 读取原文件
with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 备份原文件
backup_file = target_file + ".bak"
if not os.path.exists(backup_file):
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原文件到: {backup_file}")

# 定义要替换的旧代码
old_code = '''        for task in context.gen_task(False):
            tasks.append(task)
            if isinstance(task.result, KAGRetrievedResponse) and self.chunk_reranker:
                rerank_queries.append(
                    task.arguments.get("rewrite_query", task.arguments["query"])
                )
                chunks.append(task.result.chunk_datas)
            results.append(to_task_context_str(task.get_task_context()))'''

# 定义新代码 - 同时支持 KAGRetrievedResponse 和 RetrieverOutput
new_code = '''        for task in context.gen_task(False):
            tasks.append(task)
            # 支持 KAGRetrievedResponse (chunk_datas) 和 RetrieverOutput (chunks)
            if self.chunk_reranker:
                task_chunks = None
                if isinstance(task.result, KAGRetrievedResponse):
                    task_chunks = task.result.chunk_datas
                elif hasattr(task.result, 'chunks') and task.result.chunks:
                    # 兼容 RetrieverOutput 类型
                    task_chunks = task.result.chunks
                
                if task_chunks:
                    rerank_queries.append(
                        task.arguments.get("rewrite_query", task.arguments["query"])
                    )
                    chunks.append(task_chunks)
            results.append(to_task_context_str(task.get_task_context()))'''

if old_code in content:
    new_content = content.replace(old_code, new_code)
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ 修改成功！")
    print("修改内容：让 llm_generator 同时支持 KAGRetrievedResponse.chunk_datas 和 RetrieverOutput.chunks")
else:
    print("❌ 未找到目标代码段，可能已被修改过或文件格式不同")
    print("请手动检查文件内容")
