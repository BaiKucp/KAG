#!/usr/bin/env python3
"""
智能清理指定命名空间的数据（缓存 + Neo4j）
流程：
1. 从 Neo4j 获取指定命名空间的 Chunk 节点内容
2. 计算这些内容的哈希值（与 indexer 中的缓存 key 一致）
3. 删除对应的缓存条目
4. 最后清空 Neo4j 中的数据

用法: python smart_clean_namespace.py --namespace MedicineHigherMath
"""

import argparse
import hashlib
import os
import sqlite3
from neo4j import GraphDatabase


def generate_hash_id(content: str) -> str:
    """生成与 indexer.py 中相同的哈希 ID"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_chunk_contents_from_neo4j(namespace: str) -> list:
    """从 Neo4j 获取指定命名空间的所有 Chunk 内容"""
    
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
    
    print(f"连接到 Neo4j: {neo4j_uri}, 数据库: {neo4j_database}")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    contents = []
    chunk_ids = []
    
    with driver.session(database=neo4j_database) as session:
        # 获取所有 Chunk 节点的 content 和 id
        query = f"""
            MATCH (n:`{namespace}.Chunk`)
            RETURN n.id AS chunk_id, n.content AS content, n.section_id AS section_id
        """
        result = session.run(query)
        
        for record in result:
            chunk_id = record.get("chunk_id")
            content = record.get("content")
            section_id = record.get("section_id")
            
            if chunk_id:
                chunk_ids.append(chunk_id)
            if content:
                contents.append(content)
            if section_id:
                # 缓存 key 可能基于 section_id + sequence 的组合
                contents.append(section_id)
        
        print(f"找到 {len(chunk_ids)} 个 Chunk 节点")
    
    driver.close()
    return chunk_ids, contents


def delete_cache_by_keys(chunk_ids: list, contents: list, dry_run: bool = False):
    """根据 chunk ID 和内容删除缓存条目"""
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpt")
    cache_dirs = ["FixKnowledgeUnitExtractor", "BatchVectorizer"]
    
    # 生成可能的缓存 key 变体（字符串形式）
    possible_keys = set()
    
    for chunk_id in chunk_ids:
        if chunk_id:
            possible_keys.add(str(chunk_id))
            possible_keys.add(f"{chunk_id}->")
    
    # 基于 content 的哈希
    for content in contents:
        if content:
            hash_key = generate_hash_id(str(content))
            possible_keys.add(hash_key)
            possible_keys.add(f"{hash_key}->")
    
    print(f"生成了 {len(possible_keys)} 个可能的缓存 key")
    
    total_deleted = 0
    
    for cache_name in cache_dirs:
        cache_path = os.path.join(ckpt_dir, cache_name, "cache.db")
        
        if not os.path.exists(cache_path):
            print(f"缓存文件不存在: {cache_path}")
            continue
        
        print(f"\n处理缓存: {cache_name}")
        
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()
        
        # 统计总条目
        cursor.execute("SELECT COUNT(*) FROM Cache")
        total = cursor.fetchone()[0]
        print(f"  总缓存条目: {total}")
        
        # 获取所有缓存 key（可能是 BLOB 或 TEXT）
        cursor.execute("SELECT rowid, key FROM Cache")
        all_rows = cursor.fetchall()
        
        # 找出匹配的 rowid
        rowids_to_delete = []
        for rowid, cache_key in all_rows:
            # 将 key 转换为字符串进行比较
            if isinstance(cache_key, bytes):
                try:
                    key_str = cache_key.decode('utf-8')
                except:
                    key_str = str(cache_key)
            else:
                key_str = str(cache_key) if cache_key else ""
            
            # 检查是否匹配任何可能的 key
            for possible_key in possible_keys:
                if possible_key in key_str or key_str.startswith(possible_key):
                    rowids_to_delete.append(rowid)
                    break
        
        print(f"  匹配到 {len(rowids_to_delete)} 个缓存条目待删除")
        
        if rowids_to_delete and not dry_run:
            # 分批删除（使用 rowid）
            batch_size = 100
            for i in range(0, len(rowids_to_delete), batch_size):
                batch = rowids_to_delete[i:i+batch_size]
                placeholders = ",".join(["?" for _ in batch])
                cursor.execute(f"DELETE FROM Cache WHERE rowid IN ({placeholders})", batch)
            
            conn.commit()
            total_deleted += len(rowids_to_delete)
            print(f"  已删除 {len(rowids_to_delete)} 条缓存记录")
            
            # VACUUM
            print("  执行 VACUUM 压缩...")
            conn.execute("VACUUM")
        
        conn.close()
    
    return total_deleted


def delete_namespace_from_neo4j(namespace: str, dry_run: bool = False):
    """删除 Neo4j 中指定命名空间的所有节点和关系"""
    
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j@openspg")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "medicine")
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session(database=neo4j_database) as session:
        # 统计节点数
        count_query = f"""
            MATCH (n)
            WHERE ANY(l IN labels(n) WHERE l STARTS WITH '{namespace}.')
            RETURN count(n) as count
        """
        result = session.run(count_query)
        count = result.single()["count"]
        print(f"\nNeo4j 中有 {count} 个 {namespace}.* 节点")
        
        if count == 0 or dry_run:
            driver.close()
            return
        
        # 删除关系
        print(f"正在删除 Neo4j 中 {namespace}.* 的关系和节点...")
        
        delete_rels_query = f"""
            MATCH (n)-[r]-()
            WHERE ANY(l IN labels(n) WHERE l STARTS WITH '{namespace}.')
            DELETE r
            RETURN count(r) as deleted
        """
        result = session.run(delete_rels_query)
        deleted_rels = result.single()["deleted"]
        print(f"  已删除 {deleted_rels} 条关系")
        
        # 删除节点
        delete_nodes_query = f"""
            MATCH (n)
            WHERE ANY(l IN labels(n) WHERE l STARTS WITH '{namespace}.')
            DELETE n
            RETURN count(n) as deleted
        """
        result = session.run(delete_nodes_query)
        deleted_nodes = result.single()["deleted"]
        print(f"  已删除 {deleted_nodes} 个节点")
    
    driver.close()


def main():
    parser = argparse.ArgumentParser(description="智能清理指定命名空间的数据（缓存 + Neo4j）")
    parser.add_argument("--namespace", type=str, required=True,
                        help="要清理的命名空间（如 MedicineHigherMath, Pathophysiology）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计不实际删除")
    parser.add_argument("--cache-only", action="store_true",
                        help="只清理缓存，不删除 Neo4j 数据")
    parser.add_argument("--neo4j-only", action="store_true",
                        help="只删除 Neo4j 数据，不清理缓存")
    args = parser.parse_args()
    
    print(f"=== 智能清理命名空间: {args.namespace} ===")
    
    if args.dry_run:
        print("[DRY RUN 模式 - 只统计不删除]")
    
    # Step 1: 从 Neo4j 获取 Chunk 信息
    if not args.neo4j_only:
        print("\n[Step 1] 从 Neo4j 获取 Chunk 信息...")
        chunk_ids, contents = get_chunk_contents_from_neo4j(args.namespace)
        
        if chunk_ids or contents:
            # Step 2: 删除缓存
            print("\n[Step 2] 清理缓存条目...")
            deleted = delete_cache_by_keys(chunk_ids, contents, args.dry_run)
            print(f"缓存清理完成，共删除 {deleted} 条")
        else:
            print("未找到 Chunk 数据，跳过缓存清理")
    
    # Step 3: 删除 Neo4j 数据
    if not args.cache_only:
        print("\n[Step 3] 清理 Neo4j 数据...")
        delete_namespace_from_neo4j(args.namespace, args.dry_run)
    
    print("\n=== 清理完成 ===")


if __name__ == "__main__":
    main()
