"""
MinIO Client Module for KAG

提供 MinIO 对象存储的客户端封装，用于下载教材文件。
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MinIO 配置（与 Java 后端一致）
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minio@openspg")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "textbooks")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

# MinIO 客户端（延迟初始化）
_minio_client = None


def get_minio_client():
    """获取 MinIO 客户端单例"""
    global _minio_client
    if _minio_client is None:
        try:
            from minio import Minio
            _minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_SECURE
            )
            logger.info(f"MinIO 客户端已初始化: {MINIO_ENDPOINT}")
        except ImportError:
            logger.warning("minio 包未安装，请运行: pip install minio")
            return None
        except Exception as e:
            logger.error(f"MinIO 客户端初始化失败: {e}")
            return None
    return _minio_client


def download_from_minio(object_name: str, local_path: str) -> bool:
    """
    从 MinIO 下载文件到本地。
    
    Args:
        object_name: MinIO 中的对象路径（如 "1/content.md"）
        local_path: 本地保存路径
        
    Returns:
        下载是否成功
    """
    client = get_minio_client()
    if client is None:
        logger.error("MinIO 客户端未初始化")
        return False
    
    try:
        # 确保目标目录存在
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        client.fget_object(MINIO_BUCKET, object_name, local_path)
        logger.info(f"从 MinIO 下载成功: {MINIO_BUCKET}/{object_name} -> {local_path}")
        return True
    except Exception as e:
        logger.error(f"从 MinIO 下载失败: {object_name} - {e}")
        return False


def object_exists(object_name: str) -> bool:
    """检查对象是否存在于 MinIO"""
    client = get_minio_client()
    if client is None:
        return False
    
    try:
        client.stat_object(MINIO_BUCKET, object_name)
        return True
    except Exception:
        return False


def download_textbook_files(textbook_id: int, temp_dir: str) -> dict:
    """
    下载教材相关的所有 MD 文件。
    
    Args:
        textbook_id: 教材 ID
        temp_dir: 临时目录路径
        
    Returns:
        包含本地文件路径的字典，如:
        {
            "content_path": "/tmp/1/content.md",
            "catalog_path": "/tmp/1/catalog.md"  # 可能为 None
        }
    """
    result = {
        "content_path": None,
        "catalog_path": None
    }
    
    # 尝试下载内容文件
    content_object = f"{textbook_id}/content.md"
    content_local = os.path.join(temp_dir, str(textbook_id), "content.md")
    if download_from_minio(content_object, content_local):
        result["content_path"] = content_local
    
    # 尝试下载目录文件（可选）
    catalog_object = f"{textbook_id}/catalog.md"
    catalog_local = os.path.join(temp_dir, str(textbook_id), "catalog.md")
    if object_exists(catalog_object):
        if download_from_minio(catalog_object, catalog_local):
            result["catalog_path"] = catalog_local
    
    return result


def parse_minio_path(minio_path: str) -> Optional[str]:
    """
    解析 MinIO 路径，提取对象名。
    
    Args:
        minio_path: MinIO 路径，如 "textbooks/1/content.md"
        
    Returns:
        对象名，如 "1/content.md"
    """
    if not minio_path:
        return None
    
    # 移除桶名前缀
    if minio_path.startswith(MINIO_BUCKET + "/"):
        return minio_path[len(MINIO_BUCKET) + 1:]
    
    return minio_path
