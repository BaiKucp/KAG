# KAG 知识增强生成服务 - 技术操作手册

本文档旨在指导操作人员完成 MedExamEasy 系统中 KAG（Knowledge Augmented Generation）模块的部署、初始化及日常使用。

## 1. 环境准备

确保您的服务器或本地机器满足以下要求：

*   **操作系统**: Windows 10/11 (推荐使用 WSL2) 或 Linux/macOS
*   **Docker**: 已安装并启动 Docker Desktop
*   **Python**: 3.10 或更高版本
*   **Navicat/数据库工具**: 用于必要的数据校验

## 2. 启动基础服务 (OpenSPG)

KAG 依赖 OpenSPG 引擎（包含图存储 Neo4j、搜索引擎、MySQL 等）。

1.  **打开终端 (PowerShell)**，进入 KAG 部署目录：
    ```powershell
    cd "E:\AIIA Project\MedExamEasy\DevOps\kag"
    ```
2.  **启动容器服务**：
    ```powershell
    docker-compose up -d
    ```
3.  **验证启动**：
    *   访问 OpenSPG 控制台: [http://127.0.0.1:8887](http://127.0.0.1:8887)
    *   默认账号: `openspg`
    *   默认密码: `openspg@kag`
    *   Neo4j 控制台: [http://127.0.0.1:7474](http://127.0.0.1:7474) (账号 `neo4j` / 密码 `neo4j@openspg`)

## 3. 创建 OpenSPG 项目 (初始化)

在首次使用 KAG 功能前，需要初始化 OpenSPG 项目并提交 schema。可以使用命令行工具 `knext` 快速完成。

1.  **打开终端 (PowerShell)**，确保当前目录为 `E:\AIIA Project\MedExamEasy\DevOps\kag`。

2.  **安装依赖 (如果尚未安装)**：
    由于源码已移动到 `KAG` 子目录，请执行：
    ```powershell
    cd KAG
    pip install -e .
    cd ..
    ```

3.  **创建/注册项目**：
    由于我们重构了目录，请先进入 `KAG` 子目录：
    ```powershell
    cd KAG
    knext project create --config_path ./Medicine/kag_config.yaml
    ```
    > **提示**：该命令会自动读取 `Medicine/kag_config.yaml` 中的配置，并在服务端创建对应的项目。

4.  **提交 Schema**：
    进入项目目录并提交 schema 定义：
    ```powershell
    cd Medicine
    knext schema commit
    ```
    > **成功标志**：看到 "Schema is successfully committed" 或 "There is no diff..." 即表示成功。

## 4. 启动 KAG API 服务

**注意**：如果您在第 2 步执行 `docker-compose up -d` 时已经启动了 `kag-api` 容器（默认配置包含），则 KAG API 服务已经自动运行在 `http://127.0.0.1:8000`。**您可以跳过本节，直接进行前端操作。**

如果您希望**手动在本地运行** API 服务（例如用于调试或 Docker 中未包含该服务），请执行以下步骤：

1.  **打开终端 (PowerShell)**。
2.  **进入服务目录**：
    ```powershell
    cd "E:\AIIA Project\MedExamEasy\DevOps\kag\KAG\Medicine\solver"
    ```
3.  **安装依赖 (首次运行)**：
    如果是第一次运行，请参考 `DevOps/kag/README_cn.md` 安装必要的 Python 包：
    ```powershell
    pip install -r requirements.txt
    pip install -e ../..  # 安装 KAG 包
    ```
4.  **启动 API**：
    Windows 用户可以直接运行目录下的启动脚本：
    ```powershell
    start_api.bat
    ```
    或者手动运行命令：
    ```powershell
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    ```
5.  **验证启动**：
    *   看到类似 `Uvicorn running on http://0.0.0.0:8000` 的日志即表示启动成功。

## 5. 前端操作指南 (知识抽取)

服务启动后，即可在前端界面进行教材的知识图谱抽取操作。

1.  **登录系统**：
    *   访问 MedExamEasy 管理后台 (例如 [http://localhost:3000](http://localhost:3000))。
2.  **进入教材管理**：
    *   点击左侧菜单 **"教材管理"**。
3.  **上传教材**：
    *   如果列表为空，点击右上角 **"上传教材"**，上传 PDF 或 Markdown 格式的医学教材。
4.  **开始抽取**：
    *   在教材列表中，找到目标教材。
    *   点击操作栏的 **"知识抽取"** 图标（通常是一个图谱或魔法棒图标）。
5.  **监控进度**：
    *   系统会弹出 **"知识图谱抽取进度"** 弹窗（如下图所示）。
    *   您可以看到：
        *   **Total Chunks**: 总切片数
        *   **Processed**: 已处理数量
        *   **Extracted Knowledge**: 已抽取知识点数
        *   **Time Remaining**: 预计剩余时间
    *   **注意**：抽取过程可能较长（视教材大小而定），请保持网络连接，不要关闭 API 服务窗口。

## 6. 常见问题排查

*   **Q: 启动 API 报错 "kag_config.yaml not found"**
    *   A: 确保在正确的工作目录 (`Medicine/solver`) 下运行命令，或检查 `api_server.py` 中的路径配置。

*   **Q: 抽取进度一直为 0%**
    *   A:检查 KAG API 控制台日志是否有报错。常见原因是 OpenSPG 服务未启动 (`Connection refused`) 或 项目 ID 不匹配。

*   **Q: 提示 "Table doesn't exist"**
    *   A: 检查 Java 后端是否已执行最新的 SQL 迁移脚本 (`init.sql`)。

---
**MedExamEasy 开发团队**
2026-01-24
