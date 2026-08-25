# 🏗️ 建筑工地物料全生命周期智能调度 AI 智能体

> **为中国建筑国际集团「海之子」杯 AI 智能体挑战计划而设计**  
> 基于 FastAPI + SQLite + DeepSeek LLM 的一体化物料智能调度系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 项目简介

本系统面向建筑工地物料管理全生命周期，以 **“为人民建好房，为工友谋幸福”** 为初心，利用 AI 智能体技术实现物料需求的精准预测、供应商的多维评估、物流的多目标优化、库存的智能调拨、损耗的闭环管理以及 ESG 碳排核算。系统支持自然语言交互，用户可直接用对话方式驱动所有业务功能，大幅降低使用门槛，具备较强的落地实用性与产业推广价值。

---

## ✨ 核心功能

| 模块 | 功能描述 |
|------|----------|
| 📊 **智能驾驶舱** | 全局 KPI 监控、实时预警、物料短缺提醒、库存分布可视化 |
| 📈 **需求预测** | 结合项目阶段、历史消耗指数平滑、动态安全库存，给出未来 7/30/90 天需求与采购建议 |
| 🏆 **供应商评估** | 熵权法客观赋权，综合价格、准时率、质量、信用、服务五维评分，含绿色加分与风险预警 |
| 🚚 **物流调度** | 成本/碳排放多目标优化，生成三套方案（燃油、新能源、混合）并推荐最优解 |
| 📦 **库存优化** | 呆滞物料识别、安全库存动态调整、EOQ 经济订货批量、跨项目调拨建议 |
| 🔍 **损耗管理** | 损耗原因帕累托分析、高损耗时段识别、责任追溯与改善建议 |
| 🌿 **ESG 报告** | 运输碳排放核算、减排量统计、碳积分计算、政策补贴匹配 |
| 🤖 **AI 智能助手** | 基于 DeepSeek Function Calling，自然语言驱动所有功能，支持流式输出、上下文记忆、知识库检索 |

---

## 🏛️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 (index.html)                   │
│  HTML5 + CSS3 + JavaScript + ECharts                        │
│  - 响应式管理界面                                            │
│  - 可拖拽/缩放的 AI 对话面板                                 │
│  - 自动跳转与结果回填                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / JSON
┌──────────────────────────▼──────────────────────────────────┐
│                      后端 (main.py)                         │
│  FastAPI + Uvicorn                                          │
│  - RESTful API 路由                                         │
│  - AI 智能体（DeepSeek 集成）                               │
│  - 业务逻辑层（预测、供应商、物流等）                        │
│  - SQLite 数据库操作                                        │
│  - 知识库检索                                               │
│  - 数据导入（JSON/CSV）                                     │
└──────────┬───────────────────────────────┬──────────────────┘
           │                               │
┌──────────▼──────────┐        ┌───────────▼──────────┐
│   SQLite 数据库      │        │  DeepSeek LLM API    │
│   (ai_scheduler.db) │        │  (Function Calling)   │
└─────────────────────┘        └──────────────────────┘
```

**核心技术栈：**
- **后端**：Python + FastAPI + Uvicorn
- **前端**：原生 HTML/CSS/JavaScript + ECharts
- **数据库**：SQLite（内置，无需额外安装）
- **AI 引擎**：DeepSeek Chat API（`openai` 库兼容接口）
- **依赖管理**：`requirements.txt`

---

## 📁 项目结构

```
项目根目录
├── main.py                  # 后端主程序（单文件，内部按功能分区）
├── index.html               # 前端管理界面（位于 static 目录下）
├── static/                  # 静态文件目录（可放置其他前端资源）
├── knowledge/               # 知识库目录（可选，放置 .md 文件）
├── requirements.txt         # Python 依赖清单
├── .env.example             # 环境变量模板
├── .env                     # 实际环境变量（注意：不要提交到代码仓库）
├── .gitignore               # Git 忽略规则（禁止提交 .env、数据库等）
├── data_template.json       # JSON 数据导入模板
└── README.md                # 本说明文档
```

### 关于单文件 `main.py` 的分区说明
虽然所有后端代码集中在 `main.py` 中，但内部通过清晰的注释分节，逻辑边界明确，便于阅读与维护。其主要分区如下：

| 分区名称 | 行号范围（大约） | 功能说明 |
|----------|------------------|----------|
| 导入 & 配置 | 1~36 | 加载依赖、配置 DeepSeek 客户端、路径常量 |
| 数据库初始化与辅助函数 | 37~200 | 建表语句、默认数据插入、通用查询/执行函数 |
| 数据获取函数 | 201~260 | 封装数据库查询，供业务逻辑调用 |
| 车辆与碳排放计算 | 261~310 | 车辆参数、成本/碳排/时间计算 |
| 业务逻辑函数 | 311~600 | 需求预测、供应商评估、物流方案、库存优化、损耗管理、动态调整、ESG 报告、驾驶舱 |
| 数据导入逻辑 | 601~750 | JSON/CSV 导入处理与白名单校验 |
| API 路由 | 751~850 | 所有 RESTful 接口定义 |
| AI 智能体 | 851~1100 | 工具定义、工具执行、会话管理、同步/流式聊天接口 |
| 静态文件挂载 & 启动 | 1101~1110 | 挂载前端目录，启动 Uvicorn |

> 💡 后续若需团队协作或规模扩展，可按此分区拆分为独立模块（如 `database.py`、`services/`、`ai/` 等），当前单文件结构利于快速部署与演示。

---

## 🚀 快速开始

### 环境要求
- Python 3.8 或更高版本
- pip（建议使用虚拟环境）

### 1. 克隆或下载项目
```bash
git clone <你的仓库地址>
cd <项目目录>
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
复制 `.env.example` 为 `.env`，并填写你的 DeepSeek API Key：
```bash
cp .env.example .env
# 编辑 .env 文件，将 DEEPSEEK_API_KEY 替换为真实密钥
```
`.env.example` 内容如下：
```env
DEEPSEEK_API_KEY=请填写你的DeepSeek API Key
```

> ⚠️ **重要提示**：`.env` 文件包含敏感信息，**切勿提交到 Git 仓库**。请确保 `.gitignore` 中包含 `.env` 规则（当前 `.gitignore` 已包含）。

### 4. 初始化数据库（自动）
首次启动时，系统会自动创建 SQLite 数据库 `ai_scheduler.db` 并插入演示数据。无需手动建库。

### 5. 启动服务
在项目根目录下执行：
```bash
python main.py
```
或使用 Uvicorn 命令：
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
启动成功后，浏览器会自动打开 `http://127.0.0.1:8000`。

### 6. 访问系统
- **前端界面**：`http://127.0.0.1:8000/`
- **API 文档**：`http://127.0.0.1:8000/docs`（自动生成的 Swagger UI）

---

## 🖥️ 使用说明

### 界面导航
左侧边栏包含 9 个功能模块，点击即可切换：
1. **驾驶舱**：全局 KPI、预警清单、物料分布与趋势
2. **需求预测**：选择项目、物料、预测期限，查看预测详情
3. **供应商评估**：选择物料，查看供应商排名与推荐
4. **物流调度**：输入运输量和距离，获取三套方案对比
5. **库存优化**：查看健康度、呆滞物料与调拨建议
6. **损耗管理**：分析损耗原因与记录
7. **动态调整**：查看待处理事项与变更留痕
8. **ESG 报告**：碳排放统计、补贴匹配、月度趋势
9. **数据导入**：上传 JSON 或 CSV 文件更新系统数据

### AI 智能助手
- 点击右下角悬浮机器人按钮（🤖）打开对话面板
- 支持自然语言提问，例如：
  - “预测项目X未来7天混凝土需求”
  - “评估一下混凝土供应商”
  - “生成混凝土运输方案”
  - “库存有哪些呆滞物料？”
  - “这个月的碳排放是多少？”
- AI 助手会自动调用相应的工具函数，获取后端实时数据，并在回复后**自动跳转至相关页面并回填参数**。
- 支持多轮对话上下文记忆，可拖动和缩放对话窗口，支持清除会话。

---

## 📡 API 文档

启动服务后，访问 `http://127.0.0.1:8000/docs` 可查看完整 API 交互文档。主要端点如下：

| 方法 | 路径 | 功能说明 |
|------|------|----------|
| GET | `/api/dashboard` | 驾驶舱总览数据 |
| GET | `/api/predict` | 需求预测（参数：project_id, material_id, days） |
| GET | `/api/suppliers` | 供应商评估（参数：material_id） |
| GET | `/api/transport` | 物流调度方案（参数：material_id, quantity_ton, distance_km） |
| GET | `/api/inventory` | 库存优化分析 |
| GET | `/api/loss` | 损耗管理与分析 |
| GET | `/api/changes` | 动态调整记录与待办 |
| GET | `/api/esg` | ESG 报告数据 |
| GET | `/api/projects` | 获取项目列表 |
| GET | `/api/materials` | 获取物料列表 |
| GET | `/api/knowledge/search` | 知识库检索（参数：query） |
| POST | `/api/import/json` | 导入 JSON 完整数据包 |
| POST | `/api/import/csv` | 导入 CSV 文件（表单：file, data_type） |
| POST | `/api/ai_chat` | AI 同步对话（JSON：message, session_id?） |
| POST | `/api/ai_chat_stream` | AI 流式对话（SSE） |
| GET | `/api/ai/active_alerts` | 获取高危预警列表 |

### 示例请求（使用 curl）
```bash
# 需求预测
curl -X GET "http://127.0.0.1:8000/api/predict?project_id=P001&material_id=M001&days=7"

# AI 对话
curl -X POST "http://127.0.0.1:8000/api/ai_chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "预测项目X未来7天混凝土需求"}'
```

---

## 🤖 AI 智能体设计亮点

### 1. 工具调用（Function Calling）
系统将核心业务逻辑封装为 8 个 AI 工具，通过 `TOOLS` 列表定义 JSON Schema，并利用 DeepSeek 的 Function Calling 自动选择与调用。工具包括：
- `predict_demand`：需求预测
- `evaluate_supplier`：供应商评估
- `generate_transport_plan`：物流方案生成
- `optimize_inventory`：库存优化
- `loss_management`：损耗管理
- `dynamic_adjustment`：动态调整
- `esg_report`：ESG 报告
- `dashboard_overview`：驾驶舱总览

### 2. 动态上下文注入
每次对话前，后端自动从数据库读取当前可用的项目列表和物料列表，并注入到 System Prompt 中，使模型能够准确理解“项目X”对应 `P001` 等业务实体，避免幻觉。

### 3. 知识库检索
系统支持从 `knowledge/` 目录加载 Markdown 文件并分段存入内存。根据用户提问进行关键词匹配，返回最相关的知识片段作为参考上下文，增强回答的专业性。

### 4. 会话管理
- 使用内存字典 `sessions` 存储多轮对话历史，每个会话独立。
- 通过 `session_id` 识别用户，前端使用 `localStorage` 持久化会话 ID。
- 历史消息保留最近 10 轮，防止上下文过长。

### 5. 流式输出（备用）
提供 `/api/ai_chat_stream` 接口，采用 Server-Sent Events (SSE) 方式，支持无工具调用时的流式回复，提升交互体验。

### 6. 人机协同与纠偏
- 当 AI 调用工具后，后端会执行实际业务逻辑并返回结果，AI 再基于结果生成自然语言回复。
- 若 AI 判断需要页面跳转（如预测、评估），还会返回 `action` 字段，前端自动切换页面并回填参数，实现“对话即操作”。
- 若 API 调用失败，系统提供降级处理：根据关键词进行简单规则匹配，确保基本功能可用。

---

## 📥 数据导入

### JSON 导入
- 支持上传完整数据包（包含 `projects`、`materials`、`inventory` 等所有表）。
- 模板文件：`data_template.json`（项目根目录）。
- 示例：在“数据导入”页面点击“上传 JSON 数据包”，选择文件即可。

### CSV 导入
- 支持按单表类型导入 CSV 文件，例如 `projects.csv`、`materials.csv` 等。
- 文件名需包含类型关键字（如 `project`、`material` 等），系统会自动识别。
- CSV 第一行必须为列名，且列名与数据库字段名完全一致（参考 `data_template.json` 中的字段）。
- 系统使用白名单校验列名，防止 SQL 注入。

---

## ❓ 常见问题（FAQ）

### Q1: 启动时提示 `未设置 DEEPSEEK_API_KEY`？
**A:** 请确认项目根目录下存在 `.env` 文件，并按照 `.env.example` 格式正确填写 `DEEPSEEK_API_KEY`。

### Q2: 数据库文件在哪里？如何重置数据？
**A:** 数据库文件为 `ai_scheduler.db`，位于项目根目录。如需重置为演示数据，删除该文件后重启服务即可自动重建并插入默认数据。

### Q3: 前端样式没有加载或接口 404？
**A:** 请确保 `index.html` 位于 `static/` 目录下，且 `main.py` 中静态文件挂载路径配置正确（默认 `app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))`）。

### Q4: AI 助手的回复有时不准确？
**A:** 可尝试调整 `temperature` 参数（当前为 0.2，较低有利于确定性）。若需要更精准，可扩展知识库内容或优化 System Prompt。

### Q5: 如何添加自定义知识库？
**A:** 在项目根目录创建 `knowledge/` 文件夹，放入 `.md` 文件。每个段落（以空行分隔）将被视为独立知识片段，启动时自动加载。

---

## 🔮 后续规划

- **多智能体协作**：构建预测 Agent、采购 Agent、物流 Agent 的协同决策系统。
- **BIM 数据接入**：结合施工进度计划，实现更精准的物料需求预测。
- **移动端适配**：开发小程序或移动端 H5，方便工地现场使用。
- **边缘计算部署**：支持在工地局域网内离线运行，降低网络依赖。
- **实时数据同步**：对接 IoT 设备（地磅、门禁）自动采集库存与运输数据。

---

## 🧑‍💻 开发者

- **开发者**：王超然

---