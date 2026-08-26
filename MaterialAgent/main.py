import os
import random
import threading
import time
import webbrowser
import json
import sqlite3
import csv
import io
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask
app = Flask(__name__)
load_dotenv()

# DeepSeek 客户端配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量，请检查 .env 文件")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), "ai_scheduler.db")

# CSV 导入允许的列名白名单（防止 SQL 注入）
ALLOWED_COLUMNS = {
    "projects": ["id", "name", "short_name", "location", "stage", "progress", "manager", "workers", "building_area", "type"],
    "materials": ["id", "name", "unit", "safety_stock", "weight_per_unit", "est_price", "plan_7d", "plan_30d", "plan_90d", "lead_time_days", "loss_rate", "category", "storage_condition"],
    "inventory": ["project_id", "material_id", "current_stock", "last_used_days", "rotate_days", "status", "batch_no", "received_date"],
    "suppliers": ["id", "name", "material_id", "price", "market_price", "on_time_rate", "quality_rate", "credit_score", "violations", "location", "service_level", "cooperation_years", "green_certified", "capacity"],
    "transport_records": ["id", "date", "vehicle", "distance_km", "load_ton", "status"],
    "loss_records": ["id", "date", "project_id", "material_id", "quantity", "reason", "description", "loss_amount"],
    "alerts": ["id", "type", "level", "title", "detail", "time"],
    "change_logs": ["id", "time", "event", "reason", "adjustment", "confirm_by"],
}
# ==================== 自动打开浏览器 ====================
def open_browser():
    time.sleep(0.5)
    webbrowser.open("http://127.0.0.1:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=open_browser, daemon=True).start()
    init_db()
    load_knowledge_base()
    yield

app = FastAPI(title="建筑工地物料全生命周期智能调度 AI 智能体", lifespan=lifespan)

# ==================== 知识库加载与检索 ====================
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
KNOWLEDGE_BASE = []

def load_knowledge_base():
    global KNOWLEDGE_BASE
    if not KNOWLEDGE_DIR.exists():
        print(f"警告：知识库目录 {KNOWLEDGE_DIR} 不存在")
        return
    kb = []
    for md_file in KNOWLEDGE_DIR.glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for para in paragraphs:
                kb.append({"file": md_file.name, "content": para})
        except Exception as e:
            print(f"读取知识库文件 {md_file} 失败: {e}")
    KNOWLEDGE_BASE = kb
    print(f"知识库加载完成，共 {len(kb)} 个片段")

def retrieve_knowledge(query, top_k=3):
    if not KNOWLEDGE_BASE or not query:
        return []
    import re
    tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    if not tokens:
        return []
    scored = []
    for item in KNOWLEDGE_BASE:
        content_lower = item["content"].lower()
        score = sum(1 for token in tokens if token in content_lower)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]

# ==================== 数据库初始化与辅助函数 ====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        short_name TEXT,
        location TEXT,
        stage TEXT,
        progress INTEGER,
        manager TEXT,
        workers INTEGER,
        building_area REAL,
        type TEXT
    );
    CREATE TABLE IF NOT EXISTS materials (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        unit TEXT,
        safety_stock REAL,
        weight_per_unit REAL,
        est_price REAL,
        plan_7d REAL,
        plan_30d REAL,
        plan_90d REAL,
        lead_time_days INTEGER,
        loss_rate REAL,
        category TEXT,
        storage_condition TEXT
    );
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT,
        material_id TEXT,
        current_stock REAL,
        last_used_days INTEGER,
        rotate_days INTEGER,
        status TEXT,
        batch_no TEXT,
        received_date TEXT,
        FOREIGN KEY(project_id) REFERENCES projects(id),
        FOREIGN KEY(material_id) REFERENCES materials(id)
    );
    CREATE TABLE IF NOT EXISTS suppliers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        material_id TEXT,
        price REAL,
        market_price REAL,
        on_time_rate REAL,
        quality_rate REAL,
        credit_score INTEGER,
        violations INTEGER,
        location TEXT,
        service_level REAL,
        cooperation_years INTEGER,
        green_certified INTEGER,
        capacity TEXT,
        FOREIGN KEY(material_id) REFERENCES materials(id)
    );
    CREATE TABLE IF NOT EXISTS transport_records (
        id TEXT PRIMARY KEY,
        date TEXT,
        vehicle TEXT,
        distance_km REAL,
        load_ton REAL,
        status TEXT
    );
    CREATE TABLE IF NOT EXISTS loss_records (
        id TEXT PRIMARY KEY,
        date TEXT,
        project_id TEXT,
        material_id TEXT,
        quantity REAL,
        reason TEXT,
        description TEXT,
        loss_amount REAL,
        FOREIGN KEY(project_id) REFERENCES projects(id),
        FOREIGN KEY(material_id) REFERENCES materials(id)
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        type TEXT,
        level TEXT,
        title TEXT,
        detail TEXT,
        time TEXT
    );
    CREATE TABLE IF NOT EXISTS change_logs (
        id TEXT PRIMARY KEY,
        time TEXT,
        event TEXT,
        reason TEXT,
        adjustment TEXT,
        confirm_by TEXT
    );
    """)
    conn.commit()
    # 清空所有业务表，确保每次启动恢复默认数据
    tables = [
        "inventory", "suppliers", "loss_records", "transport_records",
        "alerts", "change_logs", "materials", "projects"
    ]
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
    conn.commit()

    # 插入默认演示数据（无条件执行）
    insert_default_data(conn)
    conn.close()

def insert_default_data(conn):
    """插入默认演示数据"""
    cursor = conn.cursor()
    # 项目
    projects = [
        ("P001", "杭州·西湖云锦项目", "项目X", "杭州市西湖区", "主体结构施工", 68, "张工", 156, 120000, "住宅"),
        ("P002", "杭州·滨江智造谷项目", "项目Y", "杭州市滨江区", "砌筑工程", 45, "李工", 120, 85000, "工业厂房"),
        ("P003", "杭州·余杭科创园项目", "项目Z", "杭州市余杭区", "基础施工", 25, "王工", 98, 65000, "办公园区"),
    ]
    cursor.executemany("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?)", projects)
    # 物料
    materials = [
        ("M001", "C30商品混凝土", "m³", 80, 2.4, 380, 180, 720, 2000, 3, 0.02, "结构材料", "搅拌站直供"),
        ("M002", "φ48盘扣式钢管", "根", 200, 0.03, 120, 300, 1200, 3500, 5, 0.01, "周转材料", "露天堆场"),
        ("M003", "加气混凝土砌块", "块", 3000, 0.01, 1.5, 2200, 9000, 26000, 4, 0.03, "砌筑材料", "防雨棚"),
        ("M004", "HRB400螺纹钢", "吨", 25, 1.0, 4200, 60, 260, 750, 7, 0.015, "结构材料", "室内仓库"),
        ("M005", "42.5水泥", "吨", 30, 1.0, 520, 45, 190, 560, 3, 0.005, "胶凝材料", "防潮仓库"),
    ]
    cursor.executemany("INSERT INTO materials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", materials)
    # 库存
    inventory = [
        ("P001", "M001", 35, 2, 3, "normal", "B20240128-01", "2024-01-25"),
        ("P001", "M002", 320, 45, 60, "deadstock", "B20240110-03", "2024-01-10"),
        ("P001", "M004", 18, 1, 5, "normal", "B20240126-02", "2024-01-26"),
        ("P002", "M003", 1200, 1, 4, "normal", "B20240123-01", "2024-01-23"),
        ("P002", "M005", 22, 8, 15, "warning", "B20240115-04", "2024-01-15"),
        ("P003", "M001", 10, 0, 2, "warning", "B20240127-01", "2024-01-27"),
        ("P003", "M004", 8, 3, 7, "normal", "B20240122-02", "2024-01-22"),
    ]
    cursor.executemany("INSERT INTO inventory (project_id, material_id, current_stock, last_used_days, rotate_days, status, batch_no, received_date) VALUES (?,?,?,?,?,?,?,?)", inventory)
    # 供应商
    suppliers = [
        ("S001", "浙江建工建材有限公司", "M001", 380, 390, 0.98, 0.996, 92, 0, "杭州·萧山", 0.95, 5, 1, "5000m³/月"),
        ("S002", "杭州鼎盛混凝土公司", "M001", 370, 390, 0.90, 0.96, 82, 1, "杭州·富阳", 0.85, 3, 0, "3000m³/月"),
        ("S003", "浙江海天建设材料", "M001", 365, 390, 0.85, 0.94, 68, 2, "杭州·临安", 0.75, 2, 0, "2000m³/月"),
        ("S004", "杭州中天金属材料", "M002", 118, 125, 0.95, 0.98, 88, 0, "杭州·余杭", 0.92, 4, 1, "5000根/月"),
        ("S005", "浙江绿源建材科技", "M003", 1.48, 1.55, 0.97, 0.99, 95, 0, "杭州·德清", 0.94, 6, 1, "100000块/月"),
        ("S006", "杭州钢铁贸易有限公司", "M004", 4150, 4300, 0.93, 0.97, 86, 1, "杭州·拱墅", 0.90, 3, 0, "800吨/月"),
        ("S007", "浙江南方水泥集团", "M005", 515, 535, 0.96, 0.99, 91, 0, "杭州·建德", 0.93, 5, 1, "2000吨/月"),
    ]
    cursor.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", suppliers)
    # 运输记录
    transport_records = [
        ("T001", "2024-01-05", "燃油货车", 210, 18, "已完成"),
        ("T002", "2024-01-08", "新能源货车", 160, 14, "已完成"),
        ("T003", "2024-01-12", "燃油货车", 185, 16, "已完成"),
        ("T004", "2024-01-15", "新能源货车", 230, 13, "已完成"),
        ("T005", "2024-01-18", "混合动力", 195, 17, "已完成"),
        ("T006", "2024-01-22", "新能源货车", 145, 15, "运输中"),
        ("T007", "2024-01-25", "燃油货车", 200, 19, "运输中"),
    ]
    cursor.executemany("INSERT INTO transport_records VALUES (?,?,?,?,?,?)", transport_records)
    # 损耗记录
    loss_records = [
        ("L001", "2024-01-06", "P001", "M004", 0.3, "人为浪费", "钢筋短少，夜间看管不到位", 1260),
        ("L002", "2024-01-09", "P001", "M001", 2.0, "正常损耗", "浇筑过程溢料", 760),
        ("L003", "2024-01-14", "P002", "M003", 45, "人为浪费", "搬运破损", 67.5),
        ("L004", "2024-01-20", "P002", "M005", 1.5, "丢失损毁", "露天堆放，防护不到位", 780),
    ]
    cursor.executemany("INSERT INTO loss_records VALUES (?,?,?,?,?,?,?,?)", loss_records)
    # 预警
    alerts = [
        ("A001", "库存预警", "high", "项目X混凝土库存不足", "当前库存35m³，安全库存80m³，7天内预计缺口145m³", "10分钟前"),
        ("A002", "供应商风险", "medium", "供应商C履约风险上升", "近3月准时率降至85%，有2次逾期记录", "1小时前"),
        ("A003", "运输异常", "high", "T006运输车辆偏离路线", "新能源货车偏离规划路线约15km，正在联系司机确认", "2小时前"),
        ("A004", "呆滞物料", "medium", "项目X钢管闲置45天", "320根φ48钢管长期闲置，占压资金约3.8万元", "3小时前"),
        ("A005", "损耗异常", "low", "项目Y砌块损耗偏高", "本周砌块损耗率0.8%，高于正常水平0.3%", "5小时前"),
        ("A006", "施工变更", "medium", "项目Y砌筑工程提前", "进度提前3天，需重新启动物料采购计划", "昨天"),
    ]
    cursor.executemany("INSERT INTO alerts VALUES (?,?,?,?,?,?)", alerts)
    # 变更日志
    change_logs = [
        ("C001", "2024-01-28 09:32", "项目Y砌筑工程提前3天启动", "施工进度优化，班组调配到位", "砌块采购计划从1月28日提前至1月25日", "项目经理·李工"),
        ("C002", "2024-01-27 14:15", "供应商A混凝土报价下调2%", "原材料价格波动", "更新采购基准价至380元/m³", "采购部·王主管"),
        ("C003", "2024-01-26 11:08", "项目X钢管盘点差异-8根", "夜间看管不到位", "加强安保，重点复核夜间领用", "材料员·赵工"),
    ]
    cursor.executemany("INSERT INTO change_logs VALUES (?,?,?,?,?,?)", change_logs)
    conn.commit()

def db_fetch_all(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def db_fetch_one(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def db_execute(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def db_executemany(query, params_list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany(query, params_list)
    conn.commit()
    conn.close()

# ==================== 数据获取函数（数据库版） ====================
def get_projects():
    return db_fetch_all("SELECT * FROM projects")

def get_materials():
    return db_fetch_all("SELECT * FROM materials")

def get_project(project_id):
    return db_fetch_one("SELECT * FROM projects WHERE id=?", (project_id,))

def get_material(material_id):
    return db_fetch_one("SELECT * FROM materials WHERE id=?", (material_id,))

def get_inventory(project_id, material_id):
    return db_fetch_one("SELECT * FROM inventory WHERE project_id=? AND material_id=?", (project_id, material_id))

def get_suppliers_for_material(material_id):
    return db_fetch_all("SELECT * FROM suppliers WHERE material_id=?", (material_id,))

def get_transport_records():
    return db_fetch_all("SELECT * FROM transport_records")

def get_loss_records():
    return db_fetch_all("SELECT * FROM loss_records")

def get_alerts():
    return db_fetch_all("SELECT * FROM alerts")

def get_change_logs():
    return db_fetch_all("SELECT * FROM change_logs")

# ==================== 车辆数据（保留硬编码） ====================
VEHICLES = [
    {"type": "燃油货车", "type_id": "fuel", "cost_per_ton_km": 2.5, "emission_factor": 0.18, "capacity_ton": 20,
     "avg_speed_kmh": 58, "count": 12, "load_unload_hours": 2},
    {"type": "新能源货车", "type_id": "ev", "cost_per_ton_km": 3.0, "emission_factor": 0.05, "capacity_ton": 15,
     "avg_speed_kmh": 52, "count": 6, "load_unload_hours": 2.5},
    {"type": "混合动力", "type_id": "hybrid", "cost_per_ton_km": 2.7, "emission_factor": 0.11, "capacity_ton": 18,
     "avg_speed_kmh": 55, "count": 4, "load_unload_hours": 2.2},
]

def find_vehicle(vehicle_identifier):
    """根据 type_id 或 type 中文名查找车辆"""
    for v in VEHICLES:
        if v["type_id"] == vehicle_identifier or v["type"] == vehicle_identifier:
            return v
    return None

def calc_carbon(distance_km, vehicle_identifier, load_ton):
    vehicle = find_vehicle(vehicle_identifier)
    if vehicle:
        return round(distance_km * vehicle["emission_factor"] * load_ton, 2)
    return 0

def calc_cost(distance_km, vehicle_identifier, load_ton):
    vehicle = find_vehicle(vehicle_identifier)
    if vehicle:
        return round(distance_km * vehicle["cost_per_ton_km"] * load_ton, 2)
    return 0

def calc_time(distance_km, vehicle_identifier):
    vehicle = find_vehicle(vehicle_identifier)
    if vehicle:
        travel_time = distance_km / vehicle["avg_speed_kmh"]
        load_unload_time = vehicle["load_unload_hours"]
        return round(travel_time + load_unload_time, 2)
    return 0

# ==================== 业务逻辑函数（数据库版） ====================
def predict_demand_logic(project_id, material_id, days=7):
    project = get_project(project_id)
    material = get_material(material_id)
    inv = get_inventory(project_id, material_id) or {"current_stock": 0}
    current = inv["current_stock"]
    safety = material["safety_stock"]
    lead_time = material["lead_time_days"]

    plan_key = f"plan_{days}d" if days in [7, 30, 90] else "plan_7d"
    planned_base = material[plan_key]

    # 根据项目阶段确定进度系数
    progress_factor = 1.0
    if project["stage"] == "主体结构施工":
        progress_factor = 1.2
    elif project["stage"] == "基础施工":
        progress_factor = 0.8
    elif project["stage"] == "砌筑工程":
        progress_factor = 0.9

    # 计算基础日均用量（未来7天计划用量 ÷ 7 天 * 阶段系数）
    base_daily = round(planned_base * progress_factor / 7, 2)

    # 生成确定性历史消耗（基于基础日均用量，加上固定的周期性波动）
    history = []
    for i in range(14, -1, -1):
        day = datetime.now() - timedelta(days=i)
        # 使用固定系数 (1 + 0.2 * (i % 7) / 7) 模拟周期性波动，无随机性
        usage = round(base_daily * (1 + 0.2 * (i % 7) / 7), 2)
        history.append({"date": day.strftime("%m-%d"), "usage": usage})

    # 使用简单指数平滑计算日均消耗
    alpha = 0.3
    smoothed = history[0]["usage"]
    for h in history[1:]:
        smoothed = alpha * h["usage"] + (1 - alpha) * smoothed
    avg_daily_usage = round(smoothed, 2)

    # 动态安全库存：需求标准差 × 提前期^0.5 × 服务水平系数
    recent_usages = [h["usage"] for h in history[-14:]]
    mean_usage = sum(recent_usages) / len(recent_usages)
    variance = sum((u - mean_usage) ** 2 for u in recent_usages) / len(recent_usages)
    std_dev = variance ** 0.5
    z = 1.65  # 95%服务水平
    dynamic_safety = round(z * std_dev * (lead_time ** 0.5), 2)
    safety = max(safety, dynamic_safety)

    # 最终计划用量 = 平滑日均用量 × 预测天数
    planned = round(avg_daily_usage * days, 2)

    # 计算缺口
    gap = planned + safety - current
    gap = max(0, round(gap, 2))

    if gap > 0:
        suggestion = (f"当前库存{current}{material['unit']}，安全库存需{safety}{material['unit']}，"
                      f"未来{days}天预计用量{planned}{material['unit']}，存在缺口{gap}{material['unit']}。"
                      f"建议在{lead_time}天内完成采购下单，采购量建议为{gap}。")
        status = "shortage"
    else:
        suggestion = f"库存充足，当前库存{current}{material['unit']}，安全库存{safety}{material['unit']}，无需采购。"
        status = "sufficient"

    return {
        "project": project,
        "material": material,
        "current_stock": current,
        "safety_stock": round(safety, 2),
        "planned_usage": planned,
        "avg_daily_usage": avg_daily_usage,
        "gap": gap,
        "lead_time_days": lead_time,
        "history": history,
        "suggestion": suggestion,
        "status": status,
        "progress_factor": progress_factor
    }

def entropy_weight(matrix):
    """
    熵权法计算各指标权重
    输入：二维列表/数组，每行一个供应商，每列一个指标（已经正向化且越大越好）
    输出：各指标权重列表
    """
    matrix = np.array(matrix, dtype=float)
    # 1. 归一化（min-max标准化，避免除零）
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # 防止常数列导致除零
    norm = (matrix - mins) / ranges

    # 2. 计算熵值
    m, n = norm.shape
    # 避免概率为0时log(0)错误
    p = norm / norm.sum(axis=0, keepdims=True)
    p = np.where(p == 0, 1e-10, p)
    e = -1.0 / np.log(m) * np.sum(p * np.log(p), axis=0)
    d = 1 - e
    weights = d / d.sum()
    return weights.tolist()


def evaluate_supplier_logic(material_id):
    material = get_material(material_id)
    suppliers_list = get_suppliers_for_material(material_id)
    if not suppliers_list:
        return {"material": material, "suppliers": [], "best": None, "note": "暂无该物料供应商数据"}

    # ---------- 熵权法计算权重 ----------
    # 构建指标矩阵（正向化）
    # 价格：用 (最高价 - 当前价) 使其越大越好
    max_price = max(s["price"] for s in suppliers_list)
    # 其他指标均已是正向（准时率、合格率、信用分、服务水平），但需统一量纲（都放大到0-100）
    matrix = []
    for s in suppliers_list:
        price_positive = max_price - s["price"]  # 价格正向化，越大越好
        on_time = s["on_time_rate"] * 100
        quality = s["quality_rate"] * 100
        credit = s["credit_score"]
        service = s["service_level"] * 100
        matrix.append([price_positive, on_time, quality, credit, service])

    # 如果供应商数量过少（小于3个），熵权法可能不稳定，回退到固定权重
    if len(suppliers_list) >= 3:
        try:
            weights = entropy_weight(matrix)
        except Exception:
            # 出错时使用固定权重（与原逻辑一致）
            weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    else:
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]

    # 对每个供应商进行标准化（用于计算加权得分）
    # 这里我们复用已经构建好的 matrix，对其进行标准化，然后加权
    np_matrix = np.array(matrix, dtype=float)
    mins = np_matrix.min(axis=0)
    maxs = np_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    norm_matrix = (np_matrix - mins) / ranges

    # ---------- 计算每个供应商得分 ----------
    results = []
    for idx, s in enumerate(suppliers_list):
        # 基础分（权重*标准化值，再乘100转换为百分制）
        score_without_green = np.sum(weights * norm_matrix[idx]) * 100
        green_bonus = 5 if s["green_certified"] else 0
        total = round(score_without_green + green_bonus, 2)

        # 保留原有各个维度的评分展示
        min_price = min(s2["price"] for s2 in suppliers_list)
        price_score = round((min_price / s["price"]) * 100, 1) if s["price"] > 0 else 0
        perf_score = round(s["on_time_rate"] * 100, 1)
        quality_score = round(s["quality_rate"] * 100, 1)
        credit_score = s["credit_score"]
        service_score = round(s["service_level"] * 100, 1)

        # 评级逻辑不变（阈值使用百分制）
        if total >= 90:
            rank = "优质"
        elif total >= 80:
            rank = "良好"
        elif total >= 70:
            rank = "合格"
        elif total >= 60:
            rank = "观察"
        else:
            rank = "黑名单"

        # 风险信号（原有逻辑）
        price_deviation = round((s["price"] - s["market_price"]) / s["market_price"] * 100, 1) if s["market_price"] else 0
        risk_signals = []
        if s["on_time_rate"] < 0.90:
            risk_signals.append(f"准时率偏低({s['on_time_rate']*100:.0f}%)")
        if s["quality_rate"] < 0.95:
            risk_signals.append(f"合格率偏低({s['quality_rate']*100:.0f}%)")
        if s["violations"] >= 2:
            risk_signals.append(f"违规记录{s['violations']}次")
        if price_deviation < -5:
            risk_signals.append("报价明显低于市场价，存在低价中标风险")
        if s["cooperation_years"] < 1:
            risk_signals.append("合作年限不足1年，需加强考察")

        results.append({
            **s,
            "price_score": price_score,
            "perf_score": perf_score,
            "quality_score": quality_score,
            "credit_score": credit_score,
            "service_score": service_score,
            "green_bonus": green_bonus,
            "total_score": total,
            "rank": rank,
            "price_deviation": price_deviation,
            "risk_signals": risk_signals
        })

    # 排序和推荐
    results.sort(key=lambda x: x["total_score"], reverse=True)
    best = results[0]
    lowest = min(results, key=lambda x: x["price"])

    note = ""
    if best["id"] != lowest["id"]:
        note = f"最低价供应商为{lowest['name']}（报价{lowest['price']}元），但综合评分仅{lowest['total_score']}分，可能存在风险。推荐选择{best['name']}，综合评分{best['total_score']}分。"

    # 返回时带上各指标权重（供前端展示，可选）
    return {
        "material": material,
        "suppliers": results,
        "best": best,
        "note": note,
        "weights": weights  # 新增：指标权重（顺序对应 [价格, 准时率, 合格率, 信用分, 服务水平]）
    }

def generate_transport_plan_logic(material_id, quantity_ton, distance_km=200, weight_cost=0.5, weight_carbon=0.5):
    material = get_material(material_id)
    if quantity_ton <= 0:
        return {
            "material": material,
            "quantity_ton": 0,
            "distance_km": distance_km,
            "plans": [],
            "weight_cost": weight_cost,
            "weight_carbon": weight_carbon
        }

    # 方案A：燃油
    fuel_vehicle = next(v for v in VEHICLES if v["type_id"] == "fuel")
    fuel_count = max(1, -(-quantity_ton // fuel_vehicle["capacity_ton"]))
    cost_a = calc_cost(distance_km, "fuel", quantity_ton)
    carbon_a = calc_carbon(distance_km, "fuel", quantity_ton)
    travel_time_a = distance_km / fuel_vehicle["avg_speed_kmh"]
    total_time_a = travel_time_a + fuel_vehicle["load_unload_hours"]
    plans_a = {
        "name": "方案A · 成本优先",
        "vehicle_type": "燃油货车",
        "vehicle_count": fuel_count,
        "cost": round(cost_a, 2),
        "carbon": carbon_a,
        "travel_time": round(travel_time_a, 2),
        "load_unload_time": fuel_vehicle["load_unload_hours"],
        "total_time": round(total_time_a, 2),
        "desc": "运费最低，适合工期宽松、非关键路径物料",
        "recommended": False,
        "color": "#64748b"
    }

    # 方案B：新能源
    ev_vehicle = next(v for v in VEHICLES if v["type_id"] == "ev")
    ev_count = max(1, -(-quantity_ton // ev_vehicle["capacity_ton"]))
    cost_b = calc_cost(distance_km, "ev", quantity_ton)
    carbon_b = calc_carbon(distance_km, "ev", quantity_ton)
    travel_time_b = distance_km / ev_vehicle["avg_speed_kmh"]
    total_time_b = travel_time_b + ev_vehicle["load_unload_hours"]
    plans_b = {
        "name": "方案B · 低碳优先",
        "vehicle_type": "新能源货车",
        "vehicle_count": ev_count,
        "cost": round(cost_b, 2),
        "carbon": carbon_b,
        "travel_time": round(travel_time_b, 2),
        "load_unload_time": ev_vehicle["load_unload_hours"],
        "total_time": round(total_time_b, 2),
        "desc": "碳排放最低，适合ESG考核场景",
        "recommended": False,
        "color": "#10b981"
    }

    # 方案C：混合
    hybrid_vehicle = next(v for v in VEHICLES if v["type_id"] == "hybrid")
    hybrid_count = max(1, -(-quantity_ton // hybrid_vehicle["capacity_ton"]))
    load_half = quantity_ton / 2
    cost_c = round(calc_cost(distance_km, "fuel", load_half) + calc_cost(distance_km, "ev", load_half), 2)
    carbon_c = round(calc_carbon(distance_km, "fuel", load_half) + calc_carbon(distance_km, "ev", load_half), 2)
    travel_time_c = max(distance_km / fuel_vehicle["avg_speed_kmh"], distance_km / ev_vehicle["avg_speed_kmh"])
    total_time_c = travel_time_c + max(fuel_vehicle["load_unload_hours"], ev_vehicle["load_unload_hours"])
    plans_c = {
        "name": "方案C · 均衡方案",
        "vehicle_type": "燃油+新能源",
        "vehicle_count": hybrid_count,
        "cost": round(cost_c, 2),
        "carbon": carbon_c,
        "travel_time": round(travel_time_c, 2),
        "load_unload_time": max(fuel_vehicle["load_unload_hours"], ev_vehicle["load_unload_hours"]),
        "total_time": round(total_time_c, 2),
        "desc": "成本与碳排放平衡",
        "recommended": False,
        "color": "#2563eb"
    }

    # ----- 新增：归一化并计算综合得分 -----
    plans = [plans_a, plans_b, plans_c]
    max_cost = max(plan["cost"] for plan in plans)
    max_carbon = max(plan["carbon"] for plan in plans)
    # 避免除零（实际不会发生，但安全）
    if max_cost == 0:
        max_cost = 1
    if max_carbon == 0:
        max_carbon = 1

    for plan in plans:
        norm_cost = plan["cost"] / max_cost
        norm_carbon = plan["carbon"] / max_carbon
        # 综合得分：数值越小越好（越优）
        score = weight_cost * norm_cost + weight_carbon * norm_carbon
        plan["score"] = round(score, 4)

    # 推荐得分最低的方案
    recommended_plan = min(plans, key=lambda x: x["score"])
    for plan in plans:
        plan["recommended"] = (plan is recommended_plan)

    # 根据距离附加说明（推荐逻辑以综合得分为准）
    if distance_km < 50:
        plans_a["desc"] += "（短途成本优势明显）"
    elif distance_km > 200:
        plans_b["desc"] += "（长途低碳优势明显）"

    return {
        "material": material,
        "quantity_ton": round(quantity_ton, 2),
        "distance_km": distance_km,
        "weight_cost": weight_cost,
        "weight_carbon": weight_carbon,
        "plans": plans
    }

def optimize_inventory_logic():
    deadstock_items = []
    warning_items = []
    healthy_items = []
    eoq_suggestions = []  # 用于收集所有物料的EOQ建议
    inventory = db_fetch_all("SELECT * FROM inventory")
    projects = get_projects()
    materials = get_materials()
    project_map = {p["id"]: p for p in projects}
    material_map = {m["id"]: m for m in materials}

    for inv in inventory:
        project = project_map.get(inv["project_id"], {"short_name": inv["project_id"]})
        material = material_map.get(inv["material_id"], {"name": inv["material_id"], "unit": "", "safety_stock": 0, "plan_7d": 0, "lead_time_days": 0, "est_price": 0})
        current_stock = inv["current_stock"]
        safety_stock = material["safety_stock"]
        avg_daily = material["plan_7d"] / 7 if material["plan_7d"] else 0
        lead_time = material["lead_time_days"]
        dynamic_safety = round(avg_daily * lead_time * 1.3, 2) if avg_daily > 0 else 0
        safety_stock = max(safety_stock, dynamic_safety)

        status = inv["status"]
        if current_stock < safety_stock * 0.5:
            status = "urgent"
        elif current_stock < safety_stock * 0.8:
            status = "warning"
        elif current_stock > safety_stock * 1.5 and inv["last_used_days"] > 30:
            status = "deadstock"
        else:
            status = "normal"

        item = {
            "project_id": inv["project_id"],
            "project_name": project["short_name"],
            "material_id": inv["material_id"],
            "material_name": material["name"],
            "quantity": current_stock,
            "unit": material["unit"],
            "safety_stock": round(safety_stock, 2),
            "days_idle": inv["last_used_days"],
            "rotate_days": inv["rotate_days"],
            "est_value": round(current_stock * material["est_price"], 2),
            "status": status,
            "batch_no": inv["batch_no"],
            "received_date": inv["received_date"]
        }

        # 计算EOQ：经济订货批量
        annual_demand = avg_daily * 365 if avg_daily > 0 else 0  # 年需求量（估算）
        order_cost = 500  # 每次订货固定成本（元），可根据实际调整
        holding_cost_rate = 0.1  # 年持有成本率（10%）
        unit_holding_cost = material["est_price"] * holding_cost_rate
        if annual_demand > 0 and unit_holding_cost > 0:
            eoq = round((2 * annual_demand * order_cost / unit_holding_cost) ** 0.5, 2)
        else:
            eoq = 0
        item["eoq"] = eoq

        # 收集EOQ建议（用于前端汇总展示）
        eoq_suggestions.append({
            "project_id": item["project_id"],
            "project_name": item["project_name"],
            "material_id": item["material_id"],
            "material_name": item["material_name"],
            "eoq": eoq,
            "unit": item["unit"]
        })

        # 调拨建议
        transfer_to = []
        for op in projects:
            if op["id"] == inv["project_id"]:
                continue
            op_inv = get_inventory(op["id"], inv["material_id"])
            if op_inv and op_inv["current_stock"] < material["safety_stock"] * 0.5:
                transfer_qty = round(material["safety_stock"] * 0.5 - op_inv["current_stock"], 2)
                if transfer_qty > 0:
                    transfer_to.append({
                        "project_id": op["id"],
                        "project_name": op["short_name"],
                        "transfer_quantity": transfer_qty,
                        "unit": material["unit"]
                    })
        item["transfer_to"] = transfer_to

        if status == "deadstock":
            deadstock_items.append(item)
        elif status == "urgent" or status == "warning":
            warning_items.append(item)
        else:
            healthy_items.append(item)

    total_stock_value = sum(i["est_value"] for i in deadstock_items + warning_items + healthy_items)
    deadstock_value = sum(i["est_value"] for i in deadstock_items)
    deadstock_ratio = round(deadstock_value / total_stock_value * 100, 1) if total_stock_value > 0 else 0

    return {
        "deadstock": deadstock_items,
        "warning": warning_items,
        "healthy_count": len(healthy_items),
        "total_stock_value": round(total_stock_value, 2),
        "deadstock_value": round(deadstock_value, 2),
        "deadstock_ratio": deadstock_ratio,
        "eoq_suggestions": eoq_suggestions,  # 新增：所有库存物料的EOQ建议
        "summary": f"库存总价值¥{total_stock_value}，呆滞物料价值¥{deadstock_value}，占比{deadstock_ratio}%。"
    }

def loss_management_logic():
    records = get_loss_records()
    total_loss_amount = sum(l["loss_amount"] for l in records)
    reason_stats = {}
    for l in records:
        reason = l["reason"]
        if reason not in reason_stats:
            reason_stats[reason] = {"count": 0, "amount": 0, "ratio": 0.0}
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["amount"] += l["loss_amount"]
    if total_loss_amount > 0:
        for r in reason_stats:
            reason_stats[r]["ratio"] = round(reason_stats[r]["amount"] / total_loss_amount * 100, 1)

    # 帕累托分析：按损耗金额从高到低排序，计算累计占比
    pareto_data = []
    sorted_reasons = sorted(reason_stats.items(), key=lambda x: x[1]["amount"], reverse=True)
    cumulative = 0.0
    for reason, stat in sorted_reasons:
        cumulative += stat["amount"]
        cum_ratio = round(cumulative / total_loss_amount * 100, 1) if total_loss_amount > 0 else 0.0
        pareto_data.append({
            "reason": reason,
            "count": stat["count"],
            "amount": stat["amount"],
            "ratio": stat["ratio"],
            "cum_ratio": cum_ratio
        })

    processed_records = []
    for l in records:
        project = get_project(l["project_id"])
        material = get_material(l["material_id"])
        processed_records.append({
            **l,
            "project_name": project["short_name"] if project else "未知",
            "material_name": material["name"] if material else "未知",
            "unit": material["unit"] if material else "",
            "standard_loss_rate": f"{material['loss_rate']*100}%" if material else "N/A",
            "loss_amount_str": f"¥{l['loss_amount']:,.2f}"
        })
    return {
        "records": processed_records,
        "total_loss_amount": round(total_loss_amount, 2),
        "reason_stats": reason_stats,
        "pareto_data": pareto_data,   # 新增：帕累托分析数据（按金额降序，带累计占比）
        "high_risk_time": "夜间 22:00 - 06:00",
        "suggestion": "建议加强夜间安保巡逻，对高价值物料（钢筋、水泥）实行专人看管。露天堆放物料需加盖防护，减少因天气原因导致的损耗。"
    }

def dynamic_adjustment_logic():
    return {
        "records": get_change_logs(),
        "pending_actions": [
            {"id": "P001", "title": "项目Y砌块采购提前确认", "deadline": "今日 18:00 前", "priority": "high",
             "suggested_action": "联系供应商S005确认提前供货能力，必要时启用备用供应商"},
            {"id": "P002", "title": "供应商A报价调整审批", "deadline": "明日 12:00 前", "priority": "medium",
             "suggested_action": "采购部审核价格调整合理性，并更新系统基准价"},
            {"id": "P003", "title": "项目X钢管盘点差异复核", "deadline": "今日 17:00 前", "priority": "high",
             "suggested_action": "材料员现场复核钢管数量，检查夜间监控录像，落实责任"}
        ],
        "last_update": "2024-01-28 10:00"
    }

def esg_report_logic():
    records = get_transport_records()
    if records:
        total_carbon = sum(calc_carbon(r["distance_km"], r["vehicle"], r["load_ton"]) for r in records)
        baseline_carbon = sum(calc_carbon(r["distance_km"], "fuel", r["load_ton"]) for r in records)
    else:
        total_carbon = 0
        baseline_carbon = 0
    reduction = round(baseline_carbon - total_carbon, 2)
    reduction_ratio = round(reduction / baseline_carbon * 100, 1) if baseline_carbon > 0 else 0
    carbon_credits = round(reduction * 10, 2)
    carbon_value = round(reduction / 1000 * 60, 2)

    subsidies = [
        {"name": "杭州市绿色建筑供应链补贴", "amount": 15000, "status": "可申领", "deadline": "2024-03-31",
         "condition": "年碳减排量≥10吨", "progress": "准备材料中"},
        {"name": "新能源物流车辆购置补贴", "amount": 80000, "status": "已申领", "deadline": "2024-06-30",
         "condition": "购买新能源货车并投入使用", "progress": "已完成"},
        {"name": "低碳运输示范项目奖励", "amount": 30000, "status": "可申领", "deadline": "2024-04-30",
         "condition": "季度碳减排率≥15%", "progress": "待提交数据"}
    ]
    monthly_data = []
    for month in range(1, 13):
        base = round(random.uniform(600, 1200), 1)
        red = round(base * random.uniform(0.05, 0.20), 1)
        monthly_data.append({"month": f"{month}月", "carbon": base, "reduction": red, "reduction_ratio": round(red/base*100,1)})
    emission_projects = [
        {"name": "新能源运输替代", "reduction": round(total_carbon * 0.4, 2), "description": "使用新能源货车替代燃油货车，减少直接排放"},
        {"name": "路线优化", "reduction": round(total_carbon * 0.2, 2), "description": "优化运输路线，减少绕行和空载"},
        {"name": "供应商本地化", "reduction": round(total_carbon * 0.15, 2), "description": "优先选择本地供应商，缩短运输距离"}
    ]
    return {
        "total_carbon": round(total_carbon, 2),
        "baseline_carbon": round(baseline_carbon, 2),
        "reduction": reduction,
        "reduction_ratio": reduction_ratio,
        "carbon_credits": carbon_credits,
        "carbon_value": carbon_value,
        "subsidies": subsidies,
        "monthly_data": monthly_data,
        "transport_records": records,
        "emission_projects": emission_projects,
        "scope_breakdown": {
            "scope1": round(total_carbon * 0.7, 2),
            "scope2": round(total_carbon * 0.2, 2),
            "scope3": round(total_carbon * 0.1, 2)
        }
    }

def dashboard_logic():
    inventory = db_fetch_all("SELECT * FROM inventory")
    materials = get_materials()
    projects = get_projects()
    material_map = {m["id"]: m for m in materials}
    project_map = {p["id"]: p for p in projects}

    shortage_items = []
    for inv in inventory:
        material = material_map.get(inv["material_id"])
        if material and inv["current_stock"] < material["safety_stock"] * 0.5:
            shortage_items.append({
                "project": project_map[inv["project_id"]]["short_name"],
                "material": material["name"],
                "current": inv["current_stock"],
                "safety": material["safety_stock"],
                "unit": material["unit"]
            })

    supplier_count = len(db_fetch_all("SELECT * FROM suppliers"))
    transport_count = len(get_transport_records())
    deadstock_count = len([i for i in inventory if i["status"] == "deadstock"])
    alerts = get_alerts()
    alert_count = len([a for a in alerts if a["level"] == "high"])

    total_stock_value = sum(
        material_map[inv["material_id"]]["est_price"] * inv["current_stock"] for inv in inventory
        if inv["material_id"] in material_map
    )
    monthly_purchase = round(random.uniform(200000, 500000), 2)
    total_distance = sum(r["distance_km"] for r in get_transport_records())

    material_dist = []
    for m in materials:
        total_stock = sum(inv["current_stock"] for inv in inventory if inv["material_id"] == m["id"])
        material_dist.append({
            "name": m["name"],
            "value": total_stock,
            "unit": m["unit"],
            "value_amount": round(total_stock * m["est_price"], 2)
        })

    trend = []
    for i in range(6, -1, -1):
        day = datetime.now() - timedelta(days=i)
        trend.append({"date": day.strftime("%m-%d"), "consumption": round(random.uniform(30, 80), 1), "arrival": round(random.uniform(25, 75), 1)})

    return {
        "kpi": {
            "shortage_count": len(shortage_items),
            "supplier_count": supplier_count,
            "transport_count": transport_count,
            "deadstock_count": deadstock_count,
            "alert_count": alert_count,
            "total_stock_value": round(total_stock_value, 2),
            "monthly_purchase": monthly_purchase,
            "total_distance": total_distance
        },
        "shortage_items": shortage_items,
        "alerts": alerts[:5],
        "material_dist": material_dist,
        "trend": trend,
    }

# ==================== 数据导入逻辑 ====================
def clear_all_tables():
    tables = ["inventory", "suppliers", "loss_records", "transport_records", "alerts", "change_logs", "materials", "projects"]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()

def insert_data_from_dict(data: dict):
    """追加/合并模式导入 JSON 数据，不清空原有数据，主键冲突时更新"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # 确保 inventory 表有唯一索引，防止重复插入同一项目+物料
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_inventory_unique ON inventory(project_id, material_id)")

        if "projects" in data:
            for p in data["projects"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO projects (id, name, short_name, location, stage, progress, manager, workers, building_area, type)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (p["id"], p["name"], p.get("short_name",""), p.get("location",""), p.get("stage",""),
                      p.get("progress",0), p.get("manager",""), p.get("workers",0), p.get("building_area",0), p.get("type","")))

        if "materials" in data:
            for m in data["materials"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO materials (id, name, unit, safety_stock, weight_per_unit, est_price, plan_7d, plan_30d, plan_90d, lead_time_days, loss_rate, category, storage_condition)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (m["id"], m["name"], m.get("unit",""), m.get("safety_stock",0), m.get("weight_per_unit",0),
                      m.get("est_price",0), m.get("plan_7d",0), m.get("plan_30d",0), m.get("plan_90d",0),
                      m.get("lead_time_days",0), m.get("loss_rate",0), m.get("category",""), m.get("storage_condition","")))

        if "inventory" in data:
            for inv in data["inventory"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO inventory (project_id, material_id, current_stock, last_used_days, rotate_days, status, batch_no, received_date)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (inv["project_id"], inv["material_id"], inv.get("current_stock",0), inv.get("last_used_days",0),
                      inv.get("rotate_days",0), inv.get("status","normal"), inv.get("batch_no",""), inv.get("received_date","")))

        if "suppliers" in data:
            for s in data["suppliers"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO suppliers (id, name, material_id, price, market_price, on_time_rate, quality_rate, credit_score, violations, location, service_level, cooperation_years, green_certified, capacity)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (s["id"], s["name"], s.get("material_id",""), s.get("price",0), s.get("market_price",0),
                      s.get("on_time_rate",0), s.get("quality_rate",0), s.get("credit_score",0), s.get("violations",0),
                      s.get("location",""), s.get("service_level",0), s.get("cooperation_years",0),
                      1 if s.get("green_certified", False) else 0, s.get("capacity","")))

        if "transport_records" in data:
            for t in data["transport_records"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO transport_records (id, date, vehicle, distance_km, load_ton, status)
                    VALUES (?,?,?,?,?,?)
                """, (t["id"], t.get("date",""), t.get("vehicle",""), t.get("distance_km",0), t.get("load_ton",0), t.get("status","")))

        if "loss_records" in data:
            for l in data["loss_records"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO loss_records (id, date, project_id, material_id, quantity, reason, description, loss_amount)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (l["id"], l.get("date",""), l.get("project_id",""), l.get("material_id",""), l.get("quantity",0),
                      l.get("reason",""), l.get("description",""), l.get("loss_amount",0)))

        if "alerts" in data:
            for a in data["alerts"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO alerts (id, type, level, title, detail, time)
                    VALUES (?,?,?,?,?,?)
                """, (a["id"], a.get("type",""), a.get("level","low"), a.get("title",""), a.get("detail",""), a.get("time","")))

        if "change_logs" in data:
            for c in data["change_logs"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO change_logs (id, time, event, reason, adjustment, confirm_by)
                    VALUES (?,?,?,?,?,?)
                """, (c["id"], c.get("time",""), c.get("event",""), c.get("reason",""), c.get("adjustment",""), c.get("confirm_by","")))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# ==================== API 路由 ====================
@app.get("/api/dashboard")
async def api_dashboard():
    return JSONResponse(dashboard_logic())

@app.get("/api/predict")
async def api_predict(project_id: str = "P001", material_id: str = "M001", days: int = 7):
    return JSONResponse(predict_demand_logic(project_id, material_id, days))

@app.get("/api/suppliers")
async def api_suppliers(material_id: str = "M001"):
    return JSONResponse(evaluate_supplier_logic(material_id))

@app.get("/api/transport")
async def api_transport(material_id: str = "M001", quantity_ton: float = 100, distance_km: float = 200):
    return JSONResponse(generate_transport_plan_logic(material_id, quantity_ton, distance_km))

@app.get("/api/inventory")
async def api_inventory():
    return JSONResponse(optimize_inventory_logic())

@app.get("/api/loss")
async def api_loss():
    return JSONResponse(loss_management_logic())

@app.get("/api/changes")
async def api_changes():
    return JSONResponse(dynamic_adjustment_logic())

@app.get("/api/esg")
async def api_esg():
    return JSONResponse(esg_report_logic())

@app.get("/api/projects")
async def api_projects():
    return JSONResponse(get_projects())

@app.get("/api/materials")
async def api_materials():
    return JSONResponse(get_materials())

@app.get("/api/knowledge/search")
async def search_knowledge(query: str):
    results = retrieve_knowledge(query, top_k=3)
    return JSONResponse({"results": results})

# ==================== 数据导入接口 ====================
@app.post("/api/import/json")
async def import_json(payload: dict):
    try:
        insert_data_from_dict(payload)
        return JSONResponse({"success": True, "message": "数据导入成功"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"导入失败: {str(e)}")

@app.post("/api/import/csv")
async def import_csv(file: UploadFile = File(...), data_type: str = Form(...)):
    allowed_types = list(ALLOWED_COLUMNS.keys())
    if data_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的数据类型: {data_type}")

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
        csv_reader = csv.DictReader(io.StringIO(text))
        rows = [row for row in csv_reader]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV 解析失败: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV 文件为空")

    # 获取列名并校验（白名单校验，防止 SQL 注入）
    columns = [col.strip() for col in rows[0].keys()]
    allowed_cols = ALLOWED_COLUMNS[data_type]
    if set(columns) != set(allowed_cols):
        invalid = set(columns) - set(allowed_cols)
        missing = set(allowed_cols) - set(columns)
        err_msg = "CSV 列名不合法"
        if invalid:
            err_msg += f"，包含非法字段: {', '.join(sorted(invalid))}"
        if missing:
            err_msg += f"，缺少必要字段: {', '.join(sorted(missing))}"
        raise HTTPException(status_code=400, detail=err_msg)

    # 可选：进一步确保列名安全（仅含字母数字下划线）
    import re
    for col in columns:
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', col):
            raise HTTPException(status_code=400, detail=f"列名包含非法字符: {col}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(f"DELETE FROM {data_type}")

        # 构建 INSERT 语句（列名已通过白名单验证，安全拼接）
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)  # 也可以用双引号包裹：", ".join(f'"{c}"' for c in columns)
        sql = f"INSERT INTO {data_type} ({col_str}) VALUES ({placeholders})"

        # 数值字段和布尔字段的转换规则保持原逻辑
        numeric_fields = {
            "projects": ["progress", "workers", "building_area"],
            "materials": ["safety_stock", "weight_per_unit", "est_price", "plan_7d", "plan_30d", "plan_90d", "lead_time_days", "loss_rate"],
            "inventory": ["current_stock", "last_used_days", "rotate_days"],
            "suppliers": ["price", "market_price", "on_time_rate", "quality_rate", "credit_score", "violations", "service_level", "cooperation_years", "green_certified"],
            "transport_records": ["distance_km", "load_ton"],
            "loss_records": ["quantity", "loss_amount"],
            "alerts": [],
            "change_logs": []
        }.get(data_type, [])
        bool_fields = {"suppliers": ["green_certified"]}.get(data_type, [])

        for row in rows:
            values = []
            for col in columns:
                val = row[col]
                if col in numeric_fields:
                    try:
                        val = float(val)
                    except:
                        val = 0
                if col in bool_fields:
                    val = 1 if str(val).lower() in ("true", "1", "yes") else 0
                values.append(val)
            cursor.execute(sql, values)

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"数据库写入失败: {str(e)}")
    finally:
        conn.close()

    return JSONResponse({"success": True, "message": f"CSV 导入成功，共导入 {len(rows)} 条记录"})

# ==================== AI 助手接口 ====================
class ChatMessage(BaseModel):
    message: str

BASE_SYSTEM_PROMPT = """你是建筑工地物料全生命周期智能调度 AI 助手，精通需求预测、供应商评估、物流调度、库存优化、损耗管理、动态调整和 ESG 报告。
当用户提问涉及具体业务时，请务必调用相应的函数获取实时数据，再基于数据给出专业回答。
如果用户的问题不涉及业务查询，可以礼貌地介绍自己的功能。
回答中不要提及函数调用或技术细节，用自然语言向用户汇报结果。"""

TOOL_TO_PAGE = {
    "predict_demand": "predict",
    "evaluate_supplier": "supplier",
    "generate_transport_plan": "transport",
    "optimize_inventory": "inventory",
    "loss_management": "loss",
    "dynamic_adjustment": "changes",
    "esg_report": "esg",
    "dashboard_overview": "dashboard",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_demand",
            "description": "预测指定项目、指定物料在未来N天的需求、库存缺口，并给出采购建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目ID"},
                    "material_id": {"type": "string", "description": "物料ID"},
                    "days": {"type": "integer", "description": "预测天数，默认7，可选7/30/90"}
                },
                "required": ["project_id", "material_id", "days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_supplier",
            "description": "评估指定物料的供应商，返回综合评分、排名、推荐供应商及风险提示",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": {"type": "string", "description": "物料ID"}
                },
                "required": ["material_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_transport_plan",
            "description": "生成多目标物流调度方案（成本优先、低碳优先、均衡推荐）",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": {"type": "string", "description": "物料ID"},
                    "quantity_ton": {"type": "number", "description": "运输量（吨），默认100"},
                    "distance_km": {"type": "number", "description": "运输距离（公里），默认200"}
                },
                "required": ["material_id", "quantity_ton", "distance_km"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_inventory",
            "description": "分析库存健康度，识别呆滞物料、预警物料，并给出跨项目调拨建议",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loss_management",
            "description": "获取损耗记录、损耗原因分析及高损耗时段建议",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dynamic_adjustment",
            "description": "获取动态调整记录和待处理事项",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "esg_report",
            "description": "生成 ESG 报告，包括碳排放、减排量、碳积分、政策补贴等",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dashboard_overview",
            "description": "获取驾驶舱总览数据（KPI、预警、库存分布、趋势等）",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def execute_tool(tool_name: str, arguments: dict):
    try:
        if tool_name == "predict_demand":
            return predict_demand_logic(
                project_id=arguments.get("project_id", "P001"),
                material_id=arguments.get("material_id", "M001"),
                days=int(arguments.get("days", 7))
            )
        elif tool_name == "evaluate_supplier":
            return evaluate_supplier_logic(material_id=arguments.get("material_id", "M001"))
        elif tool_name == "generate_transport_plan":
            return generate_transport_plan_logic(
                material_id=arguments.get("material_id", "M001"),
                quantity_ton=float(arguments.get("quantity_ton", 100)),
                distance_km=float(arguments.get("distance_km", 200))
            )
        elif tool_name == "optimize_inventory":
            return optimize_inventory_logic()
        elif tool_name == "loss_management":
            return loss_management_logic()
        elif tool_name == "dynamic_adjustment":
            return dynamic_adjustment_logic()
        elif tool_name == "esg_report":
            return esg_report_logic()
        elif tool_name == "dashboard_overview":
            return dashboard_logic()
        else:
            return {"error": f"未知工具: {tool_name}"}
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)}"}

def compress_tool_result(tool_name, result):
    """压缩大型工具结果，避免上下文超长"""
    if tool_name == "optimize_inventory":
        return {
            "deadstock_count": len(result.get("deadstock", [])),
            "warning_count": len(result.get("warning", [])),
            "total_stock_value": result.get("total_stock_value"),
            "deadstock_ratio": result.get("deadstock_ratio"),
            "summary": result.get("summary"),
            "eoq_suggestions_count": len(result.get("eoq_suggestions", []))
        }
    elif tool_name == "dashboard_overview":
        return {
            "kpi": result.get("kpi"),
            "alerts": result.get("alerts", [])[:3]  # 只取前3条
        }
    elif tool_name == "loss_management":
        return {
            "total_loss_amount": result.get("total_loss_amount"),
            "reason_stats": result.get("reason_stats"),
            "pareto_data": result.get("pareto_data", [])[:5]
        }
    else:
        return result

# 会话管理（内存存储，生产环境建议使用 Redis）
sessions = defaultdict(list)
MAX_HISTORY = 10  # 最多保留最近10轮对话

@app.post("/api/ai_chat")
async def ai_chat(payload: ChatMessage, session_id: str = None):
    user_msg = payload.message.strip()
    if not user_msg:
        return JSONResponse({"reply": "请输入您的问题。", "action": None})

    # 会话管理
    if not session_id:
        session_id = str(uuid.uuid4())
    history = sessions[session_id]

    # ========== 动态获取项目和物料映射 ==========
    try:
        projects = get_projects()
        materials = get_materials()
        project_lines = [f"- {p['short_name']} (ID: {p['id']}, 全称: {p['name']})" for p in projects]
        material_lines = [f"- {m['name']} (ID: {m['id']})" for m in materials]
        context_info = (
            "可用的项目列表：\n" + "\n".join(project_lines) +
            "\n\n可用的物料列表：\n" + "\n".join(material_lines)
        )
    except Exception as e:
        print(f"获取项目和物料信息失败: {e}")
        context_info = ""

    # 知识库检索
    knowledge_context = ""
    kb_results = retrieve_knowledge(user_msg, top_k=3)
    if kb_results:
        knowledge_context = "\n\n".join([f"【参考知识片段{i+1}】{item['content']}" for i, item in enumerate(kb_results)])
        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + context_info + "\n\n以下是可能相关的领域知识，请参考：\n" + knowledge_context
    else:
        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + context_info

    # 构建消息（包含历史）
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1000
        )
        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            tool_calls = assistant_msg.tool_calls
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in tool_calls
                ]
            })
            action = None
            for tc in tool_calls:
                tool_name = tc.function.name
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = execute_tool(tool_name, arguments)
                # 压缩工具结果
                compressed_result = compress_tool_result(tool_name, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(compressed_result, ensure_ascii=False)
                })
                if tool_name in TOOL_TO_PAGE:
                    page = TOOL_TO_PAGE[tool_name]
                    action_params = {}
                    if tool_name == "predict_demand":
                        action_params = {
                            "project_id": arguments.get("project_id", "P001"),
                            "material_id": arguments.get("material_id", "M001"),
                            "days": int(arguments.get("days", 7))
                        }
                    elif tool_name in ("evaluate_supplier", "generate_transport_plan"):
                        action_params = {"material_id": arguments.get("material_id", "M001")}
                    action = {"page": page, **action_params}
            final_response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            final_reply = final_response.choices[0].message.content
        else:
            final_reply = assistant_msg.content or "抱歉，我没有理解您的问题。"
            action = None

    except Exception as e:
        print(f"DeepSeek API error: {e}")
        msg = user_msg.lower()
        reply = ""
        action = None
        if "预测" in msg or "需求" in msg:
            project_id = "P001"
            material_id = "M001"
            days = 7
            if "项目y" in msg or "项目 y" in msg:
                project_id = "P002"
            elif "项目z" in msg or "项目 z" in msg:
                project_id = "P003"
            if "钢管" in msg:
                material_id = "M002"
            elif "砌块" in msg:
                material_id = "M003"
            elif "螺纹钢" in msg or "钢筋" in msg:
                material_id = "M004"
            elif "水泥" in msg:
                material_id = "M005"
            if "30天" in msg:
                days = 30
            elif "90天" in msg:
                days = 90
            reply = f"好的，正在为您预测 {get_project(project_id)['short_name']} 未来{days}天 {get_material(material_id)['name']} 需求..."
            action = {"page": "predict", "project_id": project_id, "material_id": material_id, "days": days}
        elif "供应商" in msg or "采购" in msg:
            material_id = "M001"
            if "钢管" in msg:
                material_id = "M002"
            elif "砌块" in msg:
                material_id = "M003"
            elif "螺纹钢" in msg or "钢筋" in msg:
                material_id = "M004"
            elif "水泥" in msg:
                material_id = "M005"
            reply = f"正在评估 {get_material(material_id)['name']} 的供应商..."
            action = {"page": "supplier", "material_id": material_id}
        elif "物流" in msg or "运输" in msg or "方案" in msg:
            material_id = "M001"
            if "钢管" in msg:
                material_id = "M002"
            elif "砌块" in msg:
                material_id = "M003"
            elif "螺纹钢" in msg or "钢筋" in msg:
                material_id = "M004"
            elif "水泥" in msg:
                material_id = "M005"
            reply = f"正在生成 {get_material(material_id)['name']} 的多目标物流方案..."
            action = {"page": "transport", "material_id": material_id}
        elif "库存" in msg or "呆滞" in msg or "调拨" in msg:
            reply = "正在分析库存健康度..."
            action = {"page": "inventory"}
        elif "碳" in msg or "esg" in msg:
            reply = "正在生成 ESG 报告..."
            action = {"page": "esg"}
        else:
            reply = "抱歉，我还没理解您的需求。您可以试试：“预测项目X未来7天混凝土需求”、“评估混凝土供应商”、“生成物流方案”等。"
        final_reply = reply

    # 更新会话历史
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": final_reply})
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2:]
    sessions[session_id] = history

    return JSONResponse({"reply": final_reply, "action": action, "session_id": session_id})

# 流式聊天接口（简化版：仅在无工具调用时流式输出）
@app.post("/api/ai_chat_stream")
async def ai_chat_stream(payload: ChatMessage, session_id: str = None):
    user_msg = payload.message.strip()
    if not user_msg:
        return StreamingResponse(iter(["请输入您的问题。"]), media_type="text/event-stream")

    # 会话管理
    if not session_id:
        session_id = str(uuid.uuid4())
    history = sessions[session_id]

    # ========== 动态获取项目和物料映射 ==========
    try:
        projects = get_projects()
        materials = get_materials()
        project_lines = [f"- {p['short_name']} (ID: {p['id']}, 全称: {p['name']})" for p in projects]
        material_lines = [f"- {m['name']} (ID: {m['id']})" for m in materials]
        context_info = (
            "可用的项目列表：\n" + "\n".join(project_lines) +
            "\n\n可用的物料列表：\n" + "\n".join(material_lines)
        )
    except Exception as e:
        print(f"获取项目和物料信息失败: {e}")
        context_info = ""

    # 知识库检索
    knowledge_context = ""
    kb_results = retrieve_knowledge(user_msg, top_k=3)
    if kb_results:
        knowledge_context = "\n\n".join([f"【参考知识片段{i+1}】{item['content']}" for i, item in enumerate(kb_results)])
        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + context_info + "\n\n以下是可能相关的领域知识，请参考：\n" + knowledge_context
    else:
        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + context_info

    # 构建消息（包含历史）
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    async def generate():
        nonlocal history
        try:
            # 先尝试流式（不带工具调用，如果模型选择调用工具则后续处理）
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
                temperature=0.2,
                max_tokens=1000
            )
            tool_calls_buffer = []
            current_tool_call = None
            content_buffer = ""

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    content_buffer += delta.content
                    yield f"data: {json.dumps({'content': delta.content}, ensure_ascii=False)}\n\n"
                elif delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.index >= len(tool_calls_buffer):
                            tool_calls_buffer.append({"id": "", "function": {"name": "", "arguments": ""}})
                        tc = tool_calls_buffer[tc_delta.index]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

            # 如果检测到工具调用，则使用非流式方式处理（简化）
            if tool_calls_buffer:
                # 将工具调用加入 messages
                messages.append({
                    "role": "assistant",
                    "content": content_buffer or "",
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                        for tc in tool_calls_buffer
                    ]
                })
                action = None
                for tc in tool_calls_buffer:
                    tool_name = tc["function"]["name"]
                    arguments = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                    result = execute_tool(tool_name, arguments)
                    compressed_result = compress_tool_result(tool_name, result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(compressed_result, ensure_ascii=False)
                    })
                    if tool_name in TOOL_TO_PAGE:
                        page = TOOL_TO_PAGE[tool_name]
                        action_params = {}
                        if tool_name == "predict_demand":
                            action_params = {
                                "project_id": arguments.get("project_id", "P001"),
                                "material_id": arguments.get("material_id", "M001"),
                                "days": int(arguments.get("days", 7))
                            }
                        elif tool_name in ("evaluate_supplier", "generate_transport_plan"):
                            action_params = {"material_id": arguments.get("material_id", "M001")}
                        action = {"page": page, **action_params}
                # 获取最终回复（非流式）
                final_response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1000
                )
                final_reply = final_response.choices[0].message.content
                yield f"data: {json.dumps({'content': final_reply, 'action': action}, ensure_ascii=False)}\n\n"
            else:
                # 没有工具调用，直接输出已经流式发送的 content_buffer，并附带 action=None
                yield f"data: {json.dumps({'action': None}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

            # 更新会话历史
            assistant_reply = content_buffer if not tool_calls_buffer else final_reply
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_reply})
            if len(history) > MAX_HISTORY * 2:
                history = history[-MAX_HISTORY * 2:]
            sessions[session_id] = history

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# 主动预警接口
@app.get("/api/ai/active_alerts")
async def api_active_alerts():
    alerts = get_alerts()
    high_alerts = [a for a in alerts if a["level"] == "high"]
    return JSONResponse({"high_alerts": high_alerts})

# ==================== 静态文件挂载 ====================
STATIC_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
