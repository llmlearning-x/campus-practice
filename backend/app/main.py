from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import json
import sqlite3
import threading
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from .config import settings
from .database import get_db, init_database, transaction, utc_now
from .rag_engine_v2 import HHURAGEngine, rag_engine
from .security import create_token, decode_token, hash_password, verify_password


class RegisterBody(BaseModel):
    name: str = Field(min_length=2, max_length=40)
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class ChatBody(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    conversation_id: int | None = None


class UserUpdateBody(BaseModel):
    role: str | None = None
    is_active: bool | None = None

# 访客模式
def public_user(row: sqlite3.Row) -> dict[str, object]:
    """把 users 行转换成对外安全的字典（白名单过滤）。

    为什么是白名单而不是黑名单：
    - 黑名单（"去掉 password_hash"）容易遗漏字段，未来加新敏感字段会泄露
    - 白名单（"只输出这 6 个字段"）天然防泄露，新增敏感字段默认就不暴露

    用作所有返回用户信息的端点的统一格式化函数：register / login / guest / me / list_users / update_user。
    """
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }


def seed_database() -> None:
    """启动时初始化：bootstrap 管理员账号 + 种一份示例文档到知识库。

    两件事：
    1. 检查默认管理员账号是否存在，不存在就 INSERT（密码硬编码）
    2. 检查 seed-hhu-guide.md 文档是否已索引，未索引就插入 SQLite + 走 RAG 引擎建立 FAISS 索引

    幂等性：多次调用安全。重复启动不会重复建账号或重复索引：
    - 管理员：SELECT WHERE email 找不到才 INSERT（已存在就拿现有 id）
    - 文档：SELECT WHERE stored_name 找到且 status='READY' + chunk_count>0 时直接 return

    生产前必须改：默认密码 + 默认邮箱 + seed 内容（河海校园介绍）都是硬编码的演示数据。
    注意：本函数不直接管理 seed 文档的 chunk_count / status 更新——由 rag_engine.add_document 内部完成。
    """
    with transaction() as db:
        admin = db.execute(
            "SELECT id FROM users WHERE email = ?", ("admin@campus.example",)
        ).fetchone()
        if not admin:
            cursor = db.execute(
                """
                INSERT INTO users(name, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, 'ADMIN', 1, ?)
                """,
                (
                    "系统管理员",
                    "admin@ca.mpus.example",
                    hash_password("admin123"),
                    utc_now(),
                ),
            )
            admin_id = int(cursor.lastrowid)
        else:
            admin_id = int(admin["id"])

        # 河海大学示例数据
        sample = (
            "河海大学是一所拥有111年办学历史，以水利为特色、工科为主，"
            '多学科协调发展的教育部直属全国重点大学，是国家"双一流"建设、'
            '"211工程"重点建设、"985工程优势学科创新平台"建设以及'
            "经国家批准设立研究生院的高校。\n\n"
            "河海大学的校训是：艰苦朴素、实事求是、严格要求、勇于探索。\n\n"
            "图书馆周一至周五开放时间为 08:00 至 22:00，周末开放时间为 "
            "09:00 至 21:00。法定节假日安排以图书馆公告为准。\n\n"
            "校园卡补办\n\n"
            "学生遗失校园卡后，可先在校园服务 App 中挂失，再携带学生证前往"
            "行政楼一层校园卡服务中心补办。补办工本费为 20 元。\n\n"
            "奖学金申请\n\n"
            "综合奖学金每学年评定一次，申请人需完成本学年培养方案规定课程，"
            "无违纪记录，并在学院通知期限内提交申请表和成绩证明。"
        )
        seed_path = settings.upload_dir / "seed-hhu-guide.md"
        seed_path.write_bytes(sample.encode("utf-8"))

        seed_document = db.execute(
            "SELECT id, status, chunk_count FROM documents WHERE stored_name = 'seed-hhu-guide.md'"
        ).fetchone()
        if (
            seed_document
            and seed_document["status"] == "READY"
            and seed_document["chunk_count"] > 0
        ):
            return

        if not seed_document:
            now = utc_now()
            cursor = db.execute(
                """
                INSERT INTO documents(
                    title, filename, stored_name, mime_type, size, status,
                    chunk_count, uploaded_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'PROCESSING', 0, ?, ?, ?)
                """,
                (
                    "河海大学校园服务指南（示例）",
                    "hhu-guide.md",
                    "seed-hhu-guide.md",
                    "text/markdown",
                    len(sample.encode()),
                    admin_id,
                    now,
                    now,
                ),
            )
            document_id = int(cursor.lastrowid)
        else:
            document_id = int(seed_document["id"])

        rag_engine.add_document(
            seed_path,
            document_id=document_id,
            title="河海大学校园服务指南（示例）",
            category="校园服务",
            db=db,
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_database()
    # 先初始化 RAG 引擎（加载/创建 FAISS 索引），再用它索引 seed 数据
    try:
        _ = HHURAGEngine()
        print("✅ DashScope + FAISS RAG engine initialized")
    except Exception as e:
        print(f"⚠️ RAG engine initialization failed: {e}")
        raise
    seed_database()

    # 在后台线程加载河海知识库，避免阻塞服务启动
    def _load_kb() -> None:
        try:
            loaded = rag_engine.load_knowledge_base(settings.knowledge_base_dir)
            print(f"✅ 知识库加载完成: {loaded} 篇文档")
        except Exception as e:
            print(f"⚠️ 知识库加载失败: {e}")

    threading.Thread(target=_load_kb, daemon=True).start()
    yield


app = FastAPI(title="河海大学校园问答助手 API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Database = Annotated[sqlite3.Connection, Depends(get_db)]


def current_user(
    db: Database, authorization: Annotated[str | None, Header()] = None
) -> sqlite3.Row:
    """FastAPI 依赖：从 Authorization header 解 JWT → 查 users → 返回 user 行。

    三道闸（任何一步失败都 401）：
    1. Header 缺失或不是 "Bearer " 开头 → 请先登录
    2. JWT 签名不匹配 / 已过期 → 登录状态已失效（decode_token 返回 None）
    3. 用户不存在 / is_active=0 → 账号不可用（被管理员停用也算）

    用作所有需要登录的端点的依赖注入。
    注意：这里不做 admin 校验——需要 admin 的端点要叠加 Depends(admin_user)。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    payload = decode_token(authorization.removeprefix("Bearer ").strip())
    if not payload:
        raise HTTPException(status_code=401, detail="登录状态已失效")
    user = db.execute(
        "SELECT * FROM users WHERE id = ?", (int(str(payload["sub"])),)
    ).fetchone()
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="账号不可用")
    return user


def admin_user(user: Annotated[sqlite3.Row, Depends(current_user)]) -> sqlite3.Row:
    """FastAPI 依赖：叠加在 current_user 之上，要求 `role == "ADMIN"`。

    双层防护思路：
    - 前端 `isAdmin()` 隐藏 UI（`AuthContext.tsx:33-35`）只是 UX 优化
    - 后端 admin_user 是最后一道硬墙——客户端篡改 localStorage 也无法越权

    用法：`db: Database, _: Annotated[Row, Depends(admin_user)]`——用 `_` 占位表示 user 参数不用。
    """
    if user["role"] != "ADMIN":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


@app.get("/api/health")
def health() -> dict[str, str]:
    """健康检查端点（无需鉴权）。

    返回：status / mode（dashscope 表示用阿里云百炼）/ version。
    用于：K8s liveness probe / 监控 / 部署后冒烟测试。
    """
    return {"status": "ok", "mode": "dashscope", "version": "2.0.0"}


@app.post("/api/auth/guest")
def guest_login(db: Database) -> dict[str, object]:
    """为嵌入 Widget 创建独立的匿名访客身份。

    流程：
    1. 生成 32 位 hex 随机串 `guest_key`
    2. INSERT users 行：name="访客"、email=guest-{key}@campus.local、role=STUDENT
    3. password_hash 用 hash_password(guest_key) 填充——这个"密码"是随机串，没人知道，
       所以访客账号永远无法用密码登录，只能凭 token 续 session
    4. 返回 JWT + user（用 public_user 白名单过滤）

    关键设计：role 写死 'STUDENT'，即使拿到 guest token 也无法调 admin 端点（被 admin_user 拦截）。
    邮箱用 UUID 后缀保证唯一性，跟真实用户邮箱不会撞。
    """
    guest_key = uuid.uuid4().hex
    cursor = db.execute(
        """
        INSERT INTO users(name, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, 'STUDENT', 1, ?)
        """,
        (
            "访客",
            f"guest-{guest_key}@campus.local",
            hash_password(guest_key),
            utc_now(),
        ),
    )
    db.commit()
    guest = db.execute(
        "SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    if not guest:
        raise HTTPException(status_code=500, detail="无法创建访客用户")
    return {
        "token": create_token(guest["id"], guest["role"]),
        "user": public_user(guest),
    }


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterBody, db: Database) -> dict[str, object]:
    """学生自助注册端点（返回 201 Created）。

    流程：
    1. email 标准化（小写，避开大小写不一致）
    2. 查重（email UNIQUE 约束兜底）→ 409 该邮箱已注册
    3. INSERT users 行，role 写死 'STUDENT'（**不能**通过这个端点注册成 ADMIN）
    4. 密码用 hash_password() 一次性 PBKDF2 哈希后存进 password_hash
    5. 返回 JWT + user（用 public_user 过滤掉 password_hash）

    与 guest 的区别：register 创建的账号有"真实密码"，可以用 `/api/auth/login` 重新登录。
    """
    email = body.email.lower()
    if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
        raise HTTPException(status_code=409, detail="该邮箱已注册")
    cursor = db.execute(
        """
        INSERT INTO users(name, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, 'STUDENT', 1, ?)
        """,
        (body.name.strip(), email, hash_password(body.password), utc_now()),
    )
    db.commit()
    user = db.execute(
        "SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    return {"token": create_token(user["id"], user["role"]), "user": public_user(user)}


@app.post("/api/auth/login")
def login(body: LoginBody, db: Database) -> dict[str, object]:
    """邮箱密码登录。

    三种响应：
    - 401：邮箱不存在 OR 密码错（合并错误信息防账号枚举攻击）
    - 403：账号存在但被 `is_active=0` 停用
    - 200：返回 JWT + user

    安全细节：
    - 错误信息统一为"邮箱或密码错误"，不让攻击者通过 401 vs 403 区分"邮箱不存在"和"密码错"
    - 密码用 verify_password（PBKDF2 + 常数时间比较），防计时攻击
    """
    user = db.execute(
        "SELECT * FROM users WHERE email = ?", (body.email.lower(),)
    ).fetchone()
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="账号已停用")
    return {"token": create_token(user["id"], user["role"]), "user": public_user(user)}


@app.get("/api/auth/me")
def me(user: Annotated[sqlite3.Row, Depends(current_user)]) -> dict[str, object]:
    """返回当前登录用户信息（无 body，纯鉴权验证端点）。

    用法：前端启动时拿 localStorage 里的 token 调一次，
    - 200 = token 有效，可以继续操作
    - 401 = token 过期或失效，需重新登录
    """


@app.get("/api/stats")
def stats(
    db: Database, _: Annotated[sqlite3.Row, Depends(current_user)]
) -> dict[str, int]:
    return {
        "documents": db.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'READY'"
        ).fetchone()[0],
        "chunks": len(rag_engine.index.docstore.docs),
        "conversations": db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
    }


@app.get("/api/documents")
def list_documents(
    db: Database, _: Annotated[sqlite3.Row, Depends(admin_user)]
) -> list[dict[str, object]]:
    rows = db.execute("""
        SELECT d.*, u.name AS uploader_name
        FROM documents d JOIN users u ON u.id = d.uploaded_by
        ORDER BY d.created_at DESC
        """).fetchall()
    return [dict(row) | {"size": int(row["size"])} for row in rows]

# test
# 111
@app.post("/api/documents", status_code=201)
async def upload_document(
    db: Database,
    user: Annotated[sqlite3.Row, Depends(admin_user)],
    title: Annotated[str, Form(min_length=2, max_length=120)],
    file: Annotated[UploadFile, File()],
) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="文件格式不支持")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="文件不能超过 10MB")

    stored_name = f"{uuid.uuid4().hex}{suffix}"
    (settings.upload_dir / stored_name).write_bytes(data)
    now = utc_now()
    cursor = db.execute(
        """
        INSERT INTO documents(
            title, filename, stored_name, mime_type, size, status,
            chunk_count, uploaded_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'PROCESSING', 0, ?, ?, ?)
        """,
        (
            title.strip(),
            file.filename or stored_name,
            stored_name,
            file.content_type or "application/octet-stream",
            len(data),
            user["id"],
            now,
            now,
        ),
    )
    # 使用 LlamaIndex 处理文档
    try:
        chunk_count = rag_engine.add_document(
            settings.upload_dir / stored_name,
            document_id=int(cursor.lastrowid),
            title=title.strip(),
            category="其他",
            db=db,
        )
        db.execute(
            """
            UPDATE documents
            SET status = 'READY', chunk_count = ?, error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (chunk_count, utc_now(), cursor.lastrowid),
        )
        db.commit()
        rag_engine.save_index()
    except Exception as exc:
        db.execute(
            "UPDATE documents SET status = 'ERROR', error = ?, updated_at = ? WHERE id = ?",
            (str(exc), utc_now(), cursor.lastrowid),
        )
        db.commit()
    row = db.execute(
        "SELECT * FROM documents WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    return dict(row)


@app.post("/api/documents/{document_id}/reprocess")
def reprocess_document(
    document_id: int,
    db: Database,
    _: Annotated[sqlite3.Row, Depends(admin_user)],
) -> dict[str, object]:
    document = db.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    path = settings.upload_dir / document["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="原始文件不存在")
    try:
        chunk_count = rag_engine.reprocess_document(
            path,
            document_id=document_id,
            title=document["title"],
            category=document["category"] or "其他",
            db=db,
        )
        db.execute(
            "UPDATE documents SET status = 'READY', chunk_count = ?, error = NULL, updated_at = ? WHERE id = ?",
            (chunk_count, utc_now(), document_id),
        )
        db.commit()
        rag_engine.save_index()
    except Exception as exc:
        db.execute(
            "UPDATE documents SET status = 'ERROR', error = ?, updated_at = ? WHERE id = ?",
            (str(exc), utc_now(), document_id),
        )
        db.commit()
    row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    db: Database,
    _: Annotated[sqlite3.Row, Depends(admin_user)],
) -> None:
    document = db.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    db.commit()
    rag_engine.delete_document(document_id)
    path = settings.upload_dir / document["stored_name"]
    if path.exists() and not document["stored_name"].startswith("seed-"):
        path.unlink()


@app.post("/api/chat/stream")
async def chat_stream(
    body: ChatBody,
    db: Database,
    user: Annotated[sqlite3.Row, Depends(current_user)],
) -> StreamingResponse:
    """SSE 流式问答端点：用户发问 → RAG 检索 → 流式生成 → 流结束一次性落库。

    完整流程：
    1. 创建/复用 conversation 行（标题 = question[:28]，首问前 28 字符）
    2. 内部 `events()` 生成器逐帧 yield SSE 给前端
    3. 流循环结束后批量插入 USER + ASSISTANT 两条消息 + 更新 conversations.updated_at
    4. 通过 StreamingResponse 返回，Content-Type: text/event-stream

    关键设计：
    - **流结束才写库**：避免半截回答污染历史记录；如果客户端中途断开，本轮不会落库
    - **conversation_id 通过 SSE meta 帧转发**：新建会话的 ID 是 INSERT 后才有，前端需要它来续问
    - **鉴权用 `current_user`**：登录后即可（STUDENT 和 ADMIN 都能调），不需要 admin 权限
    - **请求初始的 `now`** 作为 USER 消息的 created_at（早于 ASSISTANT），ASSISTANT 用 `utc_now()` 重取以反映真实生成时长
    """
    now = utc_now()
    conversation_id = body.conversation_id
    if conversation_id:
        conversation = db.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        ).fetchone()
        if not conversation:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        cursor = db.execute(
            """
            INSERT INTO conversations(user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (user["id"], body.question[:28], now, now),
        )
        conversation_id = int(cursor.lastrowid)

    # 使用 LlamaIndex RAG 引擎进行流式查询
    def events() -> Iterator[str]:
        """SSE 帧生成器：一帧一帧 yield 给 StreamingResponse。

        内部维护两个累加器：
        - `full_answer`：拼接所有 token 帧的 delta，用于落库
        - `sources`：从 meta 帧拿到的引用列表，用于落库

        流程：
        1. 调 `rag_engine.stream_query` 拿 SSE 事件流
        2. meta 帧：解析出 sources，补上 `conversation_id` 再 yield（前端需要它）
        3. token 帧：累加 `full_answer`，原样 yield 给客户端（前端按字渲染）
        4. 循环结束：把 USER + ASSISTANT 两条消息写库 + 更新会话时间

        注意：try/except 包住 JSON 解析是**防御性编程**——RAG 引擎的 SSE 协议若有
        破损，不能让服务端崩；解析失败的帧会被原样 yield 出去，前端会跳过。
        """
        full_answer = ""
        sources = []
        for event in rag_engine.stream_query(body.question, conversation_id):
            if event.startswith("event: meta"):
                try:
                    data = event.split("data: ", 1)[1]
                    parsed = json.loads(data)
                    sources = parsed.get("sources", [])
                    parsed["conversation_id"] = conversation_id
                    event = (
                        "event: meta\n"
                        f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
                    )
                except Exception:
                    pass
            elif event.startswith("event: token"):
                try:
                    data = event.split("data: ", 1)[1]
                    parsed = json.loads(data)
                    full_answer += parsed.get("text", "")
                except Exception:
                    pass
            yield event

        # 保存问答记录
        db.execute(
            """
            INSERT INTO messages(conversation_id, role, content, sources, created_at)
            VALUES (?, 'USER', ?, '[]', ?)
            """,
            (conversation_id, body.question, now),
        )
        db.execute(
            """
            INSERT INTO messages(conversation_id, role, content, sources, created_at)
            VALUES (?, 'ASSISTANT', ?, ?, ?)
            """,
            (
                conversation_id,
                full_answer,
                json.dumps(sources, ensure_ascii=False),
                utc_now(),
            ),
        )
        db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (utc_now(), conversation_id),
        )
        db.commit()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/conversations")
def list_conversations(
    db: Database, user: Annotated[sqlite3.Row, Depends(current_user)]
) -> list[dict[str, object]]:
    """列出当前用户的所有会话，按 `updated_at` 倒序（最近活跃的排前面）。

    用 LEFT JOIN + GROUP BY 一次查出会话数和最后活跃时间：
    - `COUNT(m.id) AS message_count`：每个会话的消息条数（LEFT JOIN 保证 0 消息的会话也算 0 而不被过滤）
    - `GROUP BY c.id`：按会话聚合
    - `ORDER BY c.updated_at DESC`：最近活跃的排前面（ChatPage 侧边栏用户体验）

    安全：`WHERE c.user_id = ?` 强制只能看自己的会话，跨用户数据隔离。
    无分页（校园规模够用，几千条时才需要考虑 LIMIT/OFFSET）。
    """
    rows = db.execute(
        """
        SELECT c.*, COUNT(m.id) AS message_count
        FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        """,
        (user["id"],),
    ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Database,
    user: Annotated[sqlite3.Row, Depends(current_user)],
) -> dict[str, object]:
    """获取单个会话的完整详情（含全部消息），供前端历史回放。

    返回结构：conversation 行（顶层字段）+ messages[] 数组，
    每个 message 的 `sources` 从 JSON 字符串解析回 Python list。

    用法：前端 ChatPage 在侧边栏点某个历史会话时调用一次，
    把 messages 渲染到消息列表。

    安全：双校验 `id = ? AND user_id = ?`——既验证 ID 存在，又验证属于当前用户，
    防止别人通过 URL 窥探你的会话历史。
    """
    conversation = db.execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user["id"]),
    ).fetchone()
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = db.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return dict(conversation) | {
        "messages": [
            dict(row) | {"sources": json.loads(row["sources"])} for row in messages
        ]
    }


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: int,
    db: Database,
    user: Annotated[sqlite3.Row, Depends(current_user)],
) -> None:
    """删除一个会话（连带所有消息，靠 SQL CASCADE 自动清理）。

    CASCADE 行为：
    - 删除 conversations 行后，messages 表中 conversation_id 匹配的行自动删除
    - 这是 schema 里 `FOREIGN KEY(conversation_id) REFERENCES conversations(id)
      ON DELETE CASCADE` 生效

    双校验：
    - `DELETE WHERE id = ? AND user_id = ?` 保证不能删别人的会话
    - `cursor.rowcount == 0` 处理"会话不存在"和"会话不属于你"两种情况都返回 404

    状态码 204 No Content：成功但没有返回体（HTTP DELETE 约定）。
    """
    cursor = db.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user["id"]),
    )
    db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/api/users")
def list_users(
    db: Database, _: Annotated[sqlite3.Row, Depends(admin_user)]
) -> list[dict[str, object]]:
    """管理员查看所有用户（按注册时间倒序，最新注册在最上）。

    用作 admin 后台"用户管理"页的列表。
    注意：返回的 user 信息经过 `public_user` 过滤，**不会**泄露 password_hash。
    """
    rows = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [public_user(row) for row in rows]


@app.patch("/api/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdateBody,
    db: Database,
    admin: Annotated[sqlite3.Row, Depends(admin_user)],
) -> dict[str, object]:
    """管理员修改用户角色 / 启停账号。

    支持两种修改（PATCH 语义：部分更新）：
    - `role`: "STUDENT" / "ADMIN"（角色升降）
    - `is_active`: True / False（启用 / 停用）

    body 字段为 None 时保持原值。

    关键保护：自我停用保护
    - admin 不能把自己停用（user_id == admin["id"] 且 body.is_active is False → 400）
    - 防止误把自己锁在门外无法恢复
    - 角色值用白名单 `{"STUDENT", "ADMIN"}` 校验，防止 SQL 注入或非法角色
    """
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user_id == admin["id"] and body.is_active is False:
        raise HTTPException(status_code=400, detail="不能停用当前管理员")
    role = body.role or user["role"]
    if role not in {"STUDENT", "ADMIN"}:
        raise HTTPException(status_code=400, detail="角色无效")
    active = int(body.is_active) if body.is_active is not None else user["is_active"]
    db.execute(
        "UPDATE users SET role = ?, is_active = ? WHERE id = ?",
        (role, active, user_id),
    )
    db.commit()
    updated = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public_user(updated)
