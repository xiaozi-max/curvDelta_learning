#!/usr/bin/env bash
# ============================================================================
# feishu_notify.sh — 通用飞书群机器人通知脚本 (webhook + 签名校验)
#
# 用途
#   实验开始 / 正常结束 / 异常失败时，往飞书群发一条提醒消息，无需一直盯着终端。
#   本脚本不依赖本仓库结构，可作为函数库 source，也可独立 bash 调用，跨仓库复用。
#
# 依赖
#   - bash 3.2+
#   - python3 (只用标准库 json/hmac/hashlib/base64/urllib，负责加签、JSON 转义、HTTP POST)
#   若 python3 不可用，脚本会打印提示并跳过通知（fail-open，不影响实验本身）。
#
# 安全设置与密钥 (在飞书群里创建自定义机器人时开启“签名校验”)
#   签名算法(飞书官方)：sign = base64( HMAC_SHA256( key = "<timestamp>\n<secret>", msg = "" ) )
#
# 配置来源 (按优先级从高到低，只补未设置的项)：
#   1. 环境变量  FEISHU_WEBHOOK_URL / FEISHU_WEBHOOK_SECRET
#   2. 文件      $FEISHU_CONFIG
#   3. 文件      ${XDG_CONFIG_HOME:-$HOME/.config}/feishu/webhook.env   <-- 推荐(全局共用,在仓库外)
#   4. 文件      <本脚本所在目录>/feishu_config.env                    <-- 仓库内可选
#   配置文件每行形如：FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
#   模板见同目录 feishu_config.env.example
#   ⚠️ 不要把真实 secret 提交进 git / public 仓库；推荐只放在 $HOME/.config/feishu/webhook.env。
#
# ── Source 用法(集成进实验脚本，推荐)────────────────────────────────────
#   set -euo pipefail
#   source "common_utils/feishu_notify.sh"
#   feishu_trap_begin "$(basename "$0")"   # 结束时自动发 成功/失败(带退出码/出错行)
#   feishu_notify_start "tofu-qwen3-4b crispedit sweep"    # (可选) 开始时提醒
#   ...你的实验逻辑...
#   # 单个子任务完成/失败可手动补发(可选)：
#   feishu_notify_success "task_name" "note..."
#   feishu_notify_fail    "task_name" "note..."
#
# ── 独立 CLI 用法(不 source 也能用)────────────────────────────────────────
#   bash feishu_notify.sh text "任意文本"
#   bash feishu_notify.sh start   "实验名"
#   bash feishu_notify.sh success "实验名" "补充说明"
#   bash feishu_notify.sh fail    "实验名" "补充说明"
#   bash feishu_notify.sh test                    # 发一条测试消息，验证配置是否生效
#   FEISHU_DRY_RUN=1 bash feishu_notify.sh test   # 只打印将发送的 JSON(不发网络请求)
#
# 公共函数：
#   feishu_send_text  <文本>
#   feishu_send_card  <标题> <颜色> <正文>    颜色: blue wathet turquoise green yellow
#                                             orange red carmine violet purple indigo grey
#   feishu_notify_start / _success / _fail <标签> [补充说明]
#   feishu_trap_begin [标签]                安装 EXIT/ERR trap，自动通知最终结果
# ============================================================================

# ---- 定位脚本自身目录(无论被 source 还是独立执行) ------------------------
_FEISHU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/eaf88f02-6dee-41c1-a178-492829d87e1b"
FEISHU_WEBHOOK_SECRET="oAldHxXIZVdx5JLXlw1tUb"
# ---- 配置解析：按优先级补齐 FEISHU_WEBHOOK_URL / FEISHU_WEBHOOK_SECRET -----
_feishu_load_config() {
    [[ -n "${FEISHU_WEBHOOK_URL:-}" && -n "${FEISHU_WEBHOOK_SECRET:-}" ]] && return 0
    local f
    local files=()
    [[ -n "${FEISHU_CONFIG:-}" ]] && files+=("$FEISHU_CONFIG")
    files+=("${XDG_CONFIG_HOME:-$HOME/.config}/feishu/webhook.env")
    files+=("$_FEISHU_DIR/feishu_config.env")
    for f in "${files[@]}"; do
        [[ -f "$f" ]] || continue
        # shellcheck disable=SC1090
        set -a; source "$f"; set +a
        [[ -n "${FEISHU_WEBHOOK_URL:-}" && -n "${FEISHU_WEBHOOK_SECRET:-}" ]] && return 0
    done
    return 0
}

_feishu_host() { hostname -s 2>/dev/null || hostname 2>/dev/null || echo "unknown"; }

# ---- 底层发送：kind=text|card，经 python3 完成 加签+JSON+POST -------------
# 返回码：真正发送失败时 _FEISHU_RC!=0；但高一层默认 fail-open(返回0)，
#        仅当 FEISHU_STRICT=1 (独立 CLI 用) 才向外传播失败码。
feishu_send_card() {
    local title="$1" color="$2" body="$3"
    feishu_send_json "card" "$title" "$color" "$body"
}

feishu_send_text() {
    feishu_send_json "text" "" "" "$1"
}

feishu_send_json() {
    local kind="$1" title="$2" color="$3" body="$4"
    _feishu_load_config
    _FEISHU_RC=0
    if [[ -z "${FEISHU_WEBHOOK_URL:-}" ]]; then
        echo "[feishu] skip: FEISHU_WEBHOOK_URL 未配置(见 feishu_config.env.example)" >&2
        return 0
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[feishu] skip: 找不到 python3，无法发送飞书通知" >&2
        return 0
    fi
    FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}" \
    FEISHU_WEBHOOK_SECRET="${FEISHU_WEBHOOK_SECRET:-}" \
    FEISHU_KIND="$kind" \
    FEISHU_TITLE="$title" \
    FEISHU_COLOR="$color" \
    FEISHU_DRY_RUN="${FEISHU_DRY_RUN:-0}" \
    PYTHONUTF8=1 \
    python3 - "$body" <<'PY'
import base64, hashlib, hmac, json, os, sys, time, urllib.request, urllib.error

def _sign(secret, ts):
    # 飞书官方签名：key = "<ts>\n<secret>"，消息体为空
    key = "%s\n%s" % (ts, secret)
    digest = hmac.new(key.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")

def main():
    kind  = os.environ.get("FEISHU_KIND", "text")
    title = os.environ.get("FEISHU_TITLE", "")
    color = os.environ.get("FEISHU_COLOR", "blue")
    url   = os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
    secret= os.environ.get("FEISHU_WEBHOOK_SECRET", "").strip()
    dry   = os.environ.get("FEISHU_DRY_RUN", "0") == "1"
    body  = sys.argv[1] if len(sys.argv) > 1 else ""

    if not url:
        print("[feishu] skip: empty webhook", file=sys.stderr)
        sys.exit(0)

    if kind == "text":
        payload = {"msg_type": "text", "content": {"text": body}}
    elif kind == "card":
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "template": color if color in (
                        "blue","wathet","turquoise","green","yellow","orange","red",
                        "carmine","violet","purple","indigo","grey") else "blue",
                    "title": {"tag": "plain_text", "content": title or "通知"},
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "plain_text", "content": body}}
                ],
            },
        }
    else:
        print("[feishu] unknown msg kind: %r" % kind, file=sys.stderr)
        sys.exit(1)

    if secret:                       # 开启了“签名校验”才需要带 timestamp/sign
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        payload["sign"] = _sign(secret, ts)

    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        sys.exit(0)

    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
    except Exception as e:
        print("[feishu] network error: %s" % e, file=sys.stderr)
        sys.exit(1)

    try:
        info = json.loads(raw)
    except Exception:
        print("[feishu] non-json response: %s" % raw, file=sys.stderr)
        sys.exit(1)
    code = info.get("code", info.get("StatusCode", -1))
    msg  = info.get("msg", info.get("StatusMessage", raw))
    if code == 0:
        print("[feishu] sent OK")
        sys.exit(0)
    print("[feishu] send failed: code=%s msg=%s" % (code, msg), file=sys.stderr)
    sys.exit(1)

main()
PY
    _FEISHU_RC=$?
    if [[ "${FEISHU_STRICT:-0}" == "1" ]]; then
        return "$_FEISHU_RC"
    fi
    return 0   # fail-open：通知失败不打断实验脚本
}

# ---- 高层便利函数 ---------------------------------------------------------
feishu_notify_start() {
    local label="${1:-task}"
    feishu_send_card "🚀 开始 · $label" "blue" \
        "时间: $(date '+%Y-%m-%d %H:%M:%S')  主机: $(_feishu_host)"
}

feishu_notify_success() {
    local label="${1:-task}" extra="${2:-}"
    local body="时间: $(date '+%Y-%m-%d %H:%M:%S')  主机: $(_feishu_host)"
    [[ -n "$extra" ]] && body="$body

$extra"
    feishu_send_card "✅ 完成 · $label" "green" "$body"
}

feishu_notify_fail() {
    local label="${1:-task}" extra="${2:-}"
    local body="时间: $(date '+%Y-%m-%d %H:%M:%S')  主机: $(_feishu_host)"
    [[ -n "$extra" ]] && body="$body

$extra"
    feishu_send_card "❌ 失败 · $label" "red" "$body"
}

# ---- trap 集成：一条命令搞定“结束/失败自动提醒” ---------------------------
# feishu_trap_begin 安装 EXIT trap；若脚本用了 set -e 再装 ERR trap 报出错行。
# 内部用 _FEISHU_REPORTED 去重，失败时只发一条。
_feishu_exit_handler() {
    local rc=$?
    [[ "${_FEISHU_REPORTED:-0}" == "1" ]] && return 0
    _FEISHU_REPORTED=1
    if [[ "$rc" -eq 0 ]]; then
        feishu_notify_success "${_FEISHU_TRAP_LABEL:-task}" "exit code 0"
    else
        feishu_notify_fail "${_FEISHU_TRAP_LABEL:-task}" "提前退出，exit code $rc"
    fi
    return 0
}

_feishu_err_handler() {
    local line="${1:-?}" rc="${2:-?}"   # rc 在 trap 字符串里捕获后传入
    _FEISHU_REPORTED=1
    feishu_notify_fail "${_FEISHU_TRAP_LABEL:-task}" "第 ${line} 行命令失败 (exit $rc)"
    return 0
}

feishu_trap_begin() {
    _FEISHU_TRAP_LABEL="${1:-$(basename "${0:-bash}" .sh)}"
    _FEISHU_REPORTED=0
    trap '_feishu_exit_handler' EXIT
    case "$-" in
        *e*) trap '_feishu_err_handler $LINENO $?' ERR ;;  # $? 在 trap 执行时才展开=失败命令退出码
    esac
    return 0
}

# ---- 独立 CLI 入口 --------------------------------------------------------
_feishu_usage() {
    cat >&2 <<'EOF'
用法:
  (source) source feishu_notify.sh
           feishu_trap_begin "label"      # 结束/失败自动提醒
           feishu_notify_start/success/fail "label" ["note"]
  (CLI)    bash feishu_notify.sh {text|start|success|fail|test} [args]
配置:     环境变量 FEISHU_WEBHOOK_URL/FEISHU_WEBHOOK_SECRET，
          或 ~/.config/feishu/webhook.env (推荐)，或本目录 feishu_config.env
试发:     FEISHU_DRY_RUN=1 bash feishu_notify.sh test   # 仅打印JSON不发请求
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    export FEISHU_STRICT=1
    case "${1:-help}" in
        text)    shift; feishu_send_text "${1:?usage: feishu_notify.sh text <message>}" ;;
        start)   feishu_notify_start "${2:-experiment}" ;;
        success) feishu_notify_success "${2:-experiment}" "${3:-}" ;;
        fail|failure) feishu_notify_fail "${2:-experiment}" "${3:-}" ;;
        test|ping)
            feishu_send_text "✅ Feishu notify OK — $(date '+%F %T %Z') @ $(_feishu_host)" ;;
        -h|--help|help) _feishu_usage; exit 0 ;;
        *) _feishu_usage; exit 2 ;;
    esac
fi
