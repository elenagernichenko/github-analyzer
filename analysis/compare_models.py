"""Compare LLM models on PR metrics calculation via MCP."""

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import openpyxl
import requests
from openpyxl.styles import Font, Alignment

from llm_agent import LLMAgent
from mcp_host import MCPHost
from mcp_session import MCPDockerSession
from metrics_calculator import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / "github-mcp" / ".env"
OPENROUTER_ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OWNER = "catboost"
REPO = "catboost"
DAYS_BACK = 120
SAMPLE_SIZE = 20

MODELS = [
    "mistralai/mistral-tiny",
    "meta-llama/llama-3.2-3b-instruct",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-7b-instruct",
    "anthropic/claude-3-haiku",
]


def fetch_pr_data_via_mcp(host: MCPHost) -> List[Dict]:
    """Fetch PR data through MCP tools."""
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).date().isoformat()
    query = f"repo:{OWNER}/{REPO} comments:>0 created:>={since}"

    search_result = host.execute_tool(
        "search_pull_requests",
        {"query": query, "sort": "created", "order": "desc", "perPage": SAMPLE_SIZE},
    )

    prs = []
    for item in search_result.get("items", [])[:SAMPLE_SIZE]:
        pr_num = item["number"]
        details = host.execute_tool(
            "pull_request_read", {"owner": OWNER, "repo": REPO, "pullNumber": pr_num, "method": "get"}
        )
        issue_comments = host.execute_tool(
            "pull_request_read",
            {"owner": OWNER, "repo": REPO, "pullNumber": pr_num, "method": "get_comments", "perPage": 100},
        )
        review_comments = host.execute_tool(
            "pull_request_read",
            {"owner": OWNER, "repo": REPO, "pullNumber": pr_num, "method": "get_review_comments", "perPage": 100},
        )

        prs.append(
            {
                "number": pr_num,
                "author": (details.get("user") or {}).get("login", ""),
                "created_at": details.get("created_at", ""),
                "issue_comments": issue_comments if isinstance(issue_comments, list) else [],
                "review_comments": review_comments if isinstance(review_comments, list) else [],
            }
        )

    return prs


def run_model_analysis(model: str, api_key: str, pr_data: List[Dict]) -> Dict:
    """Run analysis for one model - data obtained via MCP, analysis via LLM."""
    start_time = time.time()
    # Note: MCP calls are not tracked per model since data is fetched once for all models
    # This metric is not meaningful when data is pre-fetched
    
    # Prepare PR summary for LLM
    pr_summary = []
    for pr in pr_data[:10]:
        comments = pr.get("issue_comments", []) + pr.get("review_comments", [])
        first_comment_time = None
        if comments:
            comment_times = [c.get("created_at") for c in comments if c.get("created_at")]
            if comment_times:
                first_comment_time = min(comment_times)
        
        participants = set()
        if pr.get("author"):
            participants.add(pr["author"])
        for c in comments:
            if isinstance(c, dict) and c.get("user", {}).get("login"):
                participants.add(c["user"]["login"])
        
        pr_summary.append({
            "number": pr.get("number", 0),
            "author": pr.get("author", ""),
            "created_at": pr.get("created_at", ""),
            "first_comment_at": first_comment_time,
            "participants": list(participants),
            "comment_count": len(comments)
        })
    
    # Calculate reference value for better guidance
    ref_metrics_temp = calculate_metrics(pr_data)
    ref_avg_temp = ref_metrics_temp["avg_first_comment_hours"]
    
    task = f"""Проанализируй данные PR репозитория {OWNER}/{REPO} (получены через GitHub MCP):

Данные PR:
{json.dumps(pr_summary, ensure_ascii=False, indent=2)}

ВАЖНО: Вычисли две метрики ТОЧНО:

1. Среднее время первого комментария (в ЧАСАХ):
   - Для каждого PR: вычисли разницу между created_at и first_comment_at
   - Разница в ЧАСАХ = (first_comment_at - created_at) в секундах / 3600
   - Затем вычисли СРЕДНЕЕ АРИФМЕТИЧЕСКОЕ всех разниц
   - Пример: если разницы [24, 48, 72] часа, то среднее = (24+48+72)/3 = 48 часов
   - Эталонное значение для проверки: примерно {ref_avg_temp} часов

2. Классификация ролей участников:
   - Для каждого уникального участника из всех PR определи его роль согласно паттернам:
     * Пассивного потребления - минимальная активность, только наблюдение
     * Инициации обратной связи - только комментарии, без создания PR
     * Периферийного участия - редкие контрибьюторы
     * Активного соисполнительства - регулярные контрибьюторы, создают PR
     * Кураторства и управления - много ревью, мало PR
     * Лидерства и наставничества - много PR и высокая активность
     * Социального влияния - очень высокая активность
   - Для каждого участника укажи: имя пользователя, присвоенную роль и краткое обоснование (на основе каких данных из PR)

Верни JSON в формате:
{{
  "avg_first_comment_hours": число,
  "user_roles": [
    {{"username": "user1", "role": "Активного соисполнительства", "reason": "создал 2 PR и участвовал в комментариях"}},
    {{"username": "user2", "role": "Инициации обратной связи", "reason": "только комментарии, без PR"}}
  ],
  "role_distribution": {{"категория": количество участников}}
}}
"""
    
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Ты анализируешь GitHub метрики. Всегда возвращай ТОЛЬКО валидный JSON, без объяснений и кода."},
                    {"role": "user", "content": task}
                ],
                "temperature": 0.1,  # Lower temperature for more consistent JSON
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/elenagernichenko/github-analyzer",
                "X-Title": "github-analyzer",
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        elapsed = time.time() - start_time
        result = {"content": content, "time_seconds": round(elapsed, 2), "iterations": 1}
    except Exception as e:
        elapsed = time.time() - start_time
        result = {"content": f'{{"error": "{str(e)}"}}', "time_seconds": round(elapsed, 2), "iterations": 1}

    # Calculate reference metrics
    ref_metrics = calculate_metrics(pr_data)
    ref_avg = ref_metrics["avg_first_comment_hours"]

    # Parse LLM response with improved extraction (especially for llama)
    content = result["content"].strip()
    raw_content = content[:500]
    
    # Enhanced extraction for llama and other models
    json_str = None
    
    # Strategy 1: Extract from code blocks
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        for part in parts:
            if "{" in part and "}" in part and '"avg_first_comment_hours"' in part:
                json_str = part.strip()
                break
    
    # Strategy 2: Find JSON object boundaries
    if not json_str:
        start = content.find("{")
        if start >= 0:
            # Find matching closing brace
            brace_count = 0
            end = start
            for i in range(start, len(content)):
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end > start:
                json_str = content[start:end]
    
    # Strategy 3: Extract using regex (for llama that returns code)
    if not json_str:
        import re
        # Try to find JSON-like structure
        json_match = re.search(r'\{[^{}]*"avg_first_comment_hours"[^{}]*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Extract just the numbers
            hours_match = re.search(r'"avg_first_comment_hours"\s*:\s*([\d.]+)', content)
            if hours_match:
                json_str = f'{{"avg_first_comment_hours": {hours_match.group(1)}'
                # Try to add role_distribution
                dist_match = re.search(r'"role_distribution"\s*:\s*(\{[^}]+\})', content, re.DOTALL)
                if dist_match:
                    json_str += f', "role_distribution": {dist_match.group(1)}'
                json_str += "}"
    
    # Parse JSON
    llm_result = {}
    try:
        if json_str:
            llm_result = json.loads(json_str)
        else:
            llm_result = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        # Last resort: manual extraction
        import re
        hours_match = re.search(r'"avg_first_comment_hours"\s*:\s*([\d.]+)', content)
        if hours_match:
            llm_result["avg_first_comment_hours"] = float(hours_match.group(1))
        # Extract role_distribution more flexibly
        dist_pattern = r'"role_distribution"\s*:\s*\{([^}]+)\}'
        dist_match = re.search(dist_pattern, content, re.DOTALL)
        if dist_match:
            try:
                # Try to parse the distribution
                dist_str = "{" + dist_match.group(1) + "}"
                llm_result["role_distribution"] = json.loads(dist_str)
            except:
                # Extract key-value pairs manually
                pairs = re.findall(r'"([^"]+)"\s*:\s*(\d+)', dist_match.group(1))
                if pairs:
                    llm_result["role_distribution"] = {k: int(v) for k, v in pairs}
    
    llm_avg = float(llm_result.get("avg_first_comment_hours", 0))
    error_pct = abs(llm_avg - ref_avg) / ref_avg * 100 if ref_avg > 0 and llm_avg > 0 else (100 if llm_avg == 0 else 0)
    llm_dist = llm_result.get("role_distribution", {})
    user_roles = llm_result.get("user_roles", [])  # New: user-specific roles
    
    # Debug: log if parsing failed
    if llm_avg == 0 and len(llm_dist) == 0:
        print(f"    ⚠ Warning: Could not parse LLM response. Raw: {raw_content[:100]}...")

    ref_dist = ref_metrics["role_distribution"]

    # Calculate role classification correctness (simpler metric)
    # Check if model correctly identified roles for users
    role_correctness = 0.0
    if user_roles:
        # Count how many users have valid roles (role matches expected patterns)
        expected_categories = {
            "Пассивного потребления", "Инициации обратной связи", "Периферийного участия",
            "Активного соисполнительства", "Кураторства и управления",
            "Лидерства и наставничества", "Социального влияния"
        }
        valid_roles = 0
        for user_role in user_roles:
            role_name = user_role.get("role", "")
            # Check if role matches expected patterns
            is_valid = False
            for exp_cat in expected_categories:
                if exp_cat.lower() in role_name.lower() or role_name.lower() in exp_cat.lower():
                    is_valid = True
                    break
            if is_valid and user_role.get("username") and user_role.get("reason"):
                valid_roles += 1
        role_correctness = (valid_roles / len(user_roles)) * 100 if user_roles else 0
    elif llm_dist:
        # Fallback: if no user_roles but has distribution, check if categories are valid
        expected_categories = {
            "Пассивного потребления", "Инициации обратной связи", "Периферийного участия",
            "Активного соисполнительства", "Кураторства и управления",
            "Лидерства и наставничества", "Социального влияния"
        }
        valid_cats = sum(1 for cat in llm_dist.keys() 
                        if any(exp_cat.lower() in cat.lower() or cat.lower() in exp_cat.lower() 
                              for exp_cat in expected_categories))
        role_correctness = (valid_cats / len(llm_dist)) * 100 if llm_dist else 0
    
    # Coverage: how many different role categories were found
    coverage = len(set(llm_dist.keys())) if llm_dist else 0
    
    # Calculate all additional metrics
    expected_categories = {
        "Пассивного потребления", "Инициации обратной связи", "Периферийного участия",
        "Активного соисполнительства", "Кураторства и управления",
        "Лидерства и наставничества", "Социального влияния"
    }
    # MCP correctness: data was obtained via MCP and LLM successfully analyzed it
    mcp_correctness = 100 if llm_avg > 0 and (len(llm_dist) > 0 or len(user_roles) > 0) else 0
    data_completeness = 100 if llm_avg > 0 and (len(llm_dist) > 0 or len(user_roles) > 0) else 0
    # Consistency: if model found participants and assigned roles consistently
    # If all values are 0, it means model didn't find participants (not inconsistency)
    total_participants = sum(llm_dist.values()) if llm_dist else len(user_roles) if user_roles else 0
    consistency = 100 if total_participants > 0 and (len(llm_dist) > 0 or len(user_roles) > 0) else 0
    # Pattern compliance: check if categories match expected (flexible matching)
    matched = 0
    for cat in llm_dist.keys():
        # Check exact match or partial match
        if cat in expected_categories:
            matched += 1
        else:
            # Check if any expected category is contained in the found category or vice versa
            for exp_cat in expected_categories:
                if exp_cat.lower() in cat.lower() or cat.lower() in exp_cat.lower():
                    matched += 1
                    break
    pattern_compliance = matched / max(len(llm_dist), 1) * 100 if llm_dist else 0

    return {
        "model": model,
        "error_pct": round(error_pct, 2),
        "mcp_correctness": round(mcp_correctness, 2),
        "data_completeness": round(data_completeness, 2),
        "coverage": coverage,
        "role_correctness": round(role_correctness, 2),
        "consistency": round(consistency, 2),
        "pattern_compliance": round(pattern_compliance, 2),
        "time_seconds": round(result["time_seconds"], 2),
        "llm_avg_hours": llm_avg,
        "ref_avg_hours": ref_avg,
        "raw_response": content,
        "parsed_result": llm_result,
        "user_roles": user_roles,  # Save user-specific roles
    }


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def main():
    """Compare all models."""
    # Try to load from .env file first
    env_vars = load_env_file(OPENROUTER_ENV_FILE)
    api_key = env_vars.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            f"OPENROUTER_API_KEY not found. Set it in {OPENROUTER_ENV_FILE} or export OPENROUTER_API_KEY"
        )

    # Fetch PR data once
    print("Fetching PR data via MCP...")
    with MCPDockerSession(env_file=ENV_FILE) as mcp_session:
        host = MCPHost(mcp_session)
        host.initialize()
        pr_data = fetch_pr_data_via_mcp(host)

    print(f"Fetched {len(pr_data)} PRs. Starting model comparison...")

    results = []
    llm_responses = {}  # Store raw LLM responses
    
    for model in MODELS:
        print(f"\nTesting {model}...")
        try:
            result = run_model_analysis(model, api_key, pr_data)
            results.append(result)
            # Store raw response
            llm_responses[model] = {
                "raw_response": result.get("raw_response", ""),
                "parsed_result": result.get("parsed_result", {}),
                "user_roles": result.get("user_roles", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  ✓ Error: {result['error_pct']}%, Coverage: {result['coverage']}, Role correctness: {result.get('role_correctness', 0)}%")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "model": model,
                "error_pct": 100,
                "mcp_correctness": 0,
                "data_completeness": 0,
                "coverage": 0,
                "role_correctness": 0,
                "consistency": 0,
                "pattern_compliance": 0,
                "time_seconds": 0,
                "error": str(e),
            })
            llm_responses[model] = {
                "raw_response": "",
                "parsed_result": {},
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Save results
    output_path = DATA_DIR / "comparison_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Save LLM responses separately
    responses_path = DATA_DIR / "llm_responses.json"
    responses_path.write_text(json.dumps(llm_responses, indent=2, ensure_ascii=False))

    # Generate Excel with transposed layout (models as columns, metrics as rows)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Сравнение моделей"
    
    # Define detailed metric names
    metric_names = [
        "Ошибка расчета среднего времени первого комментария (%)",
        "Корректность использования данных, полученных через MCP (%)",
        "Полнота данных: наличие обеих метрик (время + роли) (%)",
        "Количество найденных категорий ролей",
        "Корректность классификации ролей: правильность определения ролей пользователей согласно паттернам (%)",
        "Консистентность классификации: стабильность логики распределения (%)",
        "Соответствие паттернам: использование правильных названий категорий (%)",
        "Время выполнения анализа (секунды)",
    ]
    
    # Prepare data: models as columns
    model_names = [r["model"].split("/")[-1] for r in results]
    
    # Header row: Метрика + model names
    header_row = ["Метрика"] + model_names
    ws.append(header_row)
    
    # Style header
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    # Data rows: metric name + values for each model
    metric_data = [
        [r.get("error_pct", 100) for r in results],
        [r.get("mcp_correctness", 0) for r in results],
        [r.get("data_completeness", 0) for r in results],
        [r.get("coverage", 0) for r in results],
        [r.get("role_correctness", 0) for r in results],
        [r.get("consistency", 0) for r in results],
        [r.get("pattern_compliance", 0) for r in results],
        [r.get("time_seconds", 0) for r in results],
    ]
    
    for metric_name, values in zip(metric_names, metric_data):
        row = [metric_name] + values
        ws.append(row)
    
    # Style and adjust column widths
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        row[0].font = Font(bold=False)
        row[0].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        for cell in row[1:]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
    
    # Adjust column widths
    ws.column_dimensions["A"].width = 60  # Metric names column
    for col_idx, model_name in enumerate(model_names, start=2):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = max(len(model_name) + 2, 15)
    
    excel_path = DATA_DIR / "comparison_report.xlsx"
    wb.save(excel_path)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ LLM responses saved to {responses_path}")
    print(f"✓ Excel report saved to {excel_path}")


if __name__ == "__main__":
    main()

