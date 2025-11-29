## GitHub MCP Analyzer

Инструмент демонстрирует, как локальный GitHub MCP Server используется для сбора реальных данных о pull request’ах и как поверх этого работает бесплатная LLM (OpenRouter) для текстовых комментариев.

### Как устроен алгоритм

1. **MCP сервер (`github-mcp/`)**  
   - `docker-compose.yml` поднимает официальный образ `ghcr.io/github/github-mcp-server` в read-only режиме.  
   - В `.env` лежит GitHub PAT и настройки toolsets. Контейнер даёт доступ к GitHub API через MCP протокол.

2. **Обёртка над MCP (`analysis/mcp_session.py`)**  
   - Класс `MCPDockerSession` стартует контейнер через `docker run -i --rm ...`, выполняет handshake (`initialize`/`initialized`) и отправляет JSON-RPC запросы (`tools/list`, `tools/call`).  
   - Все скрипты далее используют этот класс, чтобы звать инструменты (например `search_pull_requests`, `pull_request_read`).

3. **Сбор метрик (`analysis/pr_metrics.py`)**  
   - Вытаскивает PR c комментариями за последние 120 дней, добирает детали/комментарии/ревью, считает метрики (lifetime, время первой реакции, дискуссионность).  
   - Сохраняет результаты в `data/pr_samples.json` (сырые данные) и `analysis/pr_metrics_summary.md` (краткий текст).

4. **LLM-клиент MCP (`analysis/llm_mcp_commentary.py`)**  
   - Читает `pr_metrics_summary.md` + `pr_samples.json`, дополнительно через MCP берёт свежий список PR.  
   - Формирует подсказку и шлёт её в бесплатную модель `mistralai/mistral-tiny` через OpenRouter API (ключ лежит в корневом `.env`).  
   - Ответ модели сохраняется в `data/llm_commentary.txt`.

5. **Минимальный пример LLM (`analysis/openrouter_llm.py`)**  
   - Однофразовый запрос к той же модели, результат в `data/llm_response.txt`. Можно использовать как smoke-test ключа.

### Что за что отвечает

| Файл / папка | Роль |
| --- | --- |
| `github-mcp/docker-compose.yml` | Поднимает GitHub MCP Server (read-only) |
| `github-mcp/.env` | Хранит GitHub PAT и настройки toolsets (не коммитится) |
| `analysis/mcp_session.py` | Универсальный клиент MCP (инициализация, вызовы инструментов) |
| `analysis/pr_metrics.py` | Сбор и расчёт метрик PR через MCP |
| `data/pr_samples.json` | Сырые PR-данные (результат `pr_metrics.py`) |
| `analysis/pr_metrics_summary.md` | Текстовый дайджест метрик |
| `analysis/llm_mcp_commentary.py` | Генерация комментария LLM на основе метрик + живых данных из MCP |
| `data/llm_commentary.txt` | Итоговый комментарий LLM |
| `analysis/openrouter_llm.py` | Простейший вызов OpenRouter (для проверки ключа) |

### Как запустить с нуля

1. **Клонировать проект и подготовить токены**
   ```bash
   git clone https://github.com/elenagernichenko/github-analyzer.git
   cd github-analyzer
   # GitHub PAT для MCP
   cp github-mcp/.env.example github-mcp/.env   # если файла нет
   echo "GITHUB_PERSONAL_ACCESS_TOKEN=<your_pat>" >> github-mcp/.env
   echo "GITHUB_TOOLSETS=pull_requests,issues,repos,context" >> github-mcp/.env
   echo "GITHUB_READ_ONLY=true" >> github-mcp/.env

   # OpenRouter API key (бесплатный)
   echo "OPENROUTER_API_KEY=<openrouter_key>" >> .env
   ```

2. **Поднять MCP сервер (опционально, для постоянной работы)**
   ```bash
   cd github-mcp
   docker compose up -d
   docker compose logs -n 50   # проверить, что сервер запустился
   cd ..
   ```
   > Скрипты сами вызывают `docker run` при необходимости, так что compose можно не поднимать, если сервер нужен только на время запроса.

3. **Собрать метрики PR**
   ```bash
   cd /home/spectreofoblivion/StudyProjects/ВКР/mcp-analyzer
   python3 analysis/pr_metrics.py
   ```
   - Результаты: `data/pr_samples.json` (JSON) и `analysis/pr_metrics_summary.md` (текст).

4. **Получить комментарий от LLM на основе MCP данных**
   ```bash
   set -a && source .env        # экспортируем OPENROUTER_API_KEY
   python3 analysis/llm_mcp_commentary.py
   ```
   - Сценарий автоматически запросит MCP (`search_pull_requests`), соберёт контекст и запишет вывод в `data/llm_commentary.txt`.

5. **Тестовый единичный запрос к OpenRouter (необязательно)**
   ```bash
   python3 analysis/openrouter_llm.py
   cat data/llm_response.txt
   ```

6. **Остановить MCP сервер (если поднимался через compose)**
   ```bash
   cd github-mcp
   docker compose down
   ```

Таким образом, весь pipeline “GitHub MCP Server → Python анализ → OpenRouter LLM” запускается локально и не требует платных LLM-сервисов.

### 1. Подготовка

```bash
cd /home/spectreofoblivion/StudyProjects/ВКР/mcp-analyzer
cp github-mcp/.env.example github-mcp/.env  # при необходимости
# заполните GITHUB_PERSONAL_ACCESS_TOKEN и опционально GITHUB_TOOLSETS
```

### 2. Запуск GitHub MCP Server

```bash
cd github-mcp
docker compose up -d
# проверить логи
docker compose logs -n 50
```

### 3. Сбор метрик PR

```bash
cd /home/spectreofoblivion/StudyProjects/ВКР/mcp-analyzer
python3 analysis/pr_metrics.py
```

- Сырые данные: `data/pr_samples.json`
- Краткий отчёт: `analysis/pr_metrics_summary.md`

### 4. Остановка сервера

```bash
cd github-mcp
docker compose down
```

> Скрипт `analysis/pr_metrics.py` автоматически запускает MCP Server внутри `docker run` при обращениях, так что метрики можно пересчитывать даже без поднятого `docker compose` сервиса — compose нужен, если хотите держать MCP Server постоянно.

