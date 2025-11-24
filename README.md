## GitHub MCP Analyzer

Минимальный набор шагов, чтобы локально запустить GitHub MCP Server и собрать PR-метрики для `catboost/catboost`.

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

