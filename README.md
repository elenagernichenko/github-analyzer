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

### 3.1. Сравнение моделей LLM

Создайте файл `.env` в корне проекта (если его еще нет) и добавьте ключ OpenRouter:

```bash
cd /home/spectreofoblivion/StudyProjects/ВКР/mcp-analyzer
echo "OPENROUTER_API_KEY=your-key-here" >> .env
```

Запуск сравнения:

```bash
cd /home/spectreofoblivion/StudyProjects/ВКР/mcp-analyzer
python3 analysis/compare_models.py
```

Скрипт автоматически прочитает `OPENROUTER_API_KEY` из файла `.env` в корне проекта.

- Результаты: `data/comparison_results.json`
- Отчёт: `data/comparison_report.md`

Сравнивает 5 моделей по метрикам:
- Ошибка расчета среднего времени первого комментария
- Покрытие категорий ролей участников
- Энтропия распределения ролей
- Количество MCP вызовов
- Время выполнения

### 4. Остановка сервера

```bash
cd github-mcp
docker compose down
```

> Скрипт `analysis/pr_metrics.py` автоматически запускает MCP Server внутри `docker run` при обращениях, так что метрики можно пересчитывать даже без поднятого `docker compose` сервиса — compose нужен, если хотите держать MCP Server постоянно.

