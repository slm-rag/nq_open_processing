# NQ Open + Natural Questions (validation)

Скрипты собирают **обогащённый** сплит validation **nq_open** полями из **Natural Questions** (config `dev`, split `validation`) и экспортируют данные в JSONL для задач с документами и чанками.

## Требования

- Python 3.7+ (в репозитории может использоваться локальное окружение `env/`)
- Пакет [`datasets`](https://huggingface.co/docs/datasets) (Hugging Face)

Исходный **Natural Questions** при первом запуске слияния качается с Hub и может занять заметное время и место на диске.

## Порядок работы

### 1. Слияние (Arrow на диске)

Вопросы и ответы берутся из **nq_open**; `id`, `document`, `long_answer_candidates`, `annotations` — из **natural_questions**. Совпадение по нормализованному тексту вопроса. Проверяется, что все вопросы nq_open есть в NQ.

Аргументы: `--output` — каталог для `save_to_disk` (по умолчанию `nq_open_validation_enriched`), опционально `--report`, `--cache-dir`.

```bash
python merge_nq_open_with_natural_questions.py \
  --output nq_open_validation_enriched \
  --report merge_report.json
```

### 2. Экспорт в JSONL

Из каталога `nq_open_validation_enriched` (после `load_from_disk`):

| Скрипт | Выход | Содержимое |
|--------|--------|------------|
| `export_enriched_to_jsonl.py` | `qa_pairs.jsonl` | `question_id`, `question`, `answer` |
| `export_chunks_collection.py` | `chunks_collection.jsonl` | `doc_id`, `url`, `title`, `question_id`, `chuks` (чанки с `id`, `text`) |
| `export_chunk_relevance.py` | `chunk_relevance.jsonl` | `question_id`, `documents[]` с `doc_id` и `chunks[]` (`id`, `relevant`) |
| `export_documents_collection.py` | `documents_collection.jsonl` | `question_id`, `documents[]` с `doc_id`, `text` (полный текст без HTML-токенов) |
| `collect_dataset_stats.py` | `dataset_stats.json` (по умолчанию) | сводная статистика по Arrow-датасету (ключи на русском) |

Пример пересборки только экспортов (после правок `doc_id` / `chunk_id`):

```bash
cd /path/to/nq

python export_enriched_to_jsonl.py --input nq_open_validation_enriched --output qa_pairs.jsonl

python export_chunks_collection.py --input nq_open_validation_enriched --output chunks_collection.jsonl

python export_chunk_relevance.py --input nq_open_validation_enriched --output chunk_relevance.jsonl

python export_documents_collection.py --input nq_open_validation_enriched --output documents_collection.jsonl
```

Статистика (подробности в docstring скрипта):

```bash
python collect_dataset_stats.py --input nq_open_validation_enriched --output dataset_stats.json
```

## Идентификаторы (`nq_common.py`)

- **`doc_id`**: префикс `nq_doc_` + первые 16 hex символов SHA256 от **канонического URL** (схема и хост в нижнем регистре, path после `unquote`, без лишнего `/` в конце, query отсортирован по ключам). Один URL в разных строках → один `doc_id`.

- **`chunk_id`**: префикс `nq_chunk_` + первые 16 hex SHA256 от **очищенного текста чанка** (те же правила, что при сборке текста по токенам с `is_html=False`). Одинаковый текст → одинаковый `chunk_id` (в том числе между документами). Пустой текст: в хэш идёт `doc_id`, `start_token`, `end_token`.

Текст чанка и полного документа собирается из `document["tokens"]`: токены с `is_html=True` отбрасываются при формировании строки; индексы в `long_answer_candidates` и `annotations` остаются в исходном пространстве токенов.

## Файлы в репозитории

| Файл | Назначение |
|------|------------|
| `merge_nq_open_with_natural_questions.py` | слияние HF-датасетов → Arrow |
| `nq_common.py` | нормализация URL, `doc_id`, `chunk_id`, `span_clean_text` |
| `export_enriched_to_jsonl.py` | `qa_pairs.jsonl` |
| `export_chunks_collection.py` | `chunks_collection.jsonl` |
| `export_chunk_relevance.py` | `chunk_relevance.jsonl` |
| `export_documents_collection.py` | `documents_collection.jsonl` |
| `collect_dataset_stats.py` | сводная статистика → JSON |

Каталог `nq_open_validation_enriched/` и перечисленные в `.gitignore` JSONL/отчёт обычно не коммитятся — их нужно собрать локально.

## Источники данных (Hugging Face)

- `google-research-datasets/nq_open`, split `validation`
- `google-research-datasets/natural_questions`, config `dev`, split `validation`
