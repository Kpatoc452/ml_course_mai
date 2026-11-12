# Лабораторная работа №1: регрессия (Wine Quality)

Предсказание экспертной оценки качества вина `quality` (целое число 3–9) по 11 физико-химическим признакам.
Основная метрика — **RMSE** (порог: RMSE < 0.70), дополнительная — **MAE**.

> Новичкам: подробное объяснение всех методов, формул и функций — в [THEORY.md](THEORY.md).

## Структура папки

```
regression/
├── lab01.ipynb          # решение лабораторной: код, графики, выводы
├── lab01.pdf            # текст задания
├── requirements.txt     # зависимости с зафиксированными версиями
├── best_model.joblib    # итоговая модель (создаётся ноутбуком, ~86 МБ)
├── data/
│   ├── winequality-red.csv     # 1599 красных вин (UCI)
│   ├── winequality-white.csv   # 4898 белых вин (UCI)
│   └── winequality.names       # описание датасета
├── sem01.ipynb          # материалы семинара
└── class.pptx           # материалы семинара
```

Источник данных: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality). Разделитель в CSV — `;`.

## Запуск

Из корня репозитория (`ml_course_mai`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r regression/requirements.txt
jupyter notebook regression/lab01.ipynb
```

Ноутбук запускается целиком (**Run All**). Пути к данным относительные (`data/...`), поэтому рабочей директорией должна быть папка `regression/`, а Jupyter её выставляет сам. Полный прогон занимает несколько минут: основное время уходит на подбор гиперпараметров.

Версии библиотек в `requirements.txt` совпадают с теми, на которых сохранена `best_model.joblib`. Если какая-то версия не ставится (например, из-за другой версии Python), уберите `==версия` и перезапустите ноутбук: модель переобучится и сохранится заново.

## Устройство ноутбука `lab01.ipynb`

| № | Раздел | Что делается |
|---|--------|--------------|
| 1 | Знакомство с данными | Объединение красных и белых вин, признак `is_red`, проверка пропусков и дубликатов (~18%), распределения, корреляции (в целом и по типам вин), выводы |
| 2 | Подготовка выборок | Train/test 80/20, стратификация по (тип вина, quality), `random_state=42` |
| 3 | Метрики и baseline | Разбор MAE и RMSE; baseline — константа «среднее по train» (оптимальная для RMSE) |
| 4.1 | Linear Regression | `StandardScaler + LinearRegression`, сравнение с Ridge, анализ сырых и стандартизованных коэффициентов, мультиколлинеарность |
| 4.2 | Random Forest | Подбор `max_features`, `min_samples_leaf` через GridSearchCV, MDI- и permutation-важность |
| 4.3 | Gradient Boosting | RandomizedSearchCV по 6 гиперпараметрам, важность признаков |
| 4.4 | Extra Trees | Дополнительная модель, тот же протокол |
| 5 | Сравнение моделей | Итоговая таблица (CV RMSE, test RMSE, MAE), проверка влияния дубликатов (5.1) |
| 6 | Feature importance | Сводная тепловая карта и ранги по всем моделям и EDA; почему важность ≠ причинность |
| 7 | Анализ ошибок | Распределение ошибок, ошибка по значениям quality, матрица «истина × прогноз», разбор худших объектов |
| 8 | Итоговая модель | Переобучение лучшей модели на всех данных, сохранение в `best_model.joblib`, функция `predict_quality` |
| 9 | Общие выводы | Краткое резюме |

### Протокол честного сравнения

- Все модели обучаются на одном train и оцениваются на одном test.
- Гиперпараметры и лучшая модель выбираются **только по 5-fold CV на train** (`KFold(5, shuffle=True, random_state=42)`).
- Тест используется один раз — для финальных метрик. Permutation importance на тесте считается уже после фиксации моделей и ни на какой выбор не влияет.

## Результаты (test, 1300 объектов)

| Модель | CV RMSE | RMSE | MAE |
|---|---|---|---|
| Baseline (mean) | 0.873 | 0.873 | 0.685 |
| Linear Regression | 0.735 | 0.732 | 0.567 |
| Random Forest | 0.616 | 0.612 | 0.438 |
| Gradient Boosting | 0.627 | 0.626 | 0.412 |
| **Extra Trees** | **0.606** | **0.600** | **0.391** |

- Baseline по построению равен std(quality) ≈ 0.87. Это точка отсчёта, а не кандидат на порог.
- Линейная модель недообучается: зависимости нелинейны и различаются для красных и белых вин.
- Главные признаки во всех моделях: `alcohol`, `volatile_acidity`, `free_sulfur_dioxide`, `density`.
- **Оговорка про дубликаты:** у ~26% тестовых объектов есть точная копия в train. На винах без копии RMSE ансамблей ≈ 0.70–0.73 (подробно в разделе 5.1 ноутбука).

## Использование готовой модели

Файл `best_model.joblib` содержит словарь `{"model": ExtraTreesRegressor, "features": [...]}`. Модель обучена на всех 6497 объектах.

```python
import joblib
import pandas as pd

bundle = joblib.load("best_model.joblib")
df = pd.read_csv("data/winequality-red.csv", sep=";")
df.columns = [c.replace(" ", "_") for c in df.columns]
df["is_red"] = 1                       # 1 — красное, 0 — белое
pred = bundle["model"].predict(df[bundle["features"]])
```

В ноутбуке (раздел 8) есть функция `predict_quality(df)`, которая делает то же самое автоматически:
- принимает названия колонок с пробелами (формат UCI) или с подчёркиваниями;
- тип вина берёт из колонки `is_red` (0/1) или `type` / `color` (`"red"` / `"white"`).

Ожидаемый порядок признаков:

```
fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, is_red
```
