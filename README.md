# RiskGate: нейромережеве визначення класу ризику транзакцій і адаптивне коригування протоколу їх обробки

Програмна частина дисертаційного дослідження Аскерова В. В. «МЕТОДИ І ЗАСОБИ ЗАБЕЗПЕЧЕННЯ ЗАХИЩЕНИХ ТРАНЗАКЦІЙ ДЛЯ БЛОКЧЕЙН-СИСТЕМ З АДАПТИВНИМ КОРИГУВАННЯМ ПРОТОКОЛУ ВИКОНАННЯ ТРАНЗАКЦІЙ» (спеціальність 123 «Комп'ютерна інженерія»,
Хмельницький національний університет).

Контур обробки транзакції перед консенсусом:

```
профіль транзакції (188 ознак) → нейромережевий risk engine → невизначеність MC-Dropout
  → зонування WHITE / GRAY / BLACK → протокольна директива STD / ENH / QUAR
  → черги mempool / inspection / quarantine → журнал рішень (hash-chain + Ed25519)
```

Експерименти виконано на відкритому наборі [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
(203 769 транзакцій, 234 355 ребер графа переказів) із суворим часовим розбиттям.

## Основні результати

Тестова вибірка — часові кроки 35–49 (16 670 розмічених транзакцій); усі пороги добираються лише на валідації (кроки 31–34).

| Модель | Параметрів | ROC-AUC | AUPRC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Логістична регресія | — | 0.855 | 0.237 | 0.181 | 0.784 | 0.294 |
| Random Forest (3 запуски) | — | 0.928–0.930 | 0.788–0.789 | 0.752–0.931 | 0.715–0.733 | 0.742–0.809 |
| HistGradientBoosting | — | 0.931 | 0.790 | 0.826 | 0.721 | 0.770 |
| MLP 256-128 | 81 409 | 0.853 | 0.453 | 0.605 | 0.470 | 0.529 |
| Трансформер з токеном на ознаку (3 епохи) | 23 569 | 0.907 | 0.671 | 0.410 | 0.657 | 0.505 |
| **Груповий трансформер + PLE (робоча модель)** | **1 306 881** | **0.9365** | **0.7887** | **0.674** | **0.729** | **0.700** |

Робоча точка — F1-оптимальний поріг, обраний на валідації. Стабільність робочої архітектури за п'ятьма
початковими значеннями генератора: ROC-AUC 0.913 ± 0.025, AUPRC 0.772 ± 0.019.

Адаптивна політика зонування П-А на тесті: повнота ескалації (GRAY ∪ BLACK) 0.875, точність карантину
(BLACK) 0.992 за повноти 0.610, частка ескальованих транзакцій 0.254, пропущено у WHITE 12.5 %,
середня вартість обробки 2.48 у.о./тх.

Швидкодія (Intel Core i5-10210U, 8 ГБ ОЗП, Windows 10 Pro, Python 3.12.0, PyTorch 2.3.0 CPU, 6 потоків):
наскрізний шлюз додає 1.34 мс на транзакцію (потік 20 000 тх, пакет 256, MC-Dropout T = 5),
пропускна здатність зі шлюзом 703 тх/с; INT8-квантування не погіршує якість (ΔAUPRC < 0.001).

Повні числові результати — у [`results/`](results).

## Швидкий старт

Потрібен Python 3.10 або новіший (перевірено на 3.12.0) і ~2 ГБ вільного місця (дані ~0.7 ГБ,
решта — бібліотеки).

```bash
git clone https://github.com/am404sv/riskgate.git
cd riskgate
python -m venv .venv
```

Активуйте середовище (`.venv\Scripts\activate` у Windows або `source .venv/bin/activate` у Linux/macOS),
потім:

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` завантажує набір Elliptic (~150 МБ), будує профілі, оцінює постачену модель
`results/model_gtt.pt`, добирає політики зонування, виконує аналіз чутливості, бенчмарки і
демонстрацію шлюзу (~30 хв на 4-ядерному CPU, з них ~15 хв — бенчмарки). Повне відтворення
з навчанням усіх моделей: `python run_all.py --train` (~2 год).

## Кроки конвеєра

| Крок | Команда | Що робить | Результат |
|---|---|---|---|
| 1 | `python src/download_elliptic.py` | завантаження набору (PyG, резерв — Hugging Face) | `data/elliptic/*.csv` |
| 2 | `python src/prep_data.py` | профіль 188 ознак, розбиття 26 905 / 2 989 / 16 670 | `data/prepared/elliptic_188.npz` |
| 3 | `python src/train_models.py` | навчання MLP, трансформера з токеном на ознаку, GTT+PLE (~20 хв) | `results/model_*.pt` |
| 4 | `python src/evaluate.py` | MC-Dropout (T = 30), калібрування, робочі точки, зонування | `results/results_main.json`, `scores_gtt.npz` |
| 5 | `python src/policies.py` | політики з фіксованими порогами П-Б і П-Е | `results_main.json → gtt.policies` |
| 6 | `python src/pattern_boost.py` | патерн-рівень: ризик оточення ρ підвищує WHITE → GRAY | `results_main.json → gtt.pattern_boost` |
| 7 | `python src/adaptive_policy.py` | адаптивна квантильна політика П-А | `results_main.json → gtt.policy_adaptive` |
| 8 | `python src/analysis_extra.py` | збіжність u(T), чутливість порогів, абляція радіуса, бейзлайни, 5 запусків, Integrated Gradients (`--fast` — без навчання) | `results/results_analysis.json`, `sensitivity_grid.json` |
| 9 | `python src/bench_speed.py` | затримка інференсу, INT8, журнал Ed25519, режими STD/ENH/QUAR, наскрізний шлюз | `results/results_speed.json` |
| 10 | `python src/gateway_service.py --n 5000` | демонстрація шлюзу на потоці підписаних транзакцій | `results/session_report.json`, `journal.jsonl` |

Крок 3 перезаписує постачену модель; повернути її можна командою `git checkout results/`.

## Структура репозиторію

```
src/
  common.py            шляхи, завантаження вибірок, JSON
  features.py          профіль транзакції: 165 базових + часовий крок + 22 контекстні ознаки (r ≤ 2)
  models.py            MLP, груповий трансформер (GTT), PLE, GTT+PLE, INT8-квантування
  engine.py            навчання, MC-Dropout, температурне калібрування, пороги, зонування
  download_elliptic.py, prep_data.py, train_models.py, evaluate.py,
  policies.py, pattern_boost.py, adaptive_policy.py, analysis_extra.py,
  bench_speed.py, gateway_service.py      кроки конвеєра
results/
  model_gtt.pt         ваги робочої моделі (GTT+PLE, 5.3 МБ)
  ple_edges.npy        межі квантильних бінів PLE (за навчальною вибіркою)
  results_*.json       числові результати, наведені в дисертації
run_all.py             запуск усього конвеєра
```

## Модель і політики

**Профіль транзакції** (`features.py`): 93 локальні та 72 агреговані ознаки Elliptic, часовий крок і
22 контекстні ознаки графа переказів — степені вершини, середні ознаки попередників і наступників,
охоплення другого кола, кількість сусідів у тому самому часовому кроці, часовий розмах оточення.
Усі контекстні ознаки обчислюються за O(|E|).

**Risk engine** (`models.py`): кожна ознака кодується кусково-лінійно за 24 квантильними бінами (PLE,
Gorishniy et al., NeurIPS 2022); три смислові групи ознак проєктуються в 4 токени кожна, самоувага
працює над 13 токенами (3 шари, 4 голови, d = 64) замість 188 пофічевих, що зменшує квадратичний
член складності приблизно у 209 разів. Рішення приймається за згладженою оцінкою MC-Dropout із
температурним калібруванням.

**Зонування** (`engine.py`): BLACK — s̄ ≥ t_b; WHITE — s̄ < t_w і u < u_max; решта — GRAY.
Середня вартість обробки обчислюється з нормованими вагами режимів STD / ENH / QUAR, прийнятими
в дисертації: 1 / 6.702 / 7.472 у.о. на транзакцію.

**Адаптивна політика П-А** (`adaptive_policy.py`): t_w, u_max і патерн-поріг t_p — квантилі вікна
потоку (q_w = 0.775, q_u = 0.90, q_p = 0.95), тож мітки в роботі шлюзу не потрібні; t_b абсолютний,
мінімальний з точністю карантину ≥ 0.99 на валідації.

**Шлюз** (`gateway_service.py`): перевірка підпису Ed25519, оцінювання ризику й невизначеності,
зонування, маршрутизація в черги і журнал рішень, де кожен запис містить hash попереднього та
підпис Ed25519; перевірка журналу виявляє підміну будь-якого поля.

## Відтворюваність

- Профіль ознак і розбиття відтворюються точно; оцінювання постаченої моделі (`evaluate.py` і
  кроки 5–7) відтворює `results/results_main.json` без розбіжностей.
- Повторне навчання дає близькі, але не побітово однакові значення через багатопотокові обчислення.
- Показники швидкодії залежать від обладнання; числа в `results_speed.json` отримано на конфігурації,
  наведеній вище.

## Дані

Набір Elliptic не входить до репозиторію: його завантажує `download_elliptic.py`. Опис набору:
Weber M. et al. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks
for Financial Forensics. KDD Workshop on Anomaly Detection in Finance, 2019. Умови використання —
на сторінці набору на Kaggle.

## Ліцензія

[MIT](LICENSE)

---

## English summary

Code for the PhD research of V. V. Askerov (Khmelnytskyi National University, specialty 123
"Computer Engineering") on Methods and Tools for Ensuring Secure Transactions in Blockchain Systems 
with Adaptive Adjustment of the Transaction Execution Protocol. A 188-feature transaction profile
built from the Elliptic Bitcoin Dataset feeds a group-token transformer with piecewise-linear feature
encoding (GTT+PLE); MC-Dropout uncertainty and calibrated scores drive WHITE/GRAY/BLACK zoning, which
maps to pre-consensus processing modes and an Ed25519-signed hash-chained decision journal.
On the temporal test split (time steps 35–49) the model reaches ROC-AUC 0.9365 and AUPRC 0.7887.
Install the requirements and run `python run_all.py` to download the data and reproduce the results.
