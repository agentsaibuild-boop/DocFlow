# Deploy DocFlow на Streamlit Cloud

Кратко ръководство за публикуване на DocFlow като публичен demo за 30-минутна работа.

## Какво ще получиш

- Публично достъпен URL от вида `https://<твое-име>-docflow.streamlit.app/`
- Безплатен hosting (Streamlit Cloud community tier)
- Auto-redeploy при `git push` към `main` branch на свързания GitHub repo

## Стъпка 1 — Подготовка на repo-то

Repository-то трябва да съдържа в корена:

- `app.py` — entry point
- `requirements.txt` — Python deps (вече налично в проекта)
- `.gitignore` с **поне** `.env`, `.streamlit/secrets.toml`, `eval_output/`, `output/`

Провери:

```bash
git ls-files | grep -E "\.env|secrets\.toml" && echo "⚠️ MARK: secrets in git, abort" || echo "✓ no secrets tracked"
```

Не трябва да върне нищо, освен потвърждението.

## Стъпка 2 — Регистрация в Streamlit Cloud

1. Отвори https://share.streamlit.io
2. „Sign in" → влез с GitHub акаунта, който държи repo-то
3. „New app" → избери:
   - **Repository:** `agentsaibuild-boop/DocFlow` (или твоят fork)
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. „Deploy" — първоначалното build отнема 2–5 минути

## Стъпка 3 — Добави API ключовете в Streamlit Secrets

Без ключове приложението ще се зареди, но **никой модел няма да работи**. Добавяй ключовете така:

1. От Streamlit Cloud dashboard → твоето deployment → „⚙️ Settings" → „Secrets"
2. Постави следния блок (попълни своите ключове):

```toml
GEMINI_API_KEY = "твоят_gemini_ключ"
OPENROUTER_API_KEY = "твоят_openrouter_ключ"

# Опционално, ако искаш да фиксираш точен Gemini модел
# GEMINI_MODEL = "gemini-3.1-flash-lite"
```

3. „Save" → Streamlit ще рестартира приложението автоматично

DocFlow има bridge logic в `app.py` ([`_hydrate_env_from_streamlit_secrets()`](app.py)), който копира тези стойности в `os.environ` при стартиране — extractor-ите ги виждат така, както от локален `.env`.

## Стъпка 4 — Провери deployment-а

След като приложението се рестартира:

1. В sidebar-а под „Статус на provider-и" трябва да виждаш `✅` пред моделите, чиито ключове си задал.
2. В таб „📤 Качване" качи една примерна фактура от `samples/eurofaktura_sample.jpg` или твоя.
3. Натисни „🚀 Обработи".
4. Прехвърли се на таб „📊 Резултати" → трябва да виждаш един ред с разпознатите данни.
5. Натисни „⬇️ Свали Excel" → проверка че сваленият файл се отваря коректно.

## Често срещани проблеми

| Симптом | Възможна причина | Решение |
|---|---|---|
| Build failed: `ModuleNotFoundError` | Липсваща dep в `requirements.txt` | Добави липсващото име, push, ще се rebuild-не автоматично |
| Всички модели показват ⚠️ в sidebar-а | Secrets не са зададени или името на ключа е различно | Провери Settings → Secrets; имената трябва да са **точно** `GEMINI_API_KEY` etc. |
| Стартира, но при обработка → „⏳ Моделът е претоварен" | Безплатният Gemini quota е изчерпан | Изчакай 10 мин или пробвай Qwen / Mistral |
| Стартира, но при обработка → „🔑 Грешен или липсващ API ключ" | Невалиден ключ или typo | Регенерирай ключа от dashboard-а на услугата и обнови Secrets |
| Демо-то е публично, не искам всеки да го ползва | Streamlit Cloud има private apps на paid tier | На community tier — възможно е „Settings → Sharing → Viewer access" да приема email allowlist |

## Сигурност — какво да помниш

- **API разходи са твои.** Всеки, който отвори публичното URL и качи фактура, ползва **твоите** ключове. На безплатния Gemini tier това е безопасно (има лимити), но на OpenRouter/Mistral плащаш per-token.
- Препоръка: за публично demo сложи **само Gemini** ключ. Маркирай Qwen/Mistral като недостъпни — в sidebar-а ще се появят с ⚠️.
- Качените файлове **не се записват** на диск дълго време — DocFlow ги обработва в `tempfile` и ги изтрива веднага след извличане.

## Update flow

При промени в кода:

```bash
git push origin main
```

Streamlit Cloud засича push-а, прави pull + rebuild + restart автоматично. Очаквай 1-2 минути преди новата версия да е активна.

## Ограничения на безплатния tier

- **1 GB RAM** — DocFlow върви около 200–400 MB; голям batch с много едновременни заявки може да удари тавана.
- **Sleep** след дълъг период неактивност — първото отваряне след пауза отнема ~20 секунди.
- **1 публичен app** на акаунт.
