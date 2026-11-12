# Происхождение данных

Fanaee-T, H. (2013). **Bike Sharing**. UCI Machine Learning Repository.
[DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894).

- [Карточка датасета](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
- [Официальный ZIP](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip).
- Лицензия: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Дата получения: 12 ноября 2026 года.
- `hour.csv` и `Readme.txt` извлечены из ZIP без изменения содержимого.
- `day.csv` не используется и не включён.

В `hour.csv` фактически 17 379 строк и 17 полей. Один объект — зарегистрированный час
работы Capital Bikeshare в 2011–2012 годах. Число объектов определяется по CSV;
число на карточке UCI может отличаться. Авторское описание сохранено в `Readme.txt`.

SHA-256 `hour.csv`:

```text
e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f
```

Преобразования выполняются только в памяти ноутбука: сезон из месяца, четыре периода
суток, три уровня нормированной температуры, объединение кодов погоды 3 и 4,
три уровня спроса. Полные правила приведены в ноутбуке и `results/run_manifest.json`.
