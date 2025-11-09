import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# чтение файла
brainFrame = pd.read_csv('brainsize.txt', delimiter='\t')

# метод head
print(brainFrame.head())

# показать первые 10 строк
print(brainFrame.head(10))

# показать последние 8 строк
print(brainFrame.tail(8))

# Общая статистика по числовым столбцам
print(brainFrame.describe())

# создание menDf и womenDf
menDf = brainFrame[brainFrame['Gender'].str.strip().str.lower() == 'male']
womenDf = brainFrame[brainFrame['Gender'].str.strip().str.lower() == 'female']

womenMeanSmarts = womenDf[['PIQ','FSIQ','VIQ']].mean(axis=1)
plt.scatter(womenMeanSmarts, womenDf['MRI_Count'])
plt.show()

# матрица корреляций для всего набора (Pearson)
num = brainFrame.select_dtypes(include=[np.number])
full_corr = num.corr(method='pearson')
print(full_corr)

# Обратите внимание на диагональ слева направо в таблице корреляции,
# сгенерированной выше. Почему диагональ заполнена значениями 1? Это совпадение? Объясните.|
# Ответ: Потому что каждая переменная полностью совпадает с самой собой, корреляция переменной с самой собой всегда равна 1.

# Продолжая смотреть на таблицу корреляции выше, обратите внимание, что значения зеркалируются;
# значения под диагональю имеют зеркальный аналог над ней. Это совпадение? Объясните.
# Ответ: Коэффициент корреляции симметричен: corr(X, Y) = corr(Y, X). Поэтому элементы выше и ниже диагонали совпадают.

women_corr = womenDf.select_dtypes(include=[np.number]).corr(method='pearson')
men_corr = menDf.select_dtypes(include=[np.number]).corr(method='pearson')

# Визуализация
wcorr = womenDf.select_dtypes(include=[np.number]).corr()
sns.heatmap(wcorr)
mcorr = menDf.select_dtypes(include=[np.number]).corr()
sns.heatmap(mcorr)

# У многих пар переменных корреляция близка к нулю. Что это значит?
# Ответ: значение коэффициента около нуля означает отсутствие заметной линейной зависимости между двумя переменными в этой выборке.

# Зачем делать разделение по полу?
# Ответ: Средние показатели мужчин и женщин в среднем достаточно сильно отличаются.

# Какие переменные имеют более сильную корреляцию с размером мозга (MRI_Count)? Это ожидалось? Объясните.
# Ответ: Рост и вес чаще всего сильнее всего связаны с размером мозга (MRI_Count),
# потому что крупные люди обычно имеют и больший объём тела, и как вследствии чуть больший размер мозга.
