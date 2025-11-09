# пропуск 1
import pandas as pd

# пропуск 2
df = pd.read_csv('titanic.csv')

# пропуск 3
print(df.head())

# пропуск 4
print(df.info())

# пропуск 5
print(df.isna().sum())

# пропуск 6
df_filled = df.fillna(0)
print(df_filled.isna().sum())

# пропуск 7
df_dropped = df.dropna()
print(df_dropped.shape)

# выбор столбца по метке
passenger_names = df['Name']
print(passenger_names.head())

# выбор нескольких столбцов
subset = df[['Name', 'Age', 'Sex']]
print(subset.head())

# выбор строк по индексу (пример: строка с индексом 0)
row0 = df.loc[0]
print(row0)

# выбор строк по условию
men_over_30 = df.loc[(df['Sex'] == 'male') & (df['Age'] > 30)]
print(men_over_30.head())

# сортировка данных по столбцу 'Age' по возрастанию
df_sorted_by_age = df.sort_values('Age', ascending=True)
print(df_sorted_by_age[['Name', 'Age']].head())

# найдите долю выживших среди всех Pclass
survival_by_pclass = df.groupby('Pclass')['Survived'].mean()
print(survival_by_pclass)

# основное задание
df = pd.read_csv('titanic.csv')
missing_before = df.isna().sum()
df_filled = df.fillna(0)
missing_after = df_filled.isna().sum()
head10 = df_filled.head(10)
df_filled['Age'] = pd.to_numeric(df_filled['Age']).fillna(0)
rows_age_gt_30 = df_filled[df_filled['Age'] > 30]
df_filled['Fare'] = pd.to_numeric(df_filled['Fare']).fillna(0)
rows_age_gt_30_sorted = rows_age_gt_30.sort_values(by='Fare', ascending=False)
mean_age_by_pclass = df_filled.groupby('Pclass')['Age'].mean()
print("Пропуски до заполнения: ", missing_before, "\n")
print("Пропуски после заполнения: ", missing_after, "\n")
print("Первые 10 строк (после заполнения пропусков нулями): \n", head10.to_string(index=False), "\n")
print("Число пассажиров возрастом больше 30:", len(rows_age_gt_30_sorted), "\n")
print("Строки с Age > 30, отсортированные по Fare (топ 20): ", rows_age_gt_30_sorted.head(20).to_string(index=False), "\n")
print("Средний Age по Pclass: ", mean_age_by_pclass, "\n")