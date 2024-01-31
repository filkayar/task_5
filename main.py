import pyarrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('Titanic-train.csv')


def t1(data):
    # ЗАДАНИЕ 1
    # Вывод датасета на печать
    print(data)
    # Определение количества строк и столбцов
    num_rows, num_cols = data.shape
    print(f"Количество строк: {num_rows}, количество столбцов: {num_cols}")
    # Определение типов данных
    data_types = data.dtypes
    print("Типы данных:")
    print(data_types)


def t2(data):
    # ЗАДАНИЕ 2
    # Описательные характеристики для числовых данных
    numeric_desc = data.describe()
    print("Описательные характеристики для числовых данных:")
    print(numeric_desc)
    # Статистики данных объектного типа
    object_desc = data.describe(include=['O'])
    print("\nСтатистики данных объектного типа:")
    print(object_desc)
    # Информация о типах данных/структуре в тренировочной выборке
    data_info = data.info()
    print("\nИнформация о типах данных/структуре в тренировочной выборке:")
    print(data_info)
    print("""Выводы:
    - Для каких признаков данные неполные? Age, Cabin, Embarked
    - Сколько мужчин и женщин в выборке? 577 ж и 314 м
    - В каком порту село наибольшее число пассажиров? Southampton
    - Существуют ли дубликаты номеров билетов и какой дубликат используется наибольшее число раз? да, 347082
    - Какое наибольшее число пассажиров, использующих одну и ту же каюту? Какие это каюты? по 4 пассажира B96 и B98""")


def t3(data):
    # ЗАДАНИЕ 3
    # Построение сводной таблицы класс – выживаемость
    pivot_table = pd.pivot_table(data, index='Pclass', values='Survived', aggfunc='mean')
    print("Сводная таблица класс – выживаемость:")
    print(pivot_table)
    # Построение гистограммы с накоплением для сводной таблицы
    pivot_table.plot(kind='bar', stacked=True)
    plt.title('Выживаемость в зависимости от класса')
    plt.xlabel('Класс')
    plt.ylabel('Доля выживших')
    print("Вывод: Чем выше класс - тем выше выживаемость.")
    plt.show()


def t4(data):
    # ЗАДАНИЕ 4
    # Создание столбца "family" с суммой родственников 1-го и 2-го порядков
    data['Family'] = data['SibSp'] + data['Parch']
    # Построение сводной таблицы для количества родственников и выживаемости
    pivot_table_family = pd.pivot_table(data, index='Family', values='Survived', aggfunc='mean')
    print("Сводная таблица для количества родственников и выживаемости:")
    print(pivot_table_family)
    # Построение графика для родственников 1-го порядка
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='SibSp', hue='Survived', data=data)
    plt.title('Зависимость выживаемости от количества родственников 1-го порядка')
    # Построение графика для родственников 2-го порядка
    plt.subplot(1, 2, 2)
    sns.countplot(x='Parch', hue='Survived', data=data)
    plt.title('Зависимость выживаемости от количества родственников 2-го порядка')
    # Построение графика зависимости количества родственников от выживаемости
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Family', y='Survived', data=data)
    plt.title('Зависимость выживаемости от количества родственников')
    plt.xlabel('Количество родственников')
    plt.ylabel('Выживаемость')
    print("Самая высокая выживаемость у имеющих 3 родственников")
    plt.show()


def t5(data):
    # ЗАДАНИЕ 5
    # Создание сводной таблицы для подсчета выживших и погибших по полу
    survival_by_gender = data.groupby('Sex')['Survived'].value_counts(normalize=True).unstack()
    print("Сводная таблица для подсчета выживших и погибших по полу:")
    print(survival_by_gender)
    # Построение гистограммы с накоплением для выживших и погибших по полу
    survival_by_gender.plot(kind='bar', stacked=True)
    plt.title('Выжившие и погибшие по полу')
    plt.xlabel('Пол')
    plt.ylabel('Доля от общего количества')
    print("Вывод: У мужчин выживаемость значительно хуже")
    plt.show()


def t6(data):
    # ЗАДАНИЕ 6
    # Посмотрим, как заполнено поле Age (посчитаем ненулевые значения)
    # print("Количество ненулевых значений в поле 'Age':", data.PassengerId[data.Age.notnull()].count())
    print("Количество ненулевых значений в поле 'Age':", data['Age'].count())
    # Заполним недостающие данные медианным значением
    median_age = data['Age'].median()
    # data.Age[data.Age.isnull()] = data.Age.median()
    data['Age'].fillna(median_age)

    # Снова проверим заполнение поля Age
    print("Количество ненулевых значений в поле 'Age' после заполнения:", data['Age'].count())

    # Определим минимальный и максимальный возраст
    min_age = data['Age'].min()
    max_age = data['Age'].max()
    print("Минимальный возраст:", min_age)
    print("Максимальный возраст:", max_age)

    # Построим гистограмму распределения возрастов пассажиров
    plt.figure(figsize=(8, 5))
    data['Age'].plot.hist(bins=8, edgecolor='black')
    plt.title('Распределение возрастов пассажиров')
    plt.xlabel('Возраст')
    plt.ylabel('Частота')
    plt.show()
    print("Больше всего было пассажиров возраста между 20 и 30 годами")

    # Построим таблицы частот возрастов для выживших и погибших
    survived_ages = data[data['Survived'] == 1]['Age']
    died_ages = data[data['Survived'] == 0]['Age']
    survived_age_freq = survived_ages.value_counts().sort_index()
    died_age_freq = died_ages.value_counts().sort_index()
    print("Таблица частот возрастов выживших:\n", survived_age_freq)
    print("Таблица частот возрастов выживших:\n", died_age_freq)

    # Построим гистограммы распределения возрастов выживших и погибших
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 2, 1)
    survived_ages.plot.hist(bins=8, edgecolor='black')
    plt.title('Распределение возрастов выживших')
    plt.xlabel('Возраст')
    plt.ylabel('Частота')
    plt.subplot(1, 2, 2)
    died_ages.plot.hist(bins=8, edgecolor='black')
    plt.title('Распределение возрастов погибших')
    plt.xlabel('Возраст')
    plt.ylabel('Частота')
    plt.show()
    print("Вывод: Лучшая выживаемость у детей до 10 лет")


def t7(data):
    # ЗАДАНИЕ 7
    # Создадим три подграфика
    fig = plt.figure(figsize=[15,10])
    axes = []
    axes.append(fig.add_subplot(211))
    axes.append(fig.add_subplot(223))
    axes.append(fig.add_subplot(224))
    # a) Распределения возрастов выживших и погибших для всех пассажиров
    data[data['Survived'] == 0]['Age'].plot.hist(ax=axes[0], alpha=0.5, color='red', bins=range(0, 81, 10), edgecolor='black')
    data[data['Survived'] == 1]['Age'].plot.hist(ax=axes[0], alpha=0.5, color='blue', bins=range(0, 81, 10), edgecolor='black')
    axes[0].set_title('All Passengers')
    axes[0].legend(['Died', 'Survived'])

    # b) Распределения возрастов выживших и погибших для лиц мужского пола
    data[(data['Survived'] == 0) & (data['Sex'] == 'male')]['Age'].plot.hist(ax=axes[1], alpha=0.5, color='red', bins=range(0, 81, 10), edgecolor='black')
    data[(data['Survived'] == 1) & (data['Sex'] == 'male')]['Age'].plot.hist(ax=axes[1], alpha=0.5, color='blue', bins=range(0, 81, 10), edgecolor='black')
    axes[1].set_title('Male Passengers')
    axes[1].legend(['Died', 'Survived'])

    # c) Распределения возрастов выживших и погибших для лиц женского пола
    data[(data['Survived'] == 0) & (data['Sex'] == 'female')]['Age'].plot.hist(ax=axes[2], alpha=0.5, color='red', bins=range(0, 81, 10), edgecolor='black')
    data[(data['Survived'] == 1) & (data['Sex'] == 'female')]['Age'].plot.hist(ax=axes[2], alpha=0.5, color='blue', bins=range(0, 81, 10), edgecolor='black')
    axes[2].set_title('Female Passengers')
    axes[2].legend(['Died', 'Survived'])

    print("Вывод: У женщин больше шансов на спасение")
    plt.show()


def t8(data):
    # ЗАДАНИЕ 8
    # Построение сводной таблицы для количества родственников и выживаемости
    pivot_table_fare = pd.pivot_table(data, index='Survived', values='Fare', aggfunc='mean')
    print("Сводная таблица для стоимости билета и выживаемости:")
    print(pivot_table_fare)
    # Построение графика зависимости количества родственников от выживаемости
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Survived', y='Fare', data=data)
    plt.title('Зависимость выживаемости от стоимости билета')
    plt.ylabel('Стоимость билета')
    plt.xlabel('Выживаемость')
    print("Самая высокая выживаемость у владельцев более дорогих билетов")
    plt.show()


def t9(data):
    # ЗАДАНИЕ 9
    # Подсчет пассажиров севших в каждом порту
    passengers_by_port = data['Embarked'].value_counts()
    print(passengers_by_port)
    # Рассчет доли спасенных в зависимости от порта посадки
    survival_by_port = data.groupby('Embarked')['Survived'].mean()
    # Построение диаграммы с накоплением
    survival_by_port.plot(kind='bar', title='Доля спасенных в зависимости от порта посадки', xlabel='Порт посадки',
                          ylabel='Доля выживших', color='skyblue', alpha=0.75)
    print("Из пассажиров Шербурга больше половины выжили")
    plt.show()


def t10(data):
    # ЗАДАНИЕ 10
    # рассчет корреляционной матрицы
    numeric_data = data.select_dtypes(include='number')
    correlation_matrix = numeric_data.corr()
    # печать корреляционной матрицы
    print(correlation_matrix)
    print("Наибольшее влияние оказали класс и стоимость билета")


task = -1
while(task != 0):
    task = int(input("Введите номер задания (1-10 или 0 для выхода):"))
    match(task):
        case 1:
            t1(data)
        case 2:
            t2(data)
        case 3:
            t3(data)
        case 4:
            t4(data)
        case 5:
            t5(data)
        case 6:
            t6(data)
        case 7:
            t7(data)
        case 8:
            t8(data)
        case 9:
            t9(data)
        case 10:
            t10(data)
