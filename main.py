import pyarrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ЗАДАНИЕ № 1
    dts = pd.read_csv('Titanic-train.csv')
    # Выведем датасет
    print(dts)
    # Найдем число строк и столбцов и типы данных
    print('rows - columns: ' + str(dts.shape))
    print('data types: ')
    print(dts.dtypes)

    # ЗАДАНИЕ №2
    # описательные характеристики для числовых данных
    print("описательные характеристики для числовых данных")
    print(dts.describe())
    # статистики данных объектного типа
    print("статистики данных объектного типа")
    print(dts.describe(include=['O']))
    # больше информацию о типах данных/структуре в тренировочной выборке
    print("больше информацию о типах данных/структуре в тренировочной выборке")
    print(dts.info())
    print("Выводы:")
    print("Для каких признаков данные неполные? Age, Cabin, Embarked")
    print("Сколько мужчин и женщин в выборке? 577 ж и 314 м")
    print("В каком порту село наибольшее число пассажиров? Southampton")
    print("Существуют ли дубликаты номеров билетов и какой дубликат используется наибольшее число раз? да, 347082")
    print("Какое наибольшее число пассажиров, использующих одну и ту же каюту? Какие это каюты? по 4 пассажира B96 и B98")


    # ЗАДАНИЕ №3
    print("Определить влияние класса на то, спасся человек или нет.")
    dl = dts.pivot_table('PassengerId', 'Pclass', 'Survived', 'count')
    print(dl)
    dl.plot(kind='bar', stacked=True)

    print("Чем выше класс, тем больше выживших в процентах от общего числа")

    # ЗАДАНИЕ №4
    print("Как количество родственников 1-го и 2-го порядков влияет на вероятность спасения?")
    d2 = dts.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count')
    d3 = dts.pivot_table('PassengerId', ['Parch'], 'Survived', 'count')
    print(d2)
    print("Лучшая выживаемость у имеющих 1-2 родственников")

    # графики
    fig, axes = plt.subplots(ncols=2)
    d2.plot(ax=axes[0], title='SibSp')
    d3.plot(ax=axes[1], title='Parch')

    plt.show()