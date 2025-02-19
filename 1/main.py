from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import skew, kurtosis, mode

from .statistical_characteristics import (
    mean,
    variance,
    std,
    coefficient_of_variation,
    median,
    mode,
    skewness,
    kurtosis
)


def get_frequency_distribution(data):
    """
    Возвращает словарь {значение: частота} и упорядоченный список (значение, частота).
    """
    counter = Counter(data)
    # Преобразуем в список пар (x, freq), отсортированный по x
    freq_list = sorted(counter.items(), key=lambda x: x[0])
    return counter, freq_list


def plot_frequency_polygon(freq_list):
    """
    Строит полигон частот (график частота-значение).
    freq_list: список кортежей (значение, частота), упорядоченный по значению.
    """
    x_values = [item[0] for item in freq_list]  # уникальные значения
    y_values = [item[1] for item in freq_list]  # частоты

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title("Полигон частот")
    plt.xlabel("Значение (возраст и т.п.)")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()


def three_sigma_check_deviation(epsilon, deviation1, deviation2, deviation3):
    if deviation1 <= epsilon and deviation2 <= epsilon and deviation3 <= epsilon:
        return True
    return False


with open('1/Moskva_2021.txt') as file:
    data = list(map(int, file.readlines()))

print(f'Всего значений: {len(data)}\n')

counter = Counter(data)
# Превращаем в список пар (значение, частота), отсортированный по значению
freq_list = sorted(counter.items(), key=lambda x: x[0])
print("Статистический ряд (значение -> частота):")
for val, freq in freq_list:
    print(f"{val} -> {freq}")

arr = np.array(data)

# 3) Строим полигон частот
x_values = [item[0] for item in freq_list]
y_values = [item[1] for item in freq_list]

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.title("Полигон частот")
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.grid(True)
plt.show()


# 4) Вычисляем основные статистические характеристики
# Среднее (mean)
mean_val = mean(arr)

# Дисперсия (выборочная) и СКО (выборочное)
# Если хотите генеральные, используйте ddof=0
var_val = variance(data)  # выборочная дисперсия
std_val = std(data)  # выборочное СКО

# Коэффициент вариации
# (если среднее 0, нужно аккуратно обрабатывать деление)
cv_val = coefficient_of_variation(data)

# Медиана
median_val = median(data)

# Мода (scipy.stats.mode возвращает ModeResult)
# По умолчанию возвращает саму моду и её частоту
# В последних версиях scipy нужно распаковывать результат:
mode_val, mode_amount = mode(arr)

# Минимальное, максимальное, размах
min_val = min(data)
max_val = max(data)
range_val = max_val - min_val

# Второе минимальное и второе максимальное значения
unique_values = sorted(set(arr))
second_smallest = unique_values[1]
second_largest = unique_values[-2]

# Асимметрия и эксцесс генеральной совокупности
# Параметр bias=False делает исправление смещения
skew_val = skewness(arr)
kurt_val = kurtosis(arr)  # по умолчанию это "избыточный" (excess) kurtosis

# 5) Выводим результаты
print("\nСтатистические характеристики (выборочные):")
print(f"   Среднее (mean): {mean_val:.2f}")
print(f"   Дисперсия (variance): {var_val:.2f}")
print(f"   СКО (std): {std_val:.2f}")
print(f"   Коэффициент вариации (CV): {cv_val:.3f}%")
print(f"   Медиана (median): {median_val}")
print(f"   Мода (mode): {mode_val} (частота = {mode_amount})")
print(f"   Минимум: {min_val}")
print(f"   Максимум: {max_val}")
print(f"   Размах: {range_val}")
print(f"   Второе минимальное значение: {second_smallest}")
print(f"   Второе максимальное значение: {second_largest}")
print(f"   Асимметрия (skewness): {skew_val:.4f}")
print(f"   Эксцесс (excess kurtosis): {kurt_val:.4f}")


# 4) Проверка правила трёх сигм
lower_limit_1 = mean_val - 1 * std_val
upper_limit_1 = mean_val + 1 * std_val

lower_limit_2 = mean_val - 2 * std_val
upper_limit_2 = mean_val + 2 * std_val

lower_limit_3 = mean_val - 3 * std_val
upper_limit_3 = mean_val + 3 * std_val

# Считаем, сколько данных попадают в этот интервал
within_one_sigmas = np.sum((arr >= lower_limit_1) & (arr <= upper_limit_1))
percentage_within_one_sigmas = (within_one_sigmas / len(arr)) * 100

within_two_sigmas = np.sum((arr >= lower_limit_2) & (arr <= upper_limit_2))
percentage_within_two_sigmas = (within_two_sigmas / len(arr)) * 100

within_three_sigmas = np.sum((arr >= lower_limit_3) & (arr <= upper_limit_3))
percentage_within_three_sigmas = (within_three_sigmas / len(arr)) * 100

epsilon = 0.1
SIGMA_BOY_1 = 68.3
SIGMA_BOY_2 = 95.4
SIGMA_BOY_3 = 99.7

deviation1 = abs(percentage_within_one_sigmas - SIGMA_BOY_1)
deviation2 = abs(percentage_within_two_sigmas - SIGMA_BOY_2)
deviation3 = abs(percentage_within_three_sigmas - SIGMA_BOY_3)

print(f"\nПроверка правила трех сигм:")
print(f'\nРазрешенное отклонение: {epsilon}%')
print(f'\n\tI уровень: {lower_limit_1:.2f} - {upper_limit_1:.2f}\t{percentage_within_one_sigmas:.2f}%\t(отклонение: {deviation1:.2f}%)')
print(f'\tII уровень: {lower_limit_2:.2f} - {upper_limit_2:.2f}\t{percentage_within_two_sigmas:.2f}%\t(отклонение: {deviation2:.2f}%)')
print(f'\tIII уровень: {lower_limit_3:.2f} - {upper_limit_3:.2f}\t{percentage_within_three_sigmas:.2f}%\t(отклонение: {deviation3:.2f}%)')

check_result = three_sigma_check_deviation(epsilon, deviation1, deviation2, deviation3)
print(f'\nВывод: таким образом, можно заключить, что распределение данных {"" if check_result else "не"} соответствует нормальному распределению')

# 5) Построение графической статистической функции распределения (кумулятивная частота)
counts, bin_edges = np.histogram(arr, bins=len(arr), density=True)
cumulative = np.cumsum(counts / np.sum(counts))  # Накопленные частоты

# Построение кумулятивной функции распределения
plt.figure(figsize=(8, 5))
plt.plot(bin_edges[1:], cumulative, linestyle='-', color='b')
plt.title("Графическая функция распределения (кумулятивная частота)")
plt.xlabel("Значение")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.show()