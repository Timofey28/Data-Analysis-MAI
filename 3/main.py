import json
import random

import numpy as np
from scipy.stats import norm, t, chi2, f
import math
import matplotlib.pyplot as plt

def read_data(file_path):
    """Чтение данных из файла и преобразование в список чисел"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = list(map(float, file.read().strip().split()))
    return data


def calculate_sample_size(data, gamma=0.95, delta=3):
    """Определение объема выборки"""
    t_norm = norm.ppf(gamma / 2 + 0.5)  # Коэффициент кратности ошибки
    sigma_population = np.std(data, ddof=0)  # Населенная дисперсия
    print(sigma_population)
    n = math.ceil((t_norm ** 2 * sigma_population ** 2) / delta ** 2)  # Округляем вверх
    return n


def generate_samples(data, n, num_samples=36):
    """Генерация выборок одинакового объема"""
    return [np.random.choice(data, size=n, replace=True) for _ in range(num_samples)]


def calculate_mean(sample):
    """Вычисление выборочной средней"""
    return sum(sample) / len(sample)


def compute_sample_means(samples):
    """Вычисление средних значений выборок"""
    return [calculate_mean(sample) for sample in samples]


def compute_distribution_intervals(sample_means, interval_length=1):
    """Формирование интервального ряда распределения"""
    min_mean = math.floor(min(sample_means))
    max_mean = math.ceil(max(sample_means))
    intervals = [(start, start + interval_length) for start in range(min_mean, max_mean, interval_length)]
    frequencies = [sum(1 for mean in sample_means if interval[0] <= mean < interval[1]) for interval in intervals]
    relative_frequencies = [freq / len(sample_means) for freq in frequencies]
    return intervals, frequencies, relative_frequencies


def plot_histogram(intervals, relative_frequencies):
    """Построение гистограммы"""
    bar_width = 1
    normalized_frequencies = [freq / bar_width for freq in relative_frequencies]
    plt.bar(
        [interval[0] + bar_width / 2 for interval in intervals],
        normalized_frequencies,
        width=bar_width,
        edgecolor='black',
        alpha=0.6,
        label="Гистограмма",
        align='center'
    )
    plt.title("Гистограмма выборочных средних (до аппроксимации)")
    plt.xlabel("Значения выборочных средних")
    plt.ylabel("Частота")
    plt.legend()
    plt.savefig("histogram_before.png")  # Сохранить график


def estimate_parameters(sample_means):
    """Оценка параметров методом моментов"""
    mu_estimate = sum(sample_means) / len(sample_means)
    variance_estimate = sum((x - mu_estimate) ** 2 for x in sample_means) / len(sample_means)
    sigma_estimate = math.sqrt(variance_estimate)
    return mu_estimate, sigma_estimate


def plot_gaussian_curve(mu_estimate, sigma_estimate, min_mean, max_mean):
    """Построение кривой Гаусса"""
    x = np.linspace(min_mean - 1, max_mean + 1, 1000)
    gaussian_curve = norm.pdf(x, loc=mu_estimate, scale=sigma_estimate)
    plt.plot(x, gaussian_curve, color='red', linewidth=2, label="Кривая Гаусса")


def compute_confidence_interval(selected_sample, gamma, n):
    """Вычисление доверительного интервала"""
    x_bar = calculate_mean(selected_sample)
    s = np.std(selected_sample, ddof=1)
    k = n - 1
    t_gamma = t.ppf(1 - (1 - gamma) / 2, df=k)
    margin_of_error = (t_gamma * s) / math.sqrt(n)
    return x_bar, s, t_gamma, margin_of_error

def chi_square_test(data, bins, gamma=0.95):
    """Проверка гипотезы о нормальности распределения по критерию Пирсона"""
    observed_freq, bin_edges = np.histogram(data, bins=bins)
    mu, sigma = np.mean(data), np.std(data, ddof=0)
    n = len(data)
    expected_freq = []

    for i in range(len(bin_edges) - 1):
        p = norm.cdf(bin_edges[i + 1], mu, sigma) - norm.cdf(bin_edges[i], mu, sigma)
        expected_freq.append(p * n)

    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed_freq, expected_freq) if e > 0)
    df = len(observed_freq) - 3  # -1 (сумма вероятностей = 1), -2 (оценены mu и sigma)
    chi2_crit = chi2.ppf(gamma, df)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, chi2_crit, p_value, df


def f_test(sample1, sample2, alpha=0.05):
    """Проверка равенства дисперсий по F-критерию"""
    s1_sq = np.var(sample1, ddof=1)
    s2_sq = np.var(sample2, ddof=1)

    F = max(s1_sq, s2_sq) / min(s1_sq, s2_sq)
    dfn = len(sample1) - 1
    dfd = len(sample2) - 1

    # Односторонний тест: H1: D1 > D2
    F_crit_right = f.ppf(1 - alpha, dfn, dfd)

    # Двусторонний тест: H1: D1 ≠ D2
    F_crit_left = f.ppf(alpha / 2, dfn, dfd)
    F_crit_right_2sided = f.ppf(1 - alpha / 2, dfn, dfd)

    return {
        "F": F,
        "s1_sq": s1_sq,
        "s2_sq": s2_sq,
        "crit_right": F_crit_right,
        "crit_left": F_crit_left,
        "crit_right_2sided": F_crit_right_2sided,
        "dfn": dfn,
        "dfd": dfd
    }

def build_integer_bin_edges(min_val, max_val, num_bins):
    """Построение целочисленных границ интервалов с равной длиной"""
    interval_len = math.ceil((max_val - min_val) / num_bins)
    bin_edges = list(range(min_val, max_val + interval_len, interval_len))
    return bin_edges

from scipy.stats import norm, chi2
import numpy as np
import math

def check_normality(data, num_interv=7, alpha=0.05):
    min_age = int(min(data))
    max_age = int(max(data))
    step = math.ceil((max_age - min_age) / num_interv)

    interval_age_val = []
    interval_age_freq = []

    for i in range(min_age, max_age, step):
        interval_age_val.append([i, i + step])
        freq = 0
        if i != max_age - 1:
            for j in range(i, i + step):
                freq += data.count(j)
        else:
            for j in range(i, i + step + 1):
                freq += data.count(j)
        interval_age_freq.append(freq)

    print('Интервалы + частота: ')
    for a, b in zip(interval_age_val, interval_age_freq):
        if a[0] != interval_age_val[-1][0]:
            print(f'[{a[0]}, {a[1]}) {b}')
        else:
            print(f'[{a[0]}, {a[1]}] {b}')

    print('Сумма частот: ', sum(interval_age_freq))

    mean_interval = 0
    for i, n in zip(interval_age_val, interval_age_freq):
        mean_interval += (i[0] + i[1]) / 2 * (n / sum(interval_age_freq))

    SKO_interval = 0
    for i, n in zip(interval_age_val, interval_age_freq):
        SKO_interval += ((i[0] + i[1]) / 2 - mean_interval) ** 2 * (n / sum(interval_age_freq))
    SKO_interval = math.sqrt(SKO_interval)

    print('\nВыборочное среднее (интервал): ', mean_interval)
    print('Выборочное СКО (интервал): ', SKO_interval)

    norm_interval_age_val = (np.array(interval_age_val) - mean_interval) / SKO_interval

    prob = []
    for i in norm_interval_age_val:
        if i[0] == norm_interval_age_val[0][0]:
            prob.append((norm.cdf(i[1]) - 0.5) + 0.5)
        else:
            if i[0] == norm_interval_age_val[-1][0]:
                prob.append(0.5 - (norm.cdf(i[0]) - 0.5))
            else:
                prob.append((norm.cdf(i[1]) - 0.5) - (norm.cdf(i[0]) - 0.5))

    theoretic_freq = []
    for p in prob:
        theoretic_freq.append(p * sum(interval_age_freq))

    hi_nabl = 0
    for i in range(len(prob)):
        hi_nabl += (theoretic_freq[i] - interval_age_freq[i]) ** 2 / theoretic_freq[i]

    krit = chi2.ppf(1 - alpha, len(interval_age_val) - 3)

    print('\nНаблюдаемое хи-квадрат: ', hi_nabl)
    print('Критическое хи-квадрат: ', krit)

    if hi_nabl > krit:
        print('Наблюдаемое хи-квадрат больше критического - гипотеза отвергается')
    else:
        print('Наблюдаемое хи-квадрат меньше критичеческого - гипотеза не отвергается')

def check_normality_sample_means(sample_means, num_interv=7, alpha=0.05):
    print("\n=== Проверка нормальности выборочных средних ===")

    min_val = math.floor(min(sample_means))
    max_val = math.ceil(max(sample_means))

    step = math.ceil((max_val - min_val) / num_interv)

    # Строим интервалы с целыми границами
    interval_vals = []
    interval_freqs = []

    for i in range(min_val, max_val, step):
        interval = [i, i + step]
        interval_vals.append(interval)

    for i, (start, end) in enumerate(interval_vals):
        if i != len(interval_vals) - 1:
            freq = sum(1 for x in sample_means if start <= x < end)
        else:
            freq = sum(1 for x in sample_means if start <= x <= end)
        interval_freqs.append(freq)

    total_freq = sum(interval_freqs)
    mean_interval = sum(((a + b) / 2 * (n / total_freq)) for (a, b), n in zip(interval_vals, interval_freqs))
    sko_interval = math.sqrt(sum((((a + b) / 2 - mean_interval) ** 2) * (n / total_freq)
                                 for (a, b), n in zip(interval_vals, interval_freqs)))

    print('\nВыборочное среднее (интервал): ', mean_interval)
    print('Выборочное СКО (интервал): ', sko_interval)

    norm_interval_vals = (np.array(interval_vals) - mean_interval) / sko_interval

    prob = []
    for i, (a, b) in enumerate(norm_interval_vals):
        if i == 0:
            prob.append((norm.cdf(b) - 0.5) + 0.5)
        elif i == len(norm_interval_vals) - 1:
            prob.append(0.5 - (norm.cdf(a) - 0.5))
        else:
            prob.append((norm.cdf(b) - 0.5) - (norm.cdf(a) - 0.5))

    theoretic_freq = [p * total_freq for p in prob]

    hi_nabl = sum((t - o) ** 2 / t for t, o in zip(theoretic_freq, interval_freqs) if t > 0)
    krit = chi2.ppf(1 - alpha, len(interval_vals) - 3)

    print('\nНаблюдаемое хи-квадрат: ', hi_nabl)
    print('Критическое хи-квадрат: ', krit)

    if hi_nabl > krit:
        print('Наблюдаемое χ² больше критического — гипотеза отвергается')
    else:
        print('Наблюдаемое χ² меньше критического — гипотеза не отвергается')




def main():
    file_path = '3/Moskva_2021.txt'
    data = read_data(file_path)

    gamma = 0.95
    n = calculate_sample_size(data, gamma)
    print(f"Объем выборки: {n}")

    samples = generate_samples(data, n)
    sample_means = compute_sample_means(samples)

    print("Средние значений выборок:", ", ".join(f"{mean:.2f}" for mean in sample_means))

    intervals, frequencies, relative_frequencies = compute_distribution_intervals(sample_means)

    print("\nИнтервальный ряд распределения:")
    suma = 0
    for i, interval in enumerate(intervals):
        suma += relative_frequencies[i]
        print(
            f"Интервал [{interval[0]}, {interval[1]}): Частота = {frequencies[i]}, Относительная частота = {relative_frequencies[i]:.2f}")
    print('Сумма', suma)
    plot_histogram(intervals, relative_frequencies)

    mu_estimate, sigma_estimate = estimate_parameters(sample_means)
    print(sample_means)
    print(f"\nТочечная оценка μ: {mu_estimate:.2f}")
    print(f"Точечная оценка σ: {sigma_estimate:.2f}")

    plot_gaussian_curve(mu_estimate, sigma_estimate, math.floor(min(sample_means)), math.ceil(max(sample_means)))

    plt.title("Гистограмма и аппроксимация кривой Гаусса")
    plt.xlabel("Значения выборочных средних")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.savefig("histogram.png")  # Сохранить график

    selected_sample = random.choice(samples)
    x_bar, s, t_gamma, margin_of_error = compute_confidence_interval(selected_sample, gamma, n)

    lower_bound = x_bar - margin_of_error
    upper_bound = x_bar + margin_of_error

    print("\nРезультаты расчета доверительного интервала:")
    print(f"Выборочная средняя (x): {x_bar:.2f}")
    print(f"Исправленное среднеквадратическое отклонение (s): {s:.2f}")
    print(f"Квантиль распределения Стьюдента (t_γ): {t_gamma:.2f}")
    print(f"Точность: {margin_of_error:.2f}")
    print(f"\nГраницы доверительного интервала (γ = {gamma}): [{lower_bound:.2f}, {upper_bound:.2f}]")

    print("\n=== [1а] Проверка нормальности распределения")
    num_bins = 7

    min_age = math.floor(min(data))
    max_age = math.ceil(max(data))
    bin_edges = build_integer_bin_edges(min_age, max_age, num_bins)

    if len(bin_edges) - 1 < num_bins:
        num_bins = len(bin_edges) - 1

    # Обрезаем до нужного количества интервалов
    bin_edges = bin_edges[:num_bins + 1]

    observed_freq, bin_edges = np.histogram(data, bins=bin_edges)

    grouping_intervals = [(int(bin_edges[i]), int(bin_edges[i + 1])) for i in range(len(bin_edges) - 1)]
    grouping_data = {"intervals": grouping_intervals}
    with open("grouping_from_lr3.json", "w", encoding="utf-8") as f:
        json.dump(grouping_data, f, indent=4, ensure_ascii=False)

    check_normality(data)

    check_normality_sample_means(sample_means)

    print("\n=== [2] Проверка равенства дисперсий по F-критерию ===")
    sample1, sample2 = random.sample(samples, 2)
    f_result = f_test(sample1, sample2)

    print(f"Размеры выборок: n1 = {len(sample1)}, n2 = {len(sample2)}")
    print(f"Оценка дисперсии s1² = {f_result['s1_sq']:.2f}")
    print(f"Оценка дисперсии s2² = {f_result['s2_sq']:.2f}")
    print(f"Значение F-статистики: {f_result['F']:.2f}")
    print(f"Критическое значение F (одностороннее, α = 0.05): {f_result['crit_right']:.2f}")
    print("Вывод (H1: D1 > D2):",
          "H0 не отвергается" if f_result["F"] < f_result["crit_right"] else "H0 отвергается")

    print(
        f"\nКритические значения F (двустороннее, α = 0.05): от {f_result['crit_left']:.2f} до {f_result['crit_right_2sided']:.2f}")
    if f_result["F"] < f_result["crit_left"] or f_result["F"] > f_result["crit_right_2sided"]:
        print("Вывод (H1: D1 ≠ D2): H0 отвергается")
    else:
        print("Вывод (H1: D1 ≠ D2): H0 не отвергается")


if __name__ == "__main__":
    main()
