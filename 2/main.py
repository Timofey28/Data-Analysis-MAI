import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

from .utils import save_array_to_file


# 1. Чтение данных из файла
with open('2/Moskva_2021.txt', 'r') as f:
    population = np.array([float(line.strip()) for line in f])
mu_pop = population.mean()
sigma_pop = population.std(ddof=0)

# 2. Определение объема выборки
gamma = 0.95
delta = 3
Z = norm.ppf((1 + gamma) / 2)
n = np.ceil((Z**2 * sigma_pop**2) / delta**2).astype(int)

# 3. Генерация 36 выборок и их средних
sample_means = [np.random.choice(population, n, replace=True).mean() for _ in range(36)]
sample_means = np.array(sample_means)
print(f'Средние выборок: {sample_means}')

# 4. Построение интервального ряда
min_mean = np.floor(sample_means.min())
max_mean = np.ceil(sample_means.max())
bins = np.arange(min_mean, max_mean + 1, 1)
freq, _ = np.histogram(sample_means, bins=bins)
relative_freq = freq / 36


# 5. Гистограмма относительных частот
plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=bins, edgecolor='black', alpha=0.7, weights=np.ones_like(sample_means)/36, label='Относительные частоты')

# 6. Выравнивание методом моментов
mu_hat = sample_means.mean()
sigma_hat = sample_means.std(ddof=0)
print(f"Среднее средних: {mu_hat}")
print(f"СКО средних: {sigma_hat}")


# Кривая Гаусса
x = np.linspace(min_mean, max_mean, 1000)
pdf = norm.pdf(x, mu_hat, sigma_hat)
plt.plot(x, pdf, 'r-', label='Кривая Гаусса')

plt.xlabel('Выборочное среднее')
plt.ylabel('Относительная частота')
plt.title('Гистограмма и кривая Гаусса')
plt.legend()
plt.grid(True)
plt.show()

# 7. Доверительный интервал для одной выборки
sample = np.random.choice(population, n, replace=True)
sample1 = np.random.choice(population, n, replace=True)
sample2 = np.random.choice(population, n, replace=True)
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
alpha = 1 - 0.95
t_crit = t.ppf(1 - alpha/2, df=n-1)
print(f'Т-критическое значение: {t_crit}')
margin = t_crit * sample_std / np.sqrt(n)
ci_low = sample_mean - margin
ci_high = sample_mean + margin

print(f"Объем выборки n: {n}")
print("-------------------------------------------------")
print(f"Доверительный интервал: ({ci_low:.2f}, {ci_high:.2f})")
print(f"Точечная оценка: {sample_mean}")
print(f"Точность доверительного интервала: {margin}")


my_array = sample_means

# Указываем имя файла для сохранения
output_filename = "sample_means.txt"

# Сохраняем массив в файл
save_array_to_file(my_array, output_filename)

save_array_to_file(sample1, "sample1.txt")
save_array_to_file(sample2, "sample2.txt")