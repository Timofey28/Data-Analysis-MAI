import math
from collections import Counter


def mean(data):
    return sum(data) / len(data)

def variance(data):
    """
    Дисперсия (генеральная). Для выборочной дисперсии
    нужно делить на (n-1).
    """
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def std(data):
    """ Среднеквадратическое отклонение (генеральное). """
    return math.sqrt(variance(data))

def coefficient_of_variation(data):
    """ Коэффициент вариации = std / mean. """
    m = mean(data)
    if m != 0:
        return std(data) / m * 100
    else:
        return float('inf')  # если среднее 0, то CV не определён

def median(data):
    """ Медиана. """
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    else:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2

def mode(data):
    """ Мода. Если мод несколько, возвращаем список. """
    counter = Counter(data)
    max_freq = max(counter.values())
    # Собираем все значения, у которых частота = max_freq
    modes = [k for k, v in counter.items() if v == max_freq]
    # Если одна мода, вернём число, иначе список
    mode = modes[0]
    return mode, max_freq

def skewness(data):
    """
    Асимметрия (простой вариант).
    skew = (1/n) * Σ((x_i - mean)/std)^3
    Для выборки можно добавить поправочные коэффициенты.
    """
    m = mean(data)
    s = std(data)
    n = len(data)
    return sum(((x - m) / s)**3 for x in data) / n

def kurtosis(data):
    """
    Эксцесс (простой вариант).
    kurt = (1/n)*Σ((x_i - mean)/std)^4 - 3
    """
    m = mean(data)
    s = std(data)
    n = len(data)
    return sum(((x - m) / s)**4 for x in data) / n - 3