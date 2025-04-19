
def save_array_to_file(array, filename):
    """
    Сохраняет элементы NumPy массива в текстовый файл,
    где каждое число находится на отдельной строке.

    Args:
        array (numpy.ndarray): NumPy массив с числами для сохранения.
        filename (str): Имя файла для сохранения данных.
    """
    try:
        with open(filename, 'w') as f:  # Открываем файл для записи ('w' - write)
            for number in array:
                f.write(str(number) + '\n')  # Записываем число и символ новой строки
        print(f"Массив успешно сохранен в файл: {filename}")
    except Exception as e:
        print(f"Ошибка при записи в файл {filename}: {e}")