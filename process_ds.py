import os
import shutil
import numpy as np
import pandas as pd


class DatasetImputer:
    """
    Класс для предобработки и восстановления пропущенных значений датасета
    на основе усреднённых данных по целым секундам из всех файлов в наборе.
    
    Поддерживает:
    - Удаление некорректных/пустых файлов и папок (метод delete_empty)
    - Вычисление средних значений по секундам для каждого файла (метод mean_of_sec)
    - Агрегацию средних значений по всем файлам (make_mean_dicts и merge_mean_dicts)
    - Восстановление пропущенных секунд на основе экспоненциального сглаживания (fill_values)
    - Сортировку результатов по времени (sort_results)
    """

    def __init__(self, base_dir: str):
        """
        Инициализация обработчика.
        
        :param base_dir: Корневая директория с подпапками ('hypoxia' или  'regular')
        """
        self.base_dir = base_dir

    def mean_of_sec(self, df: pd.DataFrame) -> dict:
        """Вычисляет среднее значение по каждой целой секунде."""
        on_change_int = 0
        av_sum = 0
        av_n = 0
        t_ds = df["time_sec"]
        approximated = {}
        
        for i in range(1, len(t_ds)):
            current_sec = int(str(t_ds[i]).split('.')[0])
            if current_sec <= on_change_int:
                av_sum += df["value"][i]
                av_n += 1
            else:
                if av_n != 0:
                    approximated[on_change_int + 1] = av_sum // av_n
                else:
                    # Если нет данных для усреднения — берем предыдущее значение
                    approximated[on_change_int + 1] = df["value"][i - 1] if i > 0 else 0
                
                av_sum = df["value"][i]
                av_n = 1
                on_change_int = current_sec
        
        # Обработка последней секунды
        if av_n != 0:
            approximated[on_change_int + 1] = av_sum // av_n
        
        return approximated

    def undefined_sec(self, mean_dict: dict) -> list:
        """Находит пропущенные целые секунды в диапазоне от 1 до max."""
        if not mean_dict:
            return []
        all_secs = set(range(1, max(mean_dict.keys()) + 1))
        present = set(mean_dict.keys())
        return sorted(all_secs - present)

    def fill_values(self, min_sec: int, max_sec: int, mean_dict: dict, current_dict: dict) -> dict:
        """
        Восстанавливает пропущенные значения между min_sec и max_sec,
        двигаясь от краёв к центру.
        """
        filling = current_dict.copy()
        left = min_sec
        right = max_sec
        toggle = True  # True — левый, False — правый

        while left < right:
            if toggle:
                if left + 1 in mean_dict:
                    filling[left + 1] = (filling[left] + mean_dict[left + 1]) / 2
                else:
                    filling[left + 1] = filling[left]  # fallback
                left += 1
            else:
                if right - 1 in mean_dict:
                    filling[right - 1] = (filling[right] + mean_dict[right - 1]) / 2
                else:
                    filling[right - 1] = filling[right]
                right -= 1
            toggle = not toggle

        return filling

    def delete_empty(self, subdir: str):
        """Удаляет папки без данных или с недостаточным количеством подпапок."""
        for folder in os.listdir(self.base_dir):
            if folder == ".DS_Store":
                continue
            folder_path = os.path.join(self.base_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            contents = os.listdir(folder_path)
            if len(contents) < 2:
                shutil.rmtree(folder_path)
                continue

            subdir_path = os.path.join(folder_path, subdir)
            if os.path.exists(subdir_path) and not os.listdir(subdir_path):
                shutil.rmtree(subdir_path)

    def delete_incorrect(self):
        """Удаляет пары файлов (bpm и uterus), если они короткие или пустые."""
        for folder in os.listdir(self.base_dir):
            if folder == ".DS_Store":
                continue
            folder_path = os.path.join(self.base_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            bpm_dir = os.path.join(folder_path, "bpm")
            uterus_dir = os.path.join(folder_path, "uterus")

            if not (os.path.exists(bpm_dir) and os.path.exists(uterus_dir)):
                continue

            bpm_files = sorted(os.listdir(bpm_dir), key=str.lower)
            uterus_files = sorted(os.listdir(uterus_dir), key=str.lower)

            for i in range(min(len(bpm_files), len(uterus_files))):
                bpm_file = os.path.join(bpm_dir, bpm_files[i])
                uterus_file = os.path.join(uterus_dir, uterus_files[i])

                try:
                    df_bpm = pd.read_csv(bpm_file)
                    df_uterus = pd.read_csv(uterus_file)
                except Exception:
                    os.remove(bpm_file)
                    os.remove(uterus_file)
                    continue

                if (len(df_bpm) == 0 or len(df_uterus) == 0 or
                    df_bpm["time_sec"].max() <= 60 or len(df_bpm) <= 60 or
                    df_uterus["time_sec"].max() <= 60 or len(df_uterus) <= 60):
                    os.remove(bpm_file)
                    os.remove(uterus_file)

    def make_mean_dicts(self, subdir: str):
        """Преобразует каждый файл в усреднённый по целым секундам и перезаписывает его."""
        for folder in os.listdir(self.base_dir):
            if folder == ".DS_Store":
                continue
            subdir_path = os.path.join(self.base_dir, folder, subdir)
            if not os.path.exists(subdir_path):
                continue

            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                mean_dict = self.mean_of_sec(df)
                new_df = pd.DataFrame({
                    'time_sec': list(mean_dict.keys()),
                    'value': list(mean_dict.values())
                })
                new_df.to_csv(file_path, index=False)

    def merge_mean_dicts(self, subdir: str) -> dict:
        """Агрегирует средние значения по всем файлам в поддиректории."""
        total_sum = {}
        total_count = {}

        for folder in os.listdir(self.base_dir):
            if folder == ".DS_Store":
                continue
            subdir_path = os.path.join(self.base_dir, folder, subdir)
            if not os.path.exists(subdir_path):
                continue

            for file in os.listdir(subdir_path):
                df = pd.read_csv(os.path.join(subdir_path, file))
                for _, row in df.iterrows():
                    sec = int(row['time_sec'])
                    val = row['value']
                    total_sum[sec] = total_sum.get(sec, 0) + val
                    total_count[sec] = total_count.get(sec, 0) + 1

        return {sec: total_sum[sec] / total_count[sec] for sec in total_sum}

    def fill_undefined_pipeline(self, subdir: str, global_means: dict):
        """Восстанавливает пропущенные секунды в каждом файле на основе глобальных средних."""
        for folder in os.listdir(self.base_dir):
            if folder == ".DS_Store":
                continue
            subdir_path = os.path.join(self.base_dir, folder, subdir)
            if not os.path.exists(subdir_path):
                continue

            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                try:
                    df = pd.read_csv(file_path)
                    current_dict = dict(zip(df['time_sec'].astype(int), df['value']))
                except Exception:
                    continue

                undefined = self.undefined_sec(current_dict)
                if not undefined:
                    continue

                # Группируем последовательные пропуски
                gaps = []
                start = undefined[0]
                prev = start
                for sec in undefined[1:]:
                    if sec == prev + 1:
                        prev = sec
                    else:
                        gaps.append((start, prev))
                        start = sec
                        prev = sec
                gaps.append((start, prev))

                # Заполняем каждый промежуток
                for gap_start, gap_end in gaps:
                    left_bound = gap_start - 1
                    right_bound = gap_end + 1

                    if left_bound not in current_dict or right_bound not in current_dict:
                        continue  # Нельзя восстановить без границ

                    filled = self.fill_values(
                        min_sec=left_bound,
                        max_sec=right_bound,
                        mean_dict=global_means,
                        current_dict=current_dict
                    )

                    # Обновляем словарь
                    for sec in range(gap_start, gap_end + 1):
                        if sec in filled:
                            current_dict[sec] = filled[sec]

                # Сохраняем обратно
                result_df = pd.DataFrame({
                    'time_sec': sorted(current_dict.keys()),
                    'value': [current_dict[k] for k in sorted(current_dict.keys())]
                })
                result_df.to_csv(file_path, index=False)

    def sort_results(self, subdir: str):
        """Сортирует все файлы по времени."""
        for folder in os.listdir(self.base_dir):
            if folder == ".DS_Store":
                continue
            subdir_path = os.path.join(self.base_dir, folder, subdir)
            if not os.path.exists(subdir_path):
                continue

            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                df = df.sort_values('time_sec').reset_index(drop=True)
                df.to_csv(file_path, index=False)