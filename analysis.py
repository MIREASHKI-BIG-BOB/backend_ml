import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  




class CTGDatasetBuilder:
    """
    Сборщик обучающего датасета + обучение и использование классификатора гипоксии.
    """

    # Колонки признаков (без целевой переменной)
    FEATURE_COLUMNS = [
        "mean_bpm",
        "variability",
        "accel_per_min",
        "decel_per_min",
        "uc_per_min",
        "base_uc_tone"
    ]
    TARGET_COLUMN = "hypoxia"

    def __init__(self, hypoxia_dir: str, regular_dir: str, model_path: str = "hypoxia_model.joblib"):
        """
        :param hypoxia_dir: путь к данным с гипоксией
        :param regular_dir: путь к нормальным данным
        :param model_path: путь для сохранения/загрузки обученной модели
        """
        self.hypoxia_dir = hypoxia_dir
        self.regular_dir = regular_dir
        self.model_path = model_path
        self.model = None

    # ... (все методы _extract_metrics_from_pair, _process_directory, build_train_dataset остаются без изменений)

    def train_model(self, dataset_path: str = None, test_size: float = 0.2, random_state: int = 42):
        """
        Обучает MLP-классификатор на датасете и сохраняет его.
        
        :param dataset_path: путь к CSV с датасетом. Если None — соберёт автоматически.
        :param test_size: доля тестовой выборки
        :param random_state: для воспроизводимости
        """
    
        if dataset_path is None:
            print("Сборка датасета...")
            df = self.build_train_dataset("temp_train_dataset.csv")
            dataset_path = "temp_train_dataset.csv"
        else:
            print(f"Загрузка датасета из {dataset_path}...")
            df = pd.read_csv(dataset_path)
        required_cols = self.FEATURE_COLUMNS + [self.TARGET_COLUMN]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"В датасете отсутствуют колонки: {missing}")

        X = df[self.FEATURE_COLUMNS]
        y = df[self.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

       
        print("Обучение MLP-классификатора...")
        self.model = MLPClassifier(random_state=random_state, max_iter=1000)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("\n Отчёт по классификации на тестовой выборке:")
        print(classification_report(y_test, y_pred, target_names=["Regular", "Hypoxia"]))

        joblib.dump(self.model, self.model_path)
        print(f" Модель сохранена в {self.model_path}")

    def load_model(self):
        """Загружает обученную модель с диска."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"Модель загружена из {self.model_path}")

    def predict(self, X):
        """
        Возвращает вероятности гипоксии для новых данных.
        
        :param X: DataFrame или массив признаков (должен соответствовать FEATURE_COLUMNS)
        :return: массив вероятностей [P(regular), P(hypoxia)]
        """
        if self.model is None:
            self.load_model()
        
        if isinstance(X, pd.DataFrame):
            X = X[self.FEATURE_COLUMNS]  
        
        return self.model.predict_proba(X)

    def predict_hypoxia(self, X):
        """(0 — regular, 1 — hypoxia)."""
        if self.model is None:
            self.load_model()
        if isinstance(X, pd.DataFrame):
            X = X[self.FEATURE_COLUMNS]
        return self.model.predict(X)

    def _extract_metrics_from_pair(self, bpm_path: str, uterus_path: str) -> dict:
        """Извлекает метрики из пары файлов (bpm + uterus)."""
        try:
            df_bpm = pd.read_csv(bpm_path)
            df_uterus = pd.read_csv(uterus_path)
        except Exception:
            return None

        # Убедимся, что оба файла содержат колонку 'value'
        if 'value' not in df_bpm.columns or 'value' not in df_uterus.columns:
            return None

        min_len = min(len(df_bpm), len(df_uterus))
        if min_len == 0:
            return None

        fhr_series = df_bpm['value'].iloc[:min_len]
        uc_series = df_uterus['value'].iloc[:min_len]

        try:
            from your_module_with_analyze import analyze_ctg_data  # или импортируйте здесь
            metrics = analyze_ctg_data(fhr_series, uc_series, fs=4)
            return {
                "mean_bpm": metrics["mean_fhr_bpm"],
                "variability": metrics["variability_bpm"],
                "accel_per_min": metrics["accelerations_per_min"],
                "decel_per_min": metrics["decelerations_per_min"],
                "uc_per_min": metrics["uc_per_min"],
                "base_uc_tone": metrics["mean_uc_tone"]
            }
        except Exception:
            return None

    def _process_directory(self, base_dir: str, label: int) -> pd.DataFrame:
        """Обрабатывает одну директорию и возвращает датафрейм с меткой."""
        records = []
        if not os.path.exists(base_dir):
            print(f" Директория {base_dir} не найдена ")
            return pd.DataFrame()

        for folder in os.listdir(base_dir):
            if folder == ".DS_Store":
                continue
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            bpm_dir = os.path.join(folder_path, "bpm")
            uterus_dir = os.path.join(folder_path, "uterus")

            if not (os.path.exists(bpm_dir) and os.path.exists(uterus_dir)):
                continue

            bpm_files = sorted(os.listdir(bpm_dir))
            for bpm_file in bpm_files:
                if not bpm_file.endswith(".csv"):
                    continue
                uterus_file = bpm_file.replace(".csv", "_2.csv")
                uterus_path = os.path.join(uterus_dir, uterus_file)
                bpm_path = os.path.join(bpm_dir, bpm_file)

                if not os.path.exists(uterus_path):
                    continue

                metrics = self._extract_metrics_from_pair(bpm_path, uterus_path)
                if metrics is not None:
                    metrics["hypoxia"] = label
                    records.append(metrics)

        return pd.DataFrame(records)

    def build_train_dataset(self, output_path: str = "train_dataset.csv") -> pd.DataFrame:
        """
        Собирает датасет из обеих директорий, добавляет метку и перемешивает.
        
        :param output_path: путь для сохранения CSV
        :return: итоговый DataFrame
        """
        print("Обработка данных из regular...")
        df_regular = self._process_directory(self.regular_dir, label=0)

        print("Обработка данных из hypoxia...")
        df_hypoxia = self._process_directory(self.hypoxia_dir, label=1)

        # Объединяем
        combined = pd.concat([df_regular, df_hypoxia], ignore_index=True)

        if combined.empty:
            raise ValueError("Ни один валидный файл не найден в обеих директориях!")

        # Перемешиваем
        combined_shuffled = shuffle(combined, random_state=42).reset_index(drop=True)

        # Сохраняем
        combined_shuffled.to_csv(output_path, index=False)
        print(f" Датасет сохранён в {output_path} ({len(combined_shuffled)} записей)")
        return combined_shuffled