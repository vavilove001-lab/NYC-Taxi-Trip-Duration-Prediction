import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error

# --- ГЕОМЕТРИЯ И РАССТОЯНИЯ ---

def add_osrm_features(taxi_df: pd.DataFrame, osrm_df: pd.DataFrame) -> pd.DataFrame:
    cols_to_merge = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
    df = pd.merge(
        taxi_df,
        osrm_df[cols_to_merge],
        on='id',
        how='left'
    )
    return df

def get_haversine_distance(lat1, lng1, lat2, lng2):
    """Считает расстояние в км по формуле гаверсинуса."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    EARTH_RADIUS = 6371
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    """Считает угол направления движения в градусах."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет расстояние и угол."""
    df = df.copy()
    cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    if all(col in df.columns for col in cols):
        df['haversine_distance'] = get_haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        df['direction'] = get_angle_direction(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
    return df

# --- ВРЕМЕННЫЕ ПРИЗНАКИ ---

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет базовые временные признаки."""
    df = df.copy()
    if 'pickup_datetime' in df.columns:
        df['pickup_date'] = df['pickup_datetime'].dt.date
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df

def add_temporal_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет синусы/косинусы времени и флаги (час пик, выходные)."""
    df = df.copy()
    
    if 'pickup_hour' in df.columns:
        df['is_rush_hour'] = (
            ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) |
            ((df['pickup_hour'] >= 16) & (df['pickup_hour'] <= 19))
        ).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    
    if 'pickup_day_of_week' in df.columns:
        df['is_weekend'] = (df['pickup_day_of_week'] >= 5).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['pickup_day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['pickup_day_of_week'] / 7)
        
    return df

# --- КЛАСТЕРИЗАЦИЯ ---

def add_cluster_features(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Добавляет кластеры на основе координат."""
    df = df.copy()
    coords = np.hstack((
        df[['pickup_latitude', 'pickup_longitude']],
        df[['dropoff_latitude', 'dropoff_longitude']]
    ))
    df['geo_cluster'] = kmeans.predict(coords)
    return df

# --- ПРАЗДНИКИ ---

def add_holiday_features(df: pd.DataFrame, holiday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признак is_holiday.
    holiday_df должен содержать колонку 'date' (тип datetime) или 'date' как строку.
    """
    df = df.copy()
    
    # Нормализуем даты в справочниках
    if 'date' in holiday_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(holiday_df['date']):
            holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%Y-%m-%d')
    
    # Создаем множество дат-праздников для быстрого поиска
    holiday_dates = set(holiday_df['date'].dt.date)
    
    if 'pickup_date' in df.columns:
        df['is_holiday'] = df['pickup_date'].isin(holiday_dates).astype(int)
    elif 'pickup_datetime' in df.columns:
        # Если pickup_date еще не создан, берем дату из datetime
        df['is_holiday'] = df['pickup_datetime'].dt.date.isin(holiday_dates).astype(int)
    else:
        # Если нет ни того, ни другого, создаем временно
        if 'pickup_datetime' in df.columns:
            df['is_holiday'] = df['pickup_datetime'].dt.date.isin(holiday_dates).astype(int)
            
    return df



def fill_null_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в погодных данных медианой по колонке."""
    df = df.copy()
    
    # Погодные числовые столбцы — заполняем медианой по дате
    weather_numeric_cols = ['temperature', 'visibility', 'wind speed', 'precip']
    for col in weather_numeric_cols:
        df[col] = df[col].fillna(
            df.groupby('pickup_date')[col].transform('median')
        )
    
    # Погодные явления — заполняем 'None'
    df['events'] = df['events'].fillna('None')
    
    # OSRM-столбцы — заполняем медианой по столбцу
    osrm_cols = ['total_distance', 'total_travel_time', 'number_of_steps']
    for col in osrm_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def add_weather_features(taxi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    # Оставляем только нужные столбцы из weather_data
    weather_cols = ['time', 'temperature', 'visibility', 'wind speed', 'precip', 'events']
    weather = weather_df[weather_cols].copy()
    
    # Создаём столбцы date и hour для объединения
    weather['time'] = pd.to_datetime(weather['time'])
    weather['date'] = weather['time'].dt.date
    weather['hour'] = weather['time'].dt.hour
    
    # Удаляем исходный столбец time — он больше не нужен
    weather = weather.drop(columns=['time'])
    
    # Объединяем по дате и часу (left join — сохраняем все поездки)
    df = pd.merge(
        taxi_df,
        weather,
        left_on=['pickup_date', 'pickup_hour'],
        right_on=['date', 'hour'],
        how='left'
    )
    
    # Удаляем вспомогательные столбцы
    df = df.drop(columns=['date', 'hour'])
    
    return df

def remove_outliers(
    df: pd.DataFrame,
    speed_col: str = 'avg_speed',
    duration_col: str = 'trip_duration',
    threshold_seconds: int = 24 * 3600,
    threshold_speed: float = 300,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Удаляет выбросы по длительности поездки и средней скорости.
    
    Параметры:
        df: исходный DataFrame
        speed_col: название колонки со скоростью (км/ч)
        duration_col: название колонки с длительностью (секунды)
        threshold_seconds: максимальная длительность (по умолчанию 24 часа)
        threshold_speed: максимальная скорость (по умолчанию 300 км/ч)
        verbose: печатать статистику
        
    Возвращает:
        очищенный DataFrame
    """
    df = df.copy()
    original_len = len(df)
    
    # Считаем выбросы
    num_outliers_duration = (df[duration_col] > threshold_seconds).sum()
    num_outliers_speed = (df[speed_col] > threshold_speed).sum()
    
    if verbose:
        print(f"Выбросы по длительности (>{threshold_seconds / 3600:.0f} ч): {num_outliers_duration}")
        print(f"Выбросы по скорости (>{threshold_speed} км/ч): {num_outliers_speed}")
    
    # Маска: удаляем, если хотя бы одно условие выполнено
    mask_to_drop = (df[duration_col] > threshold_seconds) | (df[speed_col] > threshold_speed)
    
    df = df[~mask_to_drop].copy()
    
    if verbose:
        removed = original_len - len(df)
        pct = removed / original_len * 100
        print(f"Удалено: {removed:,} строк ({pct:.2f}%). Осталось: {len(df):,}")
    
    return df



def calc_rmsle_from_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    # Не даём прогнозу уйти в минус — минимальная длительность поездки 0 секунд
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))