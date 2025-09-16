import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style='whitegrid')

def generate_synthetic_data(seed=42):
    np.random.seed(seed)
    hours = pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='H')
    cities = ['CityA', 'CityB', 'CityC']
    rows = []
    for city in cities:
        base = {'CityA': 40, 'CityB': 30, 'CityC': 20}[city]
        for dt in hours:
            month = dt.month
            hour = dt.hour
            season_mul = 1.2 if month in [12,1,2] else (0.9 if month in [6,7,8] else 1.0)
            diurnal = 1.15 if hour < 6 or hour > 20 else 0.9
            temperature = 15 + 10*np.sin((month-1)/12*2*np.pi) + np.random.normal(0,2)
            humidity = np.clip(50 + 20*np.cos((hour/24)*2*np.pi) + np.random.normal(0,8), 10, 100)
            wind_speed = np.abs(np.random.normal(2.5, 1.0))
            precipitation = np.random.exponential(0.1)
            industrial_index = np.clip(np.random.normal(1.0 + (0.5 if city=='CityC' else 0), 0.2), 0.2, 3.0)

            pm25 = (base * season_mul * diurnal
                    + 0.6*industrial_index*30
                    - 0.8*precipitation*20
                    + 0.2*(100-humidity)
                    - 0.5*(temperature-15)
                    + np.random.normal(0,8))
            pm25 = max(0.5, pm25)

            rows.append({
                'datetime': dt,
                'city': city,
                'pm25': pm25,
                'temperature': round(temperature,2),
                'humidity': round(humidity,1),
                'wind_speed': round(wind_speed,2),
                'precipitation': round(precipitation,3),
                'industrial_index': round(industrial_index,3)
            })
    df = pd.DataFrame(rows)

    for col in ['temperature','humidity','wind_speed']:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    outlier_mask = np.random.rand(len(df)) < 0.002
    df.loc[outlier_mask, 'pm25'] *= 6

    return df


def save_and_load_demo(df, path='data/synthetic_air_quality.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    df2 = pd.read_csv(path, parse_dates=['datetime'])
    return df2


def basic_cleaning(df):
    df = df.copy()
    if not np.issubdtype(df['datetime'].dtype, np.datetime64):
        df['datetime'] = pd.to_datetime(df['datetime'])

    df = df.sort_values(['city','datetime']).reset_index(drop=True)

    # interpolate per city using datetime as index (method='time' requires DatetimeIndex)
    for city in df['city'].unique():
        mask = df['city'] == city
        sub = df.loc[mask].sort_values('datetime')
        interp = (sub.set_index('datetime')[['temperature','humidity','wind_speed']]
                  .interpolate(method='time')
                  .fillna(method='bfill')
                  .fillna(method='ffill'))
        # write back interpolated values preserving original index order
        df.loc[sub.index, ['temperature','humidity','wind_speed']] = interp.values

    df['humidity'] = df['humidity'].clip(0,100)

    q_low, q_high = df['pm25'].quantile([0.01,0.99])
    df['pm25_winsor'] = df['pm25'].clip(q_low, q_high)

    return df


def feature_engineering(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['pm25_roll24'] = df.groupby('city')['pm25_winsor'].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df['industrial_log'] = np.log1p(df['industrial_index'])
    df = pd.get_dummies(df, columns=['city'], drop_first=True)
    return df


def train_model(df):
    features = ['temperature','humidity','wind_speed','precipitation',
                'industrial_log','pm25_roll24','hour','dayofweek','month',
                'city_CityB','city_CityC']
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['pm25_winsor']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, X_test, y_test, y_pred, mse, r2


def main():
    df = generate_synthetic_data()
    df = save_and_load_demo(df)
    df_clean = basic_cleaning(df)
    df_feat = feature_engineering(df_clean)
    model, X_test, y_test, y_pred, mse, r2 = train_model(df_feat)

    print('MSE:', mse)
    print('R2 :', r2)

    os.makedirs('output', exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('实际 pm25')
    plt.ylabel('预测 pm25')
    plt.title('真实值 vs 预测值')
    plt.tight_layout()
    plt.savefig('output/true_vs_pred.png')

if __name__ == '__main__':
    main()
