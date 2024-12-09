import numpy as np
import pandas as pd
import re
import gc

# time-related transformations

# determine if bg gap in a row (given the data should be usually either 5 or 15)
def get_avg_gap_in_min(row):
    positions = row.index.get_indexer(row.dropna().index)
    if len(positions) <= 1:
        avg_gap_in_min = -1
    else:
        avg_gap_in_min = np.diff(positions).mean() * 5
    return avg_gap_in_min

# determine the bg sampling gap for each patient (either 5 or 15 in this data)
def get_bg_gap_series(df):
    bg = df.filter(regex="bg-.*")
    p_num = df.p_num
    avg_gap_in_min = bg.apply(lambda a: get_avg_gap_in_min(a), axis=1)
    avg_gap_in_min.name = 'bg_gap' 
    df_gap = pd.concat([p_num, avg_gap_in_min], axis=1)
    bg_gap = df_gap.groupby('p_num').transform(lambda x: x.mode()[0]).astype(int).squeeze()
    return bg_gap

# create trigonomitric features from time
def get_time_in_minutes_trig(series):
    split_time = series.str.split(":", expand=True).astype(int)
    time_in_minutes = split_time.iloc[:, 0] * 60 + split_time.iloc[:, 1]
    time_in_minutes_sin = np.sin(2 * np.pi * time_in_minutes / 1440)
    time_in_minutes_cos = np.cos(2 * np.pi * time_in_minutes / 1440)
    time_trig_df = pd.concat([time_in_minutes_sin, time_in_minutes_cos], axis=1)
    return time_trig_df

# transform column name's time to minutes for a series
def time_str_to_minutes(series):
    split_time = series.str.split(":", expand=True).astype(int)
    time_in_minutes = split_time.iloc[:, 0] * 60 + split_time.iloc[:, 1]
    return time_in_minutes

# inverse transformation
def time_minutes_to_str(series):
    hours = (series/60).astype(int).astype(str).str.pad(width=2, side='left', fillchar='0')
    minutes = series.mod(60).astype(int).astype(str).str.pad(width=2, side='left', fillchar='0')
    time_as_string = hours + ":" + minutes + ":" + "00"
    return time_as_string


# transform column name's time to minutes for a single value
def time_str_to_minutes_value(x):
    time_in_minutes = int(re.search('\\w+-(\\d{1}):(\\d{2})', x).group(1)) * 60 + int(re.search('\\w+-(\\d{1}):(\\d{2})', x).group(2))
    return time_in_minutes

# determine an aproximate relative day of chronologically ordered data, per patient.
def day_determiner(df):

    split_time = df.time.str.split(":", expand=True).astype(int)
    time_in_seconds_series = split_time.iloc[:, 0] * 60 + split_time.iloc[:, 1]

    patient_time_df = pd.DataFrame(index=df.index)
    patient_time_df['p_num'] = df.p_num
    patient_time_df['time_in_seconds'] = time_in_seconds_series
    
    current_patient = patient_time_df.iloc[0].p_num
    current_time_in_seconds = patient_time_df.iloc[0].time_in_seconds
    day = 0

    days = []

    for index, row in patient_time_df.iterrows():
        new_patient = row.p_num
        new_time_in_seconds = row.time_in_seconds
        
        if new_patient == current_patient:
            if new_time_in_seconds <= current_time_in_seconds:
                day = day + 1
        else:
            day = 1

        days.append(day)
        current_patient = new_patient
        current_time_in_seconds = new_time_in_seconds

    days_series = pd.Series(data=days, index=df.index)

    return days_series

# estimate an aproximate relative day (pseudoday) of non-chronologically ordered data, per patient.
def pseudoday_determiner(df):

    current_patient = df.iloc[0].p_num
    current_id = df.iloc[0].id
    pseudoday = -1

    pseudodays = []

    for index, row in df.iterrows():
        new_patient = row.p_num
        new_id = row.id

        if new_patient == current_patient:
            if current_id != new_id:
                pseudoday = pseudoday - 1
        else:
            pseudoday = -1

        pseudodays.append(pseudoday)
        current_patient = new_patient
        current_id = new_id

    pseudodays_series = pd.Series(data=pseudodays, index=df.index)

    return pseudodays_series


# lag features transformations


# fill mnissing values with 0, until the maximum tolerance is reached
def fillna_horizontal_with_tolerance(df, tolerance=1):
    ffilled_df = df.ffill(limit=tolerance, axis=1).fillna(-1)
    bfilled_df = df.bfill(limit=tolerance, axis=1).fillna(-1)
    filled_df = df.fillna(-1)
    df = df.where((filled_df == ffilled_df) & (filled_df == bfilled_df), 0)
    return df


# preprocess blood glucose
# forward linear interpolation is used to fill missing values
def blood_glucose_interpolation_stats(df, gap, n_prior, window=None):
    X = pd.DataFrame(index=df.index)
    df = df.interpolate(axis=1, limit=3, limit_direction='forward')


    lags = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + '_lag')
    lags = lags.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior + 1]
    lags_clean = lags.replace([np.inf, -np.inf, np.nan], -1)
    X = pd.concat([X if not X.empty else None, lags_clean], axis=1)

    if gap == 1:
        
        diff = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + '_diff')
        diff = diff.diff(1, axis=1)  
        diff = diff.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
        X = pd.concat([X if not X.empty else None, diff], axis=1)

    #     acc = diff.rename(columns = lambda x: x.replace('diff', 'acc'))
    #     acc = acc.diff(-1, axis=1)
    #     X = pd.concat([X if not X.empty else None, acc], axis=1)

    #     change_percent = pd.DataFrame(
    #         data=diff.to_numpy() / lags.iloc[:,1:].to_numpy() * 100,
    #         columns=[x.replace('diff', 'change_percent') for x in diff.columns],
    #         index=diff.index
    #     ) 
    #     X = pd.concat([X if not X.empty else None, change_percent], axis=1)

    #     window_diff = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + '_to_' + str(time_str_to_minutes_value(x) + window*5) + '_diff')
    #     window_diff = window_diff.diff(window, axis=1)
    #     window_diff = window_diff.iloc[:, ::-1].iloc[:,::window].iloc[:,:int(n_prior/window)]
    #     X = pd.concat([X if not X.empty else None, window_diff], axis=1)

    #     window_mean = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + '_to_' + str(time_str_to_minutes_value(x) + window*5) + '_mean')
    #     window_mean = window_mean.T.rolling(window=window + 1).mean().T
    #     window_mean = window_mean.iloc[:, ::-1].iloc[:,::window].iloc[:,:int(n_prior/window)]
    #     X = pd.concat([X if not X.empty else None, window_mean], axis=1)

    #     std_mean = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + '_to_' + str(time_str_to_minutes_value(x) + window*5) + '_std')
    #     std_mean = std_mean.T.rolling(window=window + 1).std().T
    #     std_mean = std_mean.iloc[:, ::-1].iloc[:,::window].iloc[:,:int(n_prior/window)]
    #     X = pd.concat([X if not X.empty else None, std_mean], axis=1)
        
    elif gap > 1:

        ra = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + f'_ra_{gap}')
        ra = ra.T.rolling(window=gap * 1).mean().T
        ra = ra.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
        
    #     gf = df.rename(columns = lambda x: 'bg_' + str(time_str_to_minutes_value(x)) + '_gf')
    #     gf = pd.DataFrame(data=gaussian_filter1d(gf, sigma=1), columns=gf.columns, index=gf.index)
    #     gf = gf.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior + 1]
    
        X = pd.concat([X if not X.empty else None, ra], axis=1)
    #     X = pd.concat([X if not X.empty else None, gf], axis=1)
    
    X.replace([np.inf, -np.inf, np.nan], -99, inplace=True)
    return X


# preprocess insuline
# fill NaN with tolerance of 3 is used
def insulin_interpolation_stats(df, gap, n_prior):    
    X = pd.DataFrame(index=df.index)
    df = fillna_horizontal_with_tolerance(df, tolerance=3)

    if gap == 1:

        lags = df.rename(columns = lambda x: 'insulin_' + str(time_str_to_minutes_value(x)) + '_lag')
        lags = lags.iloc[:, ::-1].iloc[:,:n_prior]
        lags_clean = lags.replace([np.inf, -np.inf, np.nan], -1)
        X = pd.concat([X if not X.empty else None, lags_clean], axis=1)

        # diff = df.rename(columns = lambda x: 'insulin_' + str(time_str_to_minutes_value(x)) + '_diff')
        # diff = diff.diff(1, axis=1)  
        # diff = diff.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
        # X = pd.concat([X if not X.empty else None, diff], axis=1)

    elif gap > 1:

        summation = df.rename(columns = lambda x: 'insulin_' + str(time_str_to_minutes_value(x)) + f'_sum_{gap}')
        summation = summation.T.rolling(window=gap * 1).sum().T
        summation = summation.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
    
        med = df.rename(columns = lambda x: 'insulin_' + str(time_str_to_minutes_value(x)) + f'_median_{gap}')
        med = med.T.rolling(window=gap * 1).median().T
        med = med.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]

        X = pd.concat([X if not X.empty else None, summation], axis=1)
        X = pd.concat([X if not X.empty else None, med], axis=1)
        
    X.replace([np.inf, -np.inf, np.nan], -999, inplace=True)
    return X


# preprocess insuline
# fill NaN with 0 is used
def carbs_interpolation_stats(df, gap, n_prior):
    X = pd.DataFrame(index=df.index)
    df = df.fillna(0)

    if gap == 1:

        lags = df.rename(columns = lambda x: 'carbs_' + str(time_str_to_minutes_value(x)) + '_lag')
        lags = lags.iloc[:, ::-1].iloc[:,:n_prior]
    
        X = pd.concat([X if not X.empty else None, lags], axis=1)

    elif gap > 1:

        summation = df.rename(columns = lambda x: 'carbs_' + str(time_str_to_minutes_value(x)) + f'_sum_{gap}')
        summation = summation.T.rolling(window=gap * 1).sum().T
        summation = summation.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]

        X = pd.concat([X if not X.empty else None, summation], axis=1)
        
    X.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    return X


# preprocess steps
# fill NaN with tolerance of 3 is used
def steps_interpolation_stats(df, gap, n_prior):
    X = pd.DataFrame(index=df.index)
    df = fillna_horizontal_with_tolerance(df, tolerance=3)

    if gap == 1:

        lags = df.rename(columns = lambda x: 'steps_' + str(time_str_to_minutes_value(x)) + '_lag')
        lags = lags.iloc[:, ::-1].iloc[:,:n_prior]

        X = pd.concat([X if not X.empty else None, lags], axis=1)

    elif gap > 1:

        # summation = df.rename(columns = lambda x: 'steps_' + str(time_str_to_minutes_value(x)) + f'_sum_{gap}')
        # summation = summation.T.rolling(window=gap * 1).sum().T
        # summation = summation.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
    
        lags = df.rename(columns = lambda x: 'steps_' + str(time_str_to_minutes_value(x)) + f'_lag_{gap}')
        lags = lags.T.rolling(window=gap * 1).mean().T
        lags = lags.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
    
        # X = pd.concat([X if not X.empty else None, lags], axis=1)
        # X = pd.concat([X if not X.empty else None, summation], axis=1)
        X = pd.concat([X if not X.empty else None, lags], axis=1)
    
    X.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    return X
    

# preprocess heart rate
# bidirectional linear interpolation with a limit of 3 is used
def heart_rate_interpolation_stats(df, gap, n_prior):
    X = pd.DataFrame(index=df.index)
    df = df.interpolate(axis=1, limit=3, limit_direction='both')

    if gap == 1:
        lags = df.rename(columns = lambda x: 'hr_' + str(time_str_to_minutes_value(x)) + '_lag')
        lags = lags.iloc[:, ::-1].iloc[:,:n_prior]
    
        X = pd.concat([X if not X.empty else None, lags], axis=1)

    elif gap > 1:
        ra = df.rename(columns = lambda x: 'hr_' + str(time_str_to_minutes_value(x)) + f'_ra_{gap}')
        ra = ra.T.rolling(window=gap * 1).mean().T
        ra = ra.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
    
        X = pd.concat([X if not X.empty else None, ra], axis=1)
        
    X.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    return X


# preprocess calories
# bidirectional linear interpolation with a limit of 3 is used
def calories_interpolation_stats(df, gap, n_prior):
    X = pd.DataFrame(index=df.index)
    df = df.interpolate(axis=1, limit=3, limit_direction='both')

    if gap == 1:
        lags = df.rename(columns = lambda x: 'cals_' + str(time_str_to_minutes_value(x)) + '_lag')
        lags = lags.iloc[:, ::-1].iloc[:,:n_prior]
    
        X = pd.concat([X if not X.empty else None, lags], axis=1)

    elif gap > 1:
        summation = df.rename(columns = lambda x: 'cals_' + str(time_str_to_minutes_value(x)) + f'_sum_{gap}')
        summation = summation.T.rolling(window=gap * 1).sum().T
        summation = summation.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
    
        X = pd.concat([X if not X.empty else None, summation], axis=1)
        
    X.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    return X


# preprocess activity
# the description itself is not used, since all all non missing values are replaced by 1 and missing values by 0
def activity_interpolation_stats(df, gap, n_prior):
    X = pd.DataFrame(index=df.index)
    df = df.mask(df.notna().to_numpy(), 1).astype('Int64')
    df = df.fillna(0)

    if gap == 1:
        lags = df.rename(columns = lambda x: 'activity_' + str(time_str_to_minutes_value(x)) + '_lag')
        lags = lags.iloc[:, ::-1].iloc[:,:n_prior]
        X = pd.concat([X if not X.empty else None, lags], axis=1)

    elif gap > 1:
        summation = df.rename(columns = lambda x: 'activity_' + str(time_str_to_minutes_value(x)) + f'_sum_{gap}')
        summation = summation.T.rolling(window=gap * 1).sum().T
        summation = summation.iloc[:, ::-1].iloc[:,::gap].iloc[:,:n_prior]
    
        X = pd.concat([X if not X.empty else None, summation], axis=1)

    X.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    return X




def data_expander(df, gap, n_prior, addition, data_source):
    # Expand dataset according to window defined by gap * n_prior. Minimun gap in data is 5 minutes
    # For example, I used a 1 hour window of 5 minute measures (gap = 1, n_prior = 12). 
    # In case I needed a couple of additional points, I could add some with addition, reducing the number of expansions as a consequence
    # For example, I ended up choosing addition = 0
    # With gap = 1, n_prior = 12 and addition = 0, I would be able to iterate over the 6-hour data 6 * 12 - (1 * 12 + 0) = 60 times
    # Substracting the forecast horizon (1 hour = 12 gaps), I would get 48 expansion windows plus the initial window 0

    bg = df.filter(regex="bg-.*")
    ins = df.filter(regex="insulin-.*")
    carbs =df.filter(regex="carbs-.*")
    hr = df.filter(regex="hr.*")
    steps = df.filter(regex="steps-.*")
    cals = df.filter(regex="cals.*")
    activity = df.filter(regex="activity.*")
    time_in_minutes = time_str_to_minutes(df['time'])

    bg_cols = bg.columns.to_list()
    ins_cols = ins.columns.to_list()
    carbs_cols = carbs.columns.to_list()
    hr_cols = hr.columns.to_list()
    steps_cols = steps.columns.to_list()
    activity_cols = activity.columns.to_list()

    bg_cols = bg_cols[-(gap * n_prior + addition):]
    ins_cols = ins_cols[-(gap * n_prior + addition):]
    carbs_cols = carbs_cols[-(gap * n_prior + addition):]
    hr_cols = hr_cols[-(gap * n_prior + addition):]
    steps_cols = steps_cols[-(gap * n_prior + addition):]
    activity_cols = activity_cols[-(gap * n_prior + addition):]

    cols_for_drop_duplicates = ['p_num', 'time'] + bg_cols + ins_cols + carbs_cols + hr_cols + steps_cols + activity_cols

    if data_source == 'train':
        base_X = df.drop('bg+1:00', axis=1).copy()
    elif data_source == 'train_from_test':
        base_X = df.copy()
    base_X.reset_index(drop=False, inplace=True)
    base_X['iter'] = 0

    expansions = len(bg.columns.to_list()) - 12 - (gap * n_prior + addition)

    for i in range(expansions):
        row_id = df.id
        p_num = df.p_num
        
        time_in_minutes = time_in_minutes - 5
        time_in_minutes = time_in_minutes.replace(-5, 1440 - 5)
        time_as_string = time_minutes_to_str(time_in_minutes)
        time_as_string.name = 'time'

        bg = bg.shift(axis=1)
        ins = ins.shift(axis=1)
        carbs = carbs.shift(axis=1)
        hr = hr.shift(axis=1)
        steps = steps.shift(axis=1)
        cals = cals.shift(axis=1)
        activity = activity.shift(axis=1)

        shifted_base_X = pd.concat([row_id, p_num, time_as_string, bg, ins, carbs, hr, steps, cals, activity], axis=1)
        shifted_base_X.reset_index(drop=False, inplace=True)
        shifted_base_X['iter'] = i + 1
        base_X = pd.concat([shifted_base_X, base_X], ignore_index=True)            
        base_X.drop_duplicates(subset=cols_for_drop_duplicates, inplace=True, keep='last')
    
    base_X.sort_values(by=['index', 'iter', 'id'], ascending=[True, False, True], inplace=True)
    base_X.drop(['index'], axis=1, inplace=True)
    base_X = base_X[base_X['bg-0:00'].notna()]

    row_id = base_X.id
    p_num = base_X.p_num
    time = base_X.time
    iteration = base_X.iter

    bg = base_X.filter(regex="bg-.*")
    ins = base_X.filter(regex="insulin-.*")
    carbs =base_X.filter(regex="carbs-.*")
    hr = base_X.filter(regex="hr.*")
    steps = base_X.filter(regex="steps-.*")
    cals = base_X.filter(regex="cals.*")
    activity = base_X.filter(regex="activity.*")

    target = bg['bg-0:00']
    target.name = 'bg+1:00'

    bg = bg.shift(periods=12, axis=1)
    ins = ins.shift(periods=12, axis=1)
    carbs = carbs.shift(periods=12, axis=1)
    hr = hr.shift(periods=12, axis=1)
    steps = steps.shift(periods=12, axis=1)
    cals = cals.shift(periods=12, axis=1)
    activity = activity.shift(periods=12, axis=1)

    X = pd.concat([iteration, row_id, p_num, time, bg, ins, carbs, hr, steps, cals, activity, target], axis=1)
    X.reset_index(drop=True, inplace=True)

    time_in_minutes = time_str_to_minutes(X['time'])
    time_in_minutes = time_in_minutes - 60 + 1440
    time_in_minutes = time_in_minutes.mod(1440)
    time_as_string = time_minutes_to_str(time_in_minutes)
    X.time = time_as_string

    X.drop_duplicates(subset=cols_for_drop_duplicates, inplace=True)
    X.reset_index(drop=True, inplace=True)
    
    if data_source == 'train':
        X = pd.concat([X, df], ignore_index=True)
        X.drop_duplicates(subset=cols_for_drop_duplicates, inplace=True, keep='last')
        X.drop('iter', axis=1, inplace=True)
        X.reset_index(drop=True, inplace=True)
        gc.collect()
        return base_X, X
    elif data_source == 'train_from_test':
        step_list = expansions - X.iter
        X.drop('iter', axis=1, inplace=True)
        gc.collect()
        return base_X, X, step_list
    

def feature_transformer(df, gap, n_prior, data_source):
    # Applies transformers sequentially to the data.
    # Depends on the source (train, test, train from test after expansion)
    X = pd.DataFrame(index=df.index)

    bg = df.filter(regex="bg-.*")
    ins = df.filter(regex="insulin-.*")
    carbs = df.filter(regex="carbs-.*")
    steps = df.filter(regex="steps-.*")
    hr = df.filter(regex="hr.*")
    cals = df.filter(regex="cals.*")
    activity = df.filter(regex="activity.*")
    timestamp = df['time']
    
    X['p_num'] = df.p_num
    X['bg_gap'] = get_bg_gap_series(df)
    X[['time_in_minutes_sin', 'time_in_minutes_cos']] = timestamp.apply(get_time_in_minutes_trig, by_row=False)
    X = pd.concat([X if not X.empty else None, blood_glucose_interpolation_stats(bg, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)
    X = pd.concat([X if not X.empty else None, insulin_interpolation_stats(ins, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)
    X = pd.concat([X if not X.empty else None, carbs_interpolation_stats(carbs, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)
    X = pd.concat([X if not X.empty else None, steps_interpolation_stats(steps, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)
    X = pd.concat([X if not X.empty else None, heart_rate_interpolation_stats(hr, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)
    X = pd.concat([X if not X.empty else None, calories_interpolation_stats(cals, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)
    X = pd.concat([X if not X.empty else None, activity_interpolation_stats(activity, gap=gap, n_prior=n_prior)], axis=1, ignore_index=False)

    if data_source == 'train':
        y = df['bg+1:00']
        patient_groups = df.p_num
        day_groups = day_determiner(df)
        return X, y, patient_groups, day_groups
    
    elif data_source == 'test':
        return X
    
    elif data_source == 'train_from_test':
        y = df['bg+1:00']
        patient_groups = df.p_num
        pseudoday_groups = pseudoday_determiner(df)        
        return X, y, patient_groups, pseudoday_groups


