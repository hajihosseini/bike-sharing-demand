if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    from sklearn import ensemble


    #Load Data with pandas, and parse the first column into datetime
    train = pd.read_csv('train.csv', parse_dates=[0])
    test = pd.read_csv('test.csv', parse_dates=[0])

    #Feature engineering

    temp = pd.DatetimeIndex(train['datetime'])
    train['year'] = temp.year
    train['month'] = temp.month
    train['hour'] = temp.hour
    train['weekday'] = temp.weekday

    temp = pd.DatetimeIndex(test['datetime'])
    test['year'] = temp.year
    test['month'] = temp.month
    test['hour'] = temp.hour
    test['weekday'] = temp.weekday

    for col in ['casual', 'registered', 'count']:
        train['log-' + col] = train[col].apply(lambda x: np.log1p(x))

    features = ['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed', 'year',
             'month', 'weekday', 'hour']



    #Some of the important parameters for the GBM are:
    #    number of trees (n_estimators)
    #    depth of each individual tree (max_depth)
    #    loss function (loss)
    #    learning rate (learning_rate)
    clf = RandomForestRegressor(n_estimators = 1000, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
    clf.fit(train[features], train['log-count'])
    result = clf.predict(test[features])
    result = np.expm1(result)

    df=pd.DataFrame({'datetime':test['datetime'],'season':test['season'],'holiday':test['holiday'],'workingday':test['workingday'],'weather':test['weather'],'temp':test['temp'],'atemp':test['atemp'],'humidity':test['humidity'],'windspeed':test['windspeed'],'count':result})
    df.to_csv('RandomForest .csv', index = False, columns=['datetime','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','count'])



