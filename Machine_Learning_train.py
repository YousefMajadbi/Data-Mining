import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
insulin_data_1 = pd.read_csv('InsulinData.csv')
insulin_data_2 = pd.read_csv('Insulin_patient2.csv')
insulin_data = pd.concat([insulin_data_1, insulin_data_2], ignore_index=True)

cgm_data_1 = pd.read_csv('CGMData.csv')
cgm_data_2 = pd.read_csv('CGM_patient2.csv')
cgm_data = pd.concat([cgm_data_1, cgm_data_2], ignore_index=True)

# Extracting meal start times from InsulinData (non-zero, non-NaN 'BWZ Carb Input (grams)')
meal_times = insulin_data[pd.notnull(insulin_data['BWZ Carb Input (grams)']) & (insulin_data['BWZ Carb Input (grams)'] > 0)]

meal_times['datetime'] = pd.to_datetime(meal_times['Date'] + ' ' + meal_times['Time'])

# Extract glucose data corresponding to meal time (tm)
cgm_data['datetime'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])

cgm_data = cgm_data.interpolate(method='backfill',limit_direction = 'backward')

def extract_meal_data(meal_times, cgm_data):
    meal_data_list = []
    
    for tm in meal_times['datetime']:
        
        tm_minus_30min = tm - pd.Timedelta(minutes=30)
        tm_plus_2hrs = tm + pd.Timedelta(hours=2)
        
        meal_data = cgm_data[(cgm_data['datetime'] >= tm_minus_30min) & (cgm_data['datetime'] <= tm_plus_2hrs)]
        
        future_meal = meal_times[(meal_times['datetime'] > tm) & (meal_times['datetime'] < tm_plus_2hrs)]
        
        if not future_meal.empty:
            # Another meal exists at tp, replace tm with tp
            tp = future_meal['datetime'].iloc[0]
            meal_data = cgm_data[(cgm_data['datetime'] >= tp - pd.Timedelta(minutes=30)) & (cgm_data['datetime'] <= tp + pd.Timedelta(hours=2))]
        
        elif not meal_times[meal_times['datetime'] == tm_plus_2hrs].empty:
            meal_data = cgm_data[(cgm_data['datetime'] >= tm + pd.Timedelta(hours=1, minutes=30)) & (cgm_data['datetime'] <= tm + pd.Timedelta(hours=4))]
        
        if len(meal_data) == 30:  # Ensure exactly 30 readings (every 5 mins)
            meal_data_list.append(meal_data['Sensor Glucose (mg/dL)'].values)

    meal_data_matrix = np.array(meal_data_list)
    return meal_data_matrix

meal_data_matrix = extract_meal_data(meal_times, cgm_data)

def extract_no_meal_data(meal_times, cgm_data):
    no_meal_data_list = []
    
    for tm in meal_times['datetime']:
        tm_plus_2hrs = tm + pd.Timedelta(hours=2)
        
        future_meal = meal_times[(meal_times['datetime'] > tm_plus_2hrs) & (meal_times['datetime'] <= tm_plus_2hrs + pd.Timedelta(hours=2))]
        
        if future_meal.empty:  # No meal found in the post-absorptive period, so we can consider this period for no meal data
            no_meal_data = cgm_data[(cgm_data['datetime'] >= tm_plus_2hrs) & (cgm_data['datetime'] <= tm_plus_2hrs + pd.Timedelta(hours=2))]
            
            if len(no_meal_data) == 24:  # Ensure exactly 24 readings (every 5 mins over 2 hours)
                no_meal_data_list.append(no_meal_data['Sensor Glucose (mg/dL)'].values)
        else:
            continue
    
    no_meal_data_matrix = np.array(no_meal_data_list)
    return no_meal_data_matrix

no_meal_data_matrix = extract_no_meal_data(meal_times, cgm_data)

def extract_features(data_matrix):
    features = np.apply_along_axis(lambda row: [
        #np.argmax(row) * 5,
        np.mean(row),
        np.var(row),
        #(np.max(row) - row[0]) / np.std(row) if np.std(row) != 0 else 0,
        *np.abs(np.fft.fft(row)[:4]),
        np.mean(np.abs(np.diff(row,n=1))), 
        np.mean(np.abs(np.diff(np.diff(row, n=1), n=1)))
    ], axis=1, arr=data_matrix)
    return features

meal_features = extract_features(meal_data_matrix)

################main()######
# Meal feature matrix
meal_features = extract_features(meal_data_matrix)
# No meal feature matrix
no_meal_features = extract_features(no_meal_data_matrix)

# Step 1: Concatenate meal and no meal feature matrices
X = np.vstack([meal_features, no_meal_features])

# Step 2: Create label vector (1 for meal, 0 for no meal)
y = np.hstack([np.ones(meal_features.shape[0]), np.zeros(no_meal_features.shape[0])])

# Step 3: Train-test split


#model = SVC(kernel='rbf', C=1, gamma=1, class_weight='balanced')

model = DecisionTreeClassifier(
    criterion='gini',       
    max_depth=None,         
    min_samples_leaf=1,     
    min_samples_split=4,    
    class_weight='balanced',
    random_state=40         
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
for train, test in kfold.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
        
    # Step 4: Train SVM machine learning model
    model.fit(X_train, y_train)


# Step 5: Evaluate using k-fold cross-validation
#cv_scores = cross_val_score(model, X_train, y_train, cv=5)
#print(f"Cross-validation accuracy: {cv_scores.mean()}")

# Step 6: Test the model on the test data
#y_pred = model.predict(X_test)

with open('meal_no_meal_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
