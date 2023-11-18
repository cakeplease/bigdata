import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error  

# a)
# Import data
df = pd.read_csv('satisfaction_2015.csv')

# Find mean value
flight_dist_mean = df['Flight Distance'].mean()

print(f"The general mean flight distance value is {flight_dist_mean}.")

# Filter data
women_under_30 = df[(df['Gender'] == 'Female') & (df['Age'] < 30)]

# Find mean value of filtered data
flight_dist_mean = women_under_30['Flight Distance'].mean()

print(f"Mean flight distance value of women under 30 years old is {flight_dist_mean}")


# b)
# Import data
df = pd.read_csv('satisfaction_2015.csv')

# Filter data
delayed_flights = df[df['Departure Delay in Minutes'] >= 10]

# Plot data
delayed_flights['Departure Delay in Minutes'].value_counts().sort_index().plot()

# Customize and show plot
plt.title('Number of flights with departure delay of 10 minutes or more')
plt.xlabel('Departure delay (minutes)')
plt.ylabel('Number of occurrences')
plt.show()

# c)
# Import data
df = pd.read_csv('satisfaction_2015.csv')

# Plot data into histogram
plt.hist(df['Age'], bins=10, rwidth=0.5)

# Customize plot
plt.title('Distribution of the age of the passengers')
plt.xlabel('Age')
plt.ylabel('Number of passangers')
plt.show()

# d)
# Import data
df = pd.read_csv('satisfaction_2015.csv')

# Wash data
df.replace({
    'Gender': {'Female': 0, 'Male': 1},
    'satisfaction_v2': {'neutral or dissatisfied': 0, 'satisfied': 1},
    'Customer Type': {'disloyal Customer': 0, 'Loyal Customer': 1}
}, inplace=True)

X_train = df[['Gender', 'satisfaction_v2', 'Customer Type', 'Seat comfort', 'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment', 'Ease of Online booking']]
y_test = df['Age']

# Transform data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# 'gender' = 1
# 'Customer Type' = '1'
# 'satisfaction_v2' = '1'
# 'Seat comfort' = 4
# 'Food and drink' = 5
# 'Gate location' = 5
# 'Inflight wifi service' = 5
# 'Inflight entertainment' = 5
# 'Ease of Online booking'= 3

X_test_scaled = scaler.transform([[1,1,1,4,5,5,5,5,3]])

# Train and predict
model = KNeighborsRegressor(n_neighbors=6)
model.fit(X_train_scaled, y_test)
age = model.predict(X_test_scaled)[0]
print(f"The passenger is {round(age)} ({round(age, 4)}) years old.")


# e)
# Import data
df = pd.read_csv('satisfaction_2015.csv')

# Wash data
df.replace({
    'Gender': {'Female': 0, 'Male': 1},
    'Type of Travel': {'Personal Travel': 0, 'Business travel': 1},
    'Class': {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
}, inplace=True)

# Specify X and y
X = df[['Gender', 'Age', 'Type of Travel', 'Flight Distance']]
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Grow tree
classifier = DecisionTreeClassifier(min_impurity_decrease=0.01, max_depth=5)  
classifier.fit(X_train, y_train) 

# Test tree
y_pred = classifier.predict(X_test) 
print(classification_report(y_test, y_pred))

# Display tree
dotfile = open("./classification_tree.dot", 'w')
ditfile = tree.export_graphviz(classifier, out_file = dotfile, feature_names = X.columns, class_names=['Eco', 'Eco plus', 'Business'])
dotfile.close()

# Use http://webgraphviz.com/ or see classification_tree.png to show the tree.


# Question 1:
# How well does the tree predict, and how will deepening the tree affect the prediction error?
# Answer: We can see that our tree gets around 0.75 accuracy which is a ratio between correct predictions and number of predictions in total.
# In addition we can see that the precision of the Eco class is aournd 82% correct, and the Business class 66% correct. Even though the prediction 
# of Eco Plus class was not possible with current depth of the tree, the model accuracy is at around 71%.

# Deepening the tree with changing the "min_impurity_decrease" parameter to 0 would result in 100% precision for predicting Eco Plus, and around 75% prediciton for the two other classes with accuracy around 75%.
# Even though this might seem as better idea, the whole point of decision tree would not make sense longer as we have to create a very deep tree to get these results,
# and that would be more difficult to visualize and understand.

# Question 2:
# You may find that the class «Eco Plus» seems to be missing unless the tree becomes very deep. Can 
# you think of why this is?
# Answer: The reason can be that it is difficult to predict. We can see there are two options for type of travel, and 
# three options for class. It seems when we take a look at the data that there is a correlation between personal travel and eco class,
# and business travel and business class. But Eco plus seem to be firstly not as popular(significantly fewer entries) as the other two options and secondly, less consistent. 


# f)
# Import data
df = pd.read_csv('satisfaction_2015.csv')

X = df[['Age']]
y = df['Ease of Online booking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

#Evaluate the performance of the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
intercept = lr.intercept_

print("R2 score: ", r2)
#print("Mean Squared Error: ", mse)
#print(f"R2 score {round(r2, 6)} means that {round(r2, 6)*100}% of the rating score can be explained by age.")

# The low R2 indicates that the linear relationship between the age of customers and the score they've given for "Ease of Online booking" is very poor.
# Our model predicts the outcome poorly. We should consider other types of analysis, for example either checking if other variable than age give us better results with linear regression, 
# or if taking multiple variables to consideration give us better picture. 

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.title('Age vs Ease of online booking score')
plt.xlabel('Age')
plt.ylabel('Ease of online booking score')
plt.show()

