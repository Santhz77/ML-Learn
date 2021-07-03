import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

train_data = pd.read_csv("data_src/train.csv")
# x = train_data.head(25)
#
test_data = pd.read_csv("data_src/test.csv")
y = test_data.head()


women = train_data.loc[train_data.Sex == 'female']["Survived"]

print(train_data.loc[train_data.Sex == 'female']["Survived"])

print(sum(women))

print(len(women))



#
# rate_women = sum(women)/len(women)
#
# print("% of women who survived:", rate_women)

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
