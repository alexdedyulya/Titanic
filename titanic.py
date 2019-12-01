import pandas as pd
from sklearn.ensemble import RandomForestClassifier

trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")
trainData.info()
testData.info()

trainData = trainData.drop(['PassengerId','Cabin','Ticket','Name'], axis=1)
trainData["Age"] = trainData["Age"].ffill()
trainData['Age'] = trainData['Age'].astype(int)
trainData["Embarked"] = trainData["Embarked"].ffill()
trainData['Fare'] = trainData['Fare'].fillna(0)
trainData['Fare'] = trainData['Fare'].astype(int)
trainData['Sex'] = trainData['Sex'].map({"male": 0, "female": 1})
trainData['Embarked'] = trainData['Embarked'].map({"S": 0, "C": 1, "Q": 2})

testData = testData.drop(['PassengerId','Cabin','Ticket','Name'], axis=1)
testData["Age"] = testData["Age"].ffill()
testData['Age'] = testData['Age'].astype(int)
testData["Embarked"] = testData["Embarked"].ffill()
testData['Fare'] = testData['Fare'].fillna(0)
testData['Fare'] = testData['Fare'].astype(int)
testData['Sex'] = testData['Sex'].map({"male": 0, "female": 1})
testData['Embarked'] = testData['Embarked'].map({"S": 0, "C": 1, "Q": 2})

xTrain = trainData.drop("Survived", axis=1)
yTrain = trainData["Survived"]
xTest = testData

random_forest = RandomForestClassifier(n_estimators=1000, max_depth=3)
random_forest.fit(xTrain, yTrain)
yResult = random_forest.predict(xTest)


for i in range(0, len(yResult)):
    print(str(i + 892) + ',' + str(yResult[i]))