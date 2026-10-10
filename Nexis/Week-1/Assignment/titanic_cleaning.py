import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data = pd.read_csv('Titanic-Dataset.csv')
#print(data.head(5))
#print(data.info())
#print(data.describe())
print("missing values before cleaning")
print(data.isnull().sum())

data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = data.drop(columns=['Cabin'])
print("missing values after cleaning")
print(data.isnull().sum())
print("dataset shape:",data.shape)
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
print("After Label Encoding:")
print(data['Sex'].head())
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(data[['Embarked']])
encoded_df = pd.DataFrame(encoded,columns=ohe.get_feature_names_out(['Embarked']),index=data.index)
data = pd.concat([data.drop(columns=['Embarked']), encoded_df],axis=1)
print("\nAfter One-Hot Encoding:")
print(data.head())
print("\nFinal shape:", data.shape)

plt.figure(figsize=(8, 5))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()
data.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")


