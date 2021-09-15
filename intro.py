import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# create a pipeline object
iris = load_iris()

print(iris.target[[10,25,50]])
# data -> 
# target -> quem deve ser previsto
print(list(iris.target_names))
print(iris.data)
# exibindo dados em forma de tabela com pandas
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head())
# dividindo entre treino e teste
X = iris.data[:, :2]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(iris_df)
