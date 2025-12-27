import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# ==================== 1. 训练数据处理 ====================
original_titanic_train = pd.read_csv("titanic_train.csv")
cleaned_titanic_train = original_titanic_train.copy()

# 类型转换
cleaned_titanic_train['PassengerId'] = cleaned_titanic_train['PassengerId'].astype('str')
cleaned_titanic_train['Survived'] = cleaned_titanic_train['Survived'].astype('int') # 目标变量通常用int
cleaned_titanic_train['Pclass'] = cleaned_titanic_train['Pclass'].astype('str') # 统一转为str
cleaned_titanic_train['Sex'] = cleaned_titanic_train['Sex'].astype('category')
cleaned_titanic_train['Embarked'] = cleaned_titanic_train['Embarked'].astype('category')

# 填充缺失值与特征工程
average_age = cleaned_titanic_train['Age'].mean()
cleaned_titanic_train['Age'] = cleaned_titanic_train['Age'].fillna(average_age)
cleaned_titanic_train['FamilyNum'] = cleaned_titanic_train['SibSp'] + cleaned_titanic_train['Parch']

# 筛选特征
lr_titanic_train = cleaned_titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare', 'Parch', 'SibSp'], axis=1)

# 独热编码 (One-Hot Encoding)
# 注意：这里会生成 Pclass_2, Pclass_3 (因为 drop_first=True)
lr_titanic_train = pd.get_dummies(lr_titanic_train, drop_first=True, columns=['Pclass', 'Sex'], dtype=int)

# 准备 X 和 y
y = lr_titanic_train['Survived']
X = lr_titanic_train.drop(['Survived'], axis=1)

# 保存一下特征列的顺序，确保测试集和训练集顺序完全一致！
feature_columns = X.columns
print(f"使用的特征: {feature_columns}")

# 归一化 (Scaling)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(X) # 训练集使用 fit_transform

# 训练模型
lr = LogisticRegression()
model = lr.fit(x_train, y)

# ==================== 2. 测试数据处理 ====================
titanic_test = pd.read_csv("titanic_test.csv")

# 1. 填充 Age (使用训练集的均值更严谨，但这里用测试集均值也行)
titanic_test['Age'] = titanic_test['Age'].fillna(average_age)

# 2. 这里的 Pclass 必须先转为 str，否则 pd.Categorical 匹配不到
titanic_test['Pclass'] = titanic_test['Pclass'].astype('str')
titanic_test['Pclass'] = pd.Categorical(titanic_test['Pclass'], categories=['1', '2', '3'])
titanic_test['Sex'] = pd.Categorical(titanic_test['Sex'], categories=['female', 'male'])
titanic_test['Embarked'] = pd.Categorical(titanic_test['Embarked'], categories=['C', 'Q', 'S'])

# 3. 特征工程
titanic_test['FamilyNum'] = titanic_test['SibSp'] + titanic_test['Parch']

# 4. 独热编码
titanic_test = pd.get_dummies(titanic_test, drop_first=True, columns=['Pclass', 'Sex'], dtype=int)

# 5. 【关键】确保列名和顺序与 X 完全一致
# 如果测试集因为数据缺失导致某些列不存在（比如测试集没人是 Pclass 2），这步会报错或缺列
# 使用 reindex 可以自动补全缺失列为 0，并忽略多余列
X_test = titanic_test.reindex(columns=feature_columns, fill_value=0)

# 6. 【关键修正】使用之前的 sc 对象进行归一化
# 注意：这里必须用 transform，绝对不能用 fit_transform
X_test_scaled = sc.transform(X_test)

# ==================== 3. 预测 ====================
pre_reslut = model.predict(X_test_scaled)
pre_reslut_proba = model.predict_proba(X_test_scaled)

print("预测结果前10个：")
print(pre_reslut[:10])


# 如果你想生成提交文件
# submission = pd.DataFrame({
#     "PassengerId": pd.read_csv("titanic_test.csv")["PassengerId"],
#     "Survived": pre_reslut
# })
# submission.to_csv("submission.csv", index=False)
