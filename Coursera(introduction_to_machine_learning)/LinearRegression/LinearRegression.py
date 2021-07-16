
from numpy.ma import hstack
from pandas import read_csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

train_data = read_csv('./resources/salary-train.csv')
test_data = read_csv('./resources/salary-test-mini.csv')

train_data['FullDescription'] = train_data['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
test_data['FullDescription'] = test_data['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)

vec = TfidfVectorizer(min_df=5)
x_train_text = vec.fit_transform(train_data["FullDescription"])

train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
x_train_enc = enc.fit_transform(train_data[["LocationNormalized", "ContractTime"]].to_dict("records"))
x_train = hstack([x_train_text, x_train_enc])

y_train = train_data["SalaryNormalized"]
model = Ridge(alpha=1, random_state=241)
model.fit(x_train, y_train)

x_test_text = vec.transform(test_data["FullDescription"])
x_test_enc = enc.transform(test_data[["LocationNormalized", "ContractTime"]].to_dict("records"))
x_test = hstack([x_test_text, x_test_enc])
y_test = model.predict(x_test)
print(y_test)
