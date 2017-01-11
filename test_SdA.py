from SdA_wm import SdA
from load_data import load_data
import pickle
from sklearn.preprocessing import MinMaxScaler

X, Y = load_data()
min_max_scaler = MinMaxScaler()  ## min max scaler
min_max_scaler.fit(X)
X = min_max_scaler.transform(X)
print X.shape
print Y.shape

train_set_x = X[:500]
valid_set_x = X[500:650]
test_set_x = X[650:]

train_set_y = Y[:500]
valid_set_y = Y[500:650]
test_set_y = Y[650:]


print 'loading data finished...'

# sda = SdA(hidden_layers_sizes=[1500, 2000, 2000, 2000, 1500], pretrain_epochs=120, finetune_epochs=800)
sda = SdA(hidden_layers_sizes=[800, 600], pretrain_epochs=12, finetune_epochs=80)
sda.pretraining(train_set_x)
sda.finetuning(train_set_x, train_set_y, valid_set_x, valid_set_y)

with open('SdA_best_model.pkl', 'wb') \
        as f:
    pickle.dump(sda, f)

test_y_pred_proba = sda.predict_proba(test_set_x)
test_y_pred = sda.predict(test_set_x)

print test_y_pred_proba
print test_y_pred
print test_set_y
