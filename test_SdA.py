from SdA_wm import SdA
from load_data import load_data
import pickle

datasets = load_data()
valid_set_x, valid_set_y = datasets[1]
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[2]

test_set_x = test_set_x.get_value()

print 'loading data finished...'

sda = SdA(hidden_layers_sizes=[1500, 2000, 2000, 2000, 1500], pretrain_epochs=120, finetune_epochs=800)
sda.pretraining(train_set_x)
sda.finetuning(train_set_x, train_set_y, valid_set_x, valid_set_y)

with open('SdA_best_model.pkl', 'wb') \
        as f:
    pickle.dump(sda, f)

test_y_pred_proba = sda.predict_proba(test_set_x)
test_y_pred = sda.predict(test_set_x)

print test_y_pred_proba[0:20]
print test_y_pred[0:20]
print test_set_y[0:20]
