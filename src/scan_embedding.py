from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Flatten, SimpleRNN
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class Node:
    def __init__(self) -> None:
        self.lc = None
        self.rc = None
        self.idx = None
        self.operator = None
        self.data = {}


    def __init__(self, idx, operator) -> None:
        self.lc = None
        self.rc = None
        self.idx = idx
        self.operator = operator
        self.data = {}


    def __str__(self) -> str:
        return '('+ str(self.idx) + ') ' + self.operator + '\n' \
            + 'Left Child: ' + str(self.lc) + '\n'\
            + 'Right Child: ' + str(self.rc) + '\n'\
            + '\n'.join([k + ': ' + v for k, v in self.data.items()]) + '\n' \
            + '\n'

def get_min_max_vals(column_min_max_vals):
    with open('./data/tpcds_statistics.csv') as f:
        lines = f.readlines()
        for line in lines:
            col_name, min_val, max_val = line.strip().split(',')
            # print(col_name)
            column_min_max_vals[col_name] = (float(min_val), float(max_val))


def normalize_data(val, column_name):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    if (val > max_val):
        val = max_val
    elif (val < min_val):
        val = min_val
    val = float(val)
    try:
        val_norm = (val - min_val) / (max_val - min_val)
    except ZeroDivisionError:
        val_norm = val
    return val_norm


def is_not_number(s):
    try:
        float(s)
        return False
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return False
    except (TypeError, ValueError):
        pass
    return True


def get_data_and_label(path):
    sentences = []
    rows = []
    with open(path, 'rb') as f:
        forest = pickle.load(f)
        for tree in forest:
            for i in range(len(tree)):
                if tree[i].operator.startswith('Scan') and tree[i + 1].operator == 'Filter' and 'Number of output rows' in tree[i].data:
                    row = eval(tree[i].data['Number of output rows'])
                    sentence = []
                    for k in tree[i + 1].data:
                        if 'Input' in k:
                            sentence.extend(tree[i + 1].data[k][1:-1].split(', '))
                    cond = tree[i + 1].data['Condition'].replace('(', ' ').replace(')', ' ').strip().split()
                    for j in range(len(cond)):
                        if not is_not_number(cond[j]):
                            if cond[j - 2] in column_min_max_vals:
                                cond[j] = str(normalize_data(eval(cond[j]), cond[j - 2]))
                    sentence.extend(cond)
                    sentences.append(sentence)
                    rows.append(row)
    return sentences, rows


def prepare_data_and_label(sentences, rows):
    data = []
    label = []
    for sentence, row in zip(sentences, rows):
        _s = []
        for word in sentence:
            if (is_not_number(word)):
                _tmp = np.column_stack((np.array([0]), vocab_dict[word]))
                _tmp = np.reshape(_tmp, (vocab_size+1))
                assert (len(_tmp) == vocab_size+1)
                _s.append(_tmp)
            else:
                _tmp = np.full((vocab_size+1), word)
                assert (len(_tmp) == vocab_size+1)
                _s.append(_tmp)
        data.append(np.array(_s))
        label.append(row)
    return data, label


def normalize_labels(labels, min_val=None, max_val=None):
    # log tranformation withour normalize
    labels = [l if l > 0 else 1 for l in labels]
    labels = np.array([np.log(float(l)) for l in labels]).astype(np.float32)
    return labels, 0, 1

if __name__ == '__main__':
    column_min_max_vals = {}
    get_min_max_vals(column_min_max_vals)
    sentences, rows = get_data_and_label('./nn/data/hinted_forest.pkl')
    vocab_dict = {}
    vocabulary = []
    for sentence in sentences:
        for word in sentence:
            if (word not in vocabulary and is_not_number(word)):
                vocabulary.append(word)
    # print('len(vocabulary): ', len(vocabulary))
    vocab_size = len(vocabulary)
    print(vocabulary)
    _vocabulary = np.array(vocabulary)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(_vocabulary)
    encoded = to_categorical(integer_encoded)
    for v, e in zip(vocabulary, encoded):
        vocab_dict[v] = np.reshape(np.array(e), (1, vocab_size))


    data, label = prepare_data_and_label(sentences, rows)
    label_norm, min_val, max_val = normalize_labels(label)

    max_len = 0
    for sentence in sentences:
        if (len(sentence) > max_len):
            max_len = len(sentence)
    print(max_len)
    padded_sentences = pad_sequences(
        data, maxlen=max_len, padding='post', dtype='float32')

    # print(np.shape(padded_sentences))
    # print(np.shape(label_norm))

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sentences, label_norm, test_size=0.8, random_state=40)
    # print(np.shape(X_train), np.shape(X_test))
    # print(np.shape(y_train), np.shape(y_test))
    # print('X_train: ', X_train[:5])
    # print('y_train: ', y_train[:5])
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True,
            activation='relu', input_shape=(max_len, vocab_size+1)))
    model.add(SimpleRNN(128, return_sequences=True, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=optimizers.legacy.Adagrad(learning_rate=0.001),
                  loss='mse', metrics=['mse', 'mae'])

    model.summary()

    # history = model.fit(padded_sentences, label_norm, validation_split=0.2,
    #           epochs=100, batch_size=128, shuffle=True)
    # model.save('./model/embedding_model.keras')

    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # # 获取训练周期数
    # epochs = range(1, len(train_loss) + 1)

    # # 绘制损失曲线
    # plt.figure()
    # plt.plot(epochs, train_loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


    model.load_weights("./nn/model/embedding_model.keras")
    test_sentences, test_rows = get_data_and_label('./nn/data/hinted_forest.pkl')
    test_data, test_label = prepare_data_and_label(test_sentences, test_rows)

    test_padded_sentences = pad_sequences(
        test_data, maxlen=max_len, padding='post', dtype='float32')
    # print(np.shape(test_padded_sentences))
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.layers[4].output)
    intermediate_output = intermediate_layer_model.predict(test_padded_sentences)

    label_dict = {}
    for s, v in zip(test_sentences, intermediate_output):
        label_dict[' '.join(s)] = v
    with open('./data/scan_label_dict.pkl', 'wb') as file:
        pickle.dump(label_dict, file)