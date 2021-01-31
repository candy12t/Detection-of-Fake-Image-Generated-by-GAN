"""
TN(True Negative): 実際のクラスがFalseで予測もFalse(正解)
FP(False Positive): 実際のクラスはFalseで予測がTrue(不正解)
FN(False Negative): 実際のクラスはTrueで予測がFalse(不正解)
TP(True Positive): 実際のクラスがTrueで予測もTrue(正解)

[[TN, FP]
 [FN, TP]]
"""


import pickle
from datetime import datetime

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from dataset import load_dataset

SPLITS = 5
RANDOM_STATE = 2525


class Classification(object):

    def __init__(self, dataset_lists, feature='edge_hist',
                 learning_model='dnn', output_every_scores=True,
                 save_model=False, output_failed_data=False):
        self.feature = feature
        self.learning_model = learning_model
        self.output_every_scores = output_every_scores
        self.save_model = save_model
        self.output_failed_data = output_failed_data
        self.i = 0
        self.X, self.y, self.z = load_dataset(dataset_lists=dataset_lists,
                                              feature=feature)  # データセット読み込み
        self.skf = StratifiedKFold(n_splits=SPLITS,
                                   shuffle=True,
                                   random_state=RANDOM_STATE)  # 交差検証
        self.fp_list, self.fn_list = [], []

    def run(self, **params):
        print(f"{'='*10} feature: {self.feature}, "
              f"model: {self.learning_model} {'='*10}")

        # 学習テストし, 結果を出力
        accuracy, recall, cm = self.train_and_test(**params)
        self._output_scores(accuracy, recall, cm, mode='all')

        # 分類に失敗した(不正解)データを出力
        if self.output_failed_data:
            self._output_failed_data()

    # 学習, テスト
    def train_and_test(self, **params):

        self.learning_model == 'svm'
        y_test_list, y_pred_list = [], []

        for train_index, test_index in self.skf.split(self.X, self.y):
            self.i += 1
            # データ分割
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            z_test = self.z[test_index]

            # 分類器
            if self.learning_model == 'dnn':
                self.model = MLPClassifier(**params)
            elif self.learning_model == 'svm':
                self.model = SVC(**params)
            self.model.fit(X_train, y_train)  # 学習

            y_pred = self.model.predict(X_test)  # 予測

            # 全体の正解率, 検出率, 混同行列を出すために各予測, テストをまとめる
            y_pred_list.extend(y_pred)
            y_test_list.extend(y_test)

            # 学習毎に正解率, 検出率, 混同行列を算出, 出力
            if self.output_every_scores:
                accuracy, recall, cm = self._calculate_scores(y_test, y_pred)
                self._output_scores(accuracy, recall, cm, mode='every')

            # 学習済みモデルを保存
            if self.save_model:
                self._save_model()

            # 分類に失敗した(不正解)のデータを記録
            if self.output_failed_data:
                self._record_failed_data(y_test, y_pred, z_test)

        return self._calculate_scores(y_test_list, y_pred_list)

    # 正解率(accuray), 検出率(recall), 混同行列(confusion_matrix)を算出
    def _calculate_scores(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=0)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, recall, cm

    # 正解率, 検出率, 混同行列を出力
    def _output_scores(self, accuracy, recall, cm, mode):
        if mode == 'every':
            print(f"{'='*20} {self.i}/{SPLITS} {'='*20}")
        elif mode == 'all':
            print(f"{'='*20} all {'='*20}")
        print(f"accuracy: {accuracy*100} %")
        print(f"recall: {recall*100} %")
        print(cm)

    # 分類に失敗した(不正解)データ一覧を出力
    def _output_failed_data(self):
        print(f"{'='*10} classification failed data {'='*10}")
        print(f"fn = {self.fn_list}")
        print(f"fp = {self.fp_list}")

    # 学習済みモデルを保存
    def _save_model(self):
        dt = datetime.now().strftime('%Y%m%d%H%M')
        save_path = f"./../model/{self.learning_model}_{dt}_{self.i}.pickle"
        with open(save_path, mode='wb') as f:
            pickle.dump(self.model, f)

    # 分類に失敗した(不正解)のデータを記録
    def _record_failed_data(self, y_test, y_pred, z_test):
        for test, pred, filename in zip(y_test, y_pred, z_test):
            fp = test == 0 and pred == 1
            fn = test == 1 and pred == 0
            if fp:
                self.fp_list.append(filename)
            elif fn:
                self.fn_list.append(filename)
