from classification import Classification

# SVM, DNN共通パラメーター
RANDOM_STATE = 2525
# SVMパラメーター
PARAMS = [0.001, 0.01, 0.1, 1, 10, 100]
DECISION_FUNCTION_SHAPE = "ovo"
# DNNパラメーター
HIDDEN_LAYER_SIZES = (1024, 512, 256, 128)
ALPHA = [0.001, 0.01, 0.1]
MAX_ITER = 3000


class GridSearch(object):

    def __init__(self, dataset_lists, feature="edge_hist",
                 learning_model="dnn", score_type="all"):
        self.feature = feature
        self.learning_model = learning_model
        self.score_type = score_type
        self.best_accuracy = 0
        self.best_recall = 0
        self.best_params = {}
        self.best_confusion_matrix = 0
        self.clf = Classification(dataset_lists, feature, learning_model,
                                  output_every_scores=False)

    def run(self):
        print(f"{'='*30} grid search {'='*30}")
        print(f"{'='*10} feature: {self.feature}, "
              f"model: {self.learning_model}, "
              f"score_type: {self.score_type} {'='*10}")

        if self.learning_model == "dnn":
            self._dnn_grid_search()
        elif self.learning_model == "svm":
            self._svm_grid_search()
        self._output_best_scores()

    def _dnn_grid_search(self):
        for alpha in ALPHA:
            self.params = {
                "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
                "alpha": alpha,
                "max_iter": MAX_ITER,
                "random_state": RANDOM_STATE,
            }
            self.grid_search()

    def _svm_grid_search(self):
        for c in PARAMS:
            for gamma in PARAMS:
                self.params = {
                    "C": c,
                    "gamma": gamma,
                    "decision_function_shape": DECISION_FUNCTION_SHAPE,
                    "random_state": RANDOM_STATE,
                }
                self.grid_search()

    def grid_search(self):
        self.accuracy, self.recall, self.confusion_matrix = \
                self.clf.train_and_test(**self.params)
        if self.score_type == "all":
            self._update_best_scores()
        elif self.score_type == "fake":
            self._update_best_fake_scores()

    # 正解率を重視
    def _update_best_scores(self):
        if self.accuracy > self.best_accuracy:
            self._update_scores()

    # 検出率を重視
    def _update_best_fake_scores(self):
        if self.recall > self.best_recall:
            self._update_scores()

    def _update_scores(self):
        self.best_accuracy = self.accuracy
        self.best_recall = self.recall
        self.best_params = self.params
        self.best_confusion_matrix = self.confusion_matrix

    def _output_best_scores(self):
        print(f"best params: {self.best_params}")
        print(f"best accuracy: {self.best_accuracy * 100} %")
        print(f"best recall: {self.best_recall * 100} %")
        print(self.best_confusion_matrix)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", default="dnn", type=str,
                        choices=["dnn", "svm"],
                        help="select leaning model {'dnn','svm'}, \
                              default='dnn'")
    parser.add_argument("-s", "--score_type", default="all", type=str,
                        choices=["all", "fake"],
                        help="select score type {'all','fake'}, \
                              default='all'")
    args = parser.parse_args()
    model = args.model
    score_type = args.score_type

    from dataset import get_nearly_real_fake_dataset_lists
    dataset_lists = get_nearly_real_fake_dataset_lists()

    gs = GridSearch(dataset_lists, learning_model=model, score_type=score_type)
    gs.run()
