from classification import Classification

# SVMパラメーター
C = 1
GAMMA = 1
DECISION_FUNCTION_SHAPE = "ovo"
# DNNパラメーター
HIDDEN_LAYER_SIZES = (1024, 512, 256, 128)
ALPHA = 0.01
MAX_ITER = 3000
# SVM, DNN共通パラメーター
RANDOM_STATE = 2525

SVM_PARAMS = {
    "C": C,
    "gamma": GAMMA,
    "decision_function_shape": DECISION_FUNCTION_SHAPE,
    "random_state": RANDOM_STATE,
}
DNN_PARAMS = {
    "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
    "alpha": ALPHA,
    "max_iter": MAX_ITER,
    "random_state": RANDOM_STATE,
}


def main(dataset_lists, feature, model, save_model, output_failed_data):
    params = DNN_PARAMS if model == "dnn" else SVM_PARAMS
    clf_params = {
        "dataset_lists": dataset_lists,
        "feature": feature,
        "learning_model": model,
        "save_model": save_model,
        "output_failed_data": output_failed_data,
    }
    clf = Classification(**clf_params)
    clf.run(**params)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-f", "--feature", default="edge_hist", type=str,
                        choices=["hist", "edge", "edge_hist"],
                        help="select feature {'hist', 'edge', 'edge_hist'}, \
                              default='edge_hist'")
    parser.add_argument("-m", "--model", default="dnn", type=str,
                        choices=["dnn", "svm"],
                        help="select leaning model {'dnn','svm'}, \
                              default='dnn'")
    parser.add_argument("-s", "--save_model", action="store_true",
                        help="save model with option")
    parser.add_argument("-o", "--output_failed_data", action="store_true",
                        help="output classification failed data with option")
    args = parser.parse_args()
    feature = args.feature
    model = args.model
    save_model = args.save_model
    output_failed_data = args.output_failed_data

    from dataset import get_nearly_real_fake_dataset_lists
    dataset_lists = get_nearly_real_fake_dataset_lists()

    main(dataset_lists, feature, model, save_model, output_failed_data)
