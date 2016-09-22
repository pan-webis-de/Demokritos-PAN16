#!/usr/bin/python

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, confusion_matrix
from pan import ProfilingDataset
from sklearn.externals import joblib


def test_data(dataset, model, task):
    """ evaluate model on test data

    :dataset: The dataset to evaluate model on
    :model: The trained model to use for prediction
    :task: The task this is for

    """

    X, y = dataset.get_data(feature=task)
    predict = model.predict(X)
    print('\n-- Predictions for %s --' % task)
    try:
        # if it's classification we measure micro and macro scores
        acc = accuracy_score(y, predict)
        conf = confusion_matrix(y, predict, labels=list(set(y)))
        print('Accuracy : {}'.format(acc))
        print('Confusion matrix :\n {}'.format(conf))
    except:
        print('Something went wrong with the prediction!')
    dataset.set_labels(task, predict)


if __name__ == '__main__':
    parser = ArgumentParser(description='Test trained model on pan dataset')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-o', '--output', type=str,
                        required=True, dest='outfolder',
                        help='path to folder where results should be written')
    parser.add_argument('-m', '--model', type=str,
                        required=True, dest='model',
                        help='path to learned model to use for predictions')

    args = parser.parse_args()
    model = args.model
    infolder = args.infolder
    outfolder = args.outfolder
    print('Loading dataset...')
    dataset = ProfilingDataset(infolder)
    print('Loaded {} users...\n'.format(len(dataset.entries)))
    config = dataset.config
    tasks = config.tasks
    ######
    # This part for home usage
    all_models = joblib.load(model)
    ######
    if not all(task in tasks for task in all_models.keys()):
        print("The models you are using aren't all specified in config file")
        print('Did you change the config file after training???!')
        print('Exiting.. try training again.')
        exit(1)
    print('\n--------------- Thy time of Judgement ---------------')
    for task in tasks:
        test_data(dataset, all_models[task], task)
    # write output to file
    dataset.write_data(outfolder)
