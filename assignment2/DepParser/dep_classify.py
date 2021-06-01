import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression
import numpy as np


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """
    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        """
        #
        # YOUR CODE HERE
        #
        moves = []
        i, stack, pred_tree = 0, [], [0] * len(words)
        probs = model.get_log_probs(ds.dp2array(words, tags, i, stack))

        # get ranking of the probabilities
        best_prob, best_action = -float("inf"), None
        for move, prob in enumerate(probs):
            if move in self.__parser.valid_moves(i, stack, pred_tree):
                if prob > best_prob:
                    best_prob = prob
                    best_action = move

        moves.append(best_action)
        i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, moves[-1])

        while True:
            ds.add_datapoint(words, tags, i, stack, False)
            # probs = model.classify_word(ds.to_arrays()[0][-1])[0]
            probs = model.get_log_probs(ds.dp2array(words, tags, i, stack))

            best_prob, best_action = -float("inf"), None
            for move, prob in enumerate(probs):
                if move in self.__parser.valid_moves(i, stack, pred_tree):
                    if prob > best_prob:
                        best_prob = prob
                        best_action = move

            moves.append(best_action)

            ds.datapoints.pop(0)
            i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, moves[-1])
            if len(stack) == 1 and i >= len(pred_tree):
                break

        return moves


    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`
        
        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        p = self.__parser
        arc = 0
        correct_arc = 0
        sentence_cnt = 0
        sentence_total = 0
        n_move = 0
        n_move_correct = 0
        with open(test_file, encoding="utf-8") as source:
            for words, tags, tree, relations in p.trees(source):
                correct_moves = p.compute_correct_moves(tree)
                sentence_total += 1
                moves = self.build(model, words, tags, ds)
                flag = 1
                for i in range(len(correct_moves)):
                    n_move += 1
                    if correct_moves[i] != 0:
                        arc += 1
                    if moves[i] == correct_moves[i]:
                        n_move_correct += 1
                        if moves[i] != 0:
                            correct_arc += 1
                    else:
                        flag = 0
                if flag == 1:
                    sentence_cnt += 1
        print("Sentence-level accuracy: " + str(100 * sentence_cnt / sentence_total) + "%")
        print("UAS: " + str(100 * correct_arc / arc) + "%")
        print("Move accuracy: " + str(100 * n_move_correct / n_move) + "%")


if __name__ == '__main__':
    #
    # TODO:
    # 1) Replace the `create_dataset` function from dep_parser_fix.py to your dep_parser.py file
    # 2) Replace parse_dataset.py with the given new version
    #

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)

    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open('model.pkl', 'wb'))
    
    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())
    
    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev.conllu', ds)