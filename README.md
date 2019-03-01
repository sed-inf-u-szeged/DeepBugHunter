# DeepBugHunter™

Copyright (c) 2018-2019 Department of Software Engineering, University of Szeged, Hungary.


## About

DeepBugHunter is an Python framework whose goal is to make it easier to experiment with deep learning and compare its capabilities with more "traditional" machine learning algorithms. The name comes from its original purpose (testing deep neural networks in combination with static source code metrics to predict bugs) but its design is intentionally more general and extensible so it can be a helpful resource for anyone looking for a higher abstraction level tool to perform machine learning studies and model building.


### Prerequisites

The user should prepare a Python 3 environment (3.6 and up) with the following packages installed:

- numpy 1.14.3 and up
- scikit-learn 0.19.2 and up
- tensorflow 1.8.0 and up
- pandas 0.22.0 and up


### Usage

Running DeepBugHunter consists of calling the main executable (`dbh.py`) with a previously cleaned and valid CSV dataset and a list of machine learning strategies to try on it. After an optional preprocessing phase, the script automatically partitions the input dataset into training, development, and testing sets, applies the strategies, and calculates a score (we concentrated on F-measure, but others are easy to add) based on a cross validation. Most current strategies are thin wrappers around existing `sklearn` algorithms, while `sdnnc` and `cdnnc` are the simple and complex implementations of a Deep Neural Network (DNN) using Tensorflow.

In order to not duplicate the list of command line arguments (thereby avoiding the danger of getting out of sync with the source code), we rely instead on the verbose and human-readable nature of the `argparse` Python module and refer the reader to the "Arguments" section of `dbh.py`, listing the possible switches relating to the overall infrastructure.
We additionally note that the second half of the `--strategy` switch expects a whole other set of independent arguments corresponding to the machine learning strategy selected in the first half. These arguments will be parsed using the same `argparse` module, only one layer lower. Their definition can be found in the beginning of the individual strategy files withing the `strategies` folder.


### Contribution

If you would like to contribute, contact Rudolf Ferenc <zealot@inf.u-szeged.hu> and Dénes Bán <zealot@inf.u-szeged.hu> and we'll figure something out. PRs welcome.


## License

DeepBugHunter is licensed under the Apache License 2.0 (see the LICENSE file).

If you find this tool useful, please drop us a message.
