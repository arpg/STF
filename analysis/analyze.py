#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

pose_columns = ('x', 'y', 'z')
pose_errors = [label + '_error' for label in pose_columns]
suffixes = ('_truth', '_est')


def diff(truth, est):
    # For every keyframe
    # See if we have ground truth at that time
    joined = pd.merge(truth, est, on='timestamp', suffixes=suffixes,
                      how='inner')

    for label in pose_columns:
        joined[label + '_error'] = (joined.ix[:, label + '_truth'] -
                                    joined.ix[:, label + '_est'])

    joined['euc_error'] = joined[pose_errors].apply(
        lambda x: np.sqrt(x.dot(x)), axis=1)
    joined.to_csv('combined.csv')
    return joined


def transform(truth, est):
    # First pose of truth to adjust estimated poses into its
    # coordinate/time frame
    truth_origin = truth.loc[0]

    # Adjust estimated poses
    est.ix[:, pose_columns] += truth_origin

    # First frame should be at t = 0
    est['timestamp'] -= est.ix[0, 'timestamp']


def compute_metrics(pose_errors):
    stats = {}
    stats['euc_error_mean'] = pose_errors['euc_error'].mean()
    stats['euc_error_median'] = pose_errors['euc_error'].median()
    stats['euc_error_std'] = pose_errors['euc_error'].std()
    return stats


def load_data(truth_file, est_file):
    return pd.read_csv(truth_file), pd.read_csv(est_file)


def plot_metrics(data, stats):
    plt.subplot(121)
    plt.plot(data['timestamp'], data['euc_error'], label='euclidean')
    plt.plot(data['timestamp'], data['x_error'], label='x')
    plt.plot(data['timestamp'], data['y_error'], label='y')
    plt.plot(data['timestamp'], data['z_error'], label='z')
    plt.axhline(y=stats['euc_error_mean'])
    plt.legend(loc=3)
    plt.title('Errors')

    plt.subplot(122)
    plt.plot(data['x_truth'], data['y_truth'], label='truth')
    plt.plot(data['x_est'], data['y_est'], label='est')
    plt.legend(loc=3)
    plt.title('XY-plane')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare ground truth poses with estimated poses.')
    parser.add_argument('-gt', '--ground-truth', required=True)
    parser.add_argument('-est', '--estimates', required=True)
    args = parser.parse_args()

    truth, est = load_data(args.ground_truth, args.estimates)

    # Perform transformations on estimated poses, if necessary
    transform(truth, est)

    # Compare to truth poses
    pose_errors = diff(truth, est)

    # Compute metrics
    metrics = compute_metrics(pose_errors)

    # Write/print metrics
    plot_metrics(pose_errors, metrics)


if __name__ == '__main__':
    main()
