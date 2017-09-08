#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DODA plus, Prediction for Passing Paper exam.
devide csv by "colmun_name" parameter.
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import sys

class DpPPPError(Exception):
    """common error class."""

    pass

class DivideCsv(object):
    """devide csv by "colmun_name" parameter."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        data_for_prediction_file_path: str,
        predict_data_dir_path: str,
        colmun_name: str
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isfile(data_for_prediction_file_path):
                raise DpPPPError('invalid parameter: data_for_prediction_file_path')
            if not os.path.isdir(predict_data_dir_path):
                raise DpPPPError('invalid parameter: predict_data_dir_path')

        return True

    def run(
        self,
        data_for_prediction_file_path: str,
        predict_data_dir_path: str,
        colmun_name: str
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            data_for_prediction_file_path,
            predict_data_dir_path,
            colmun_name
        )

        output_csv_file_path_fmt = predict_data_dir_path \
            + '%s_%s.csv'
        ymd = datetime.today().strftime('%Y%m%d')

        # read csv
        df = pd.read_csv(
            data_for_prediction_file_path,
            dtype=np.unicode
        )
        df = df.sort_values(by=[colmun_name], ascending=True)
        unique_columns = df[colmun_name].unique()

        # divide csv
        for column in unique_columns:
            csv_file_path = output_csv_file_path_fmt % (column, ymd)
            tmp_df = df[df[colmun_name].isin([column])]\
                .sort_values(by=['score'], ascending=False)
            tmp_df.to_csv(csv_file_path, index=False)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'data_for_prediction_file_path',
        'predict_data_dir_path',
        'colmun_name'
    ]
    usage = 'Usage:\n' \
        + '    divide_csv.py\n' \
        + '        --data_for_prediction_file_path=<data_for_prediction_file_path>\n' \
        + '        --predict_data_dir_path=<predict_data_dir_path>\n' \
        + '        --colmun_name=<colmun_name>\n' \
        + '        [--env=<env>]\n' \
        + '    divide_csv.py -h | --help'

    if len(arguments) == 1:
        # no parameter
        print(usage)
        return {}

    # get parameter keys
    arguments.pop(0)
    actual_options = []
    for argument in arguments:
        if argument.startswith('-'):
            actual_options.append(argument)

    if len(actual_options) == 0:
        # no parameter key
        print(usage)
        return {}

    if '-h' in actual_options or '--help' in actual_options:
        # help
        print(usage)
        return {}

    # check essential parameters
    ret = {}
    for predicted_option in predicted_options:
        for actual_option in actual_options:
            if '--' + predicted_option + '=' in actual_option:
                value = actual_option.split('=')[1]
                if value:
                    ret[predicted_option] = value
    if len(ret) != len(predicted_options):
        print(usage)
        return {}

    # check optional parameters
    ret['env'] = 'local'
    for actual_option in actual_options:
        if '--env=' in actual_option:
            value = actual_option.split('=')[1]
            if value:
                ret['env'] = value

    return ret

if __name__ == '__main__':
    print('%s %s start.' % (datetime.today(), __file__))

    # check parameters
    parameters = __parser()
    if len(parameters) == 0:
        sys.exit(1)
    data_for_prediction_file_path = parameters['data_for_prediction_file_path']
    predict_data_dir_path = parameters['predict_data_dir_path']
    colmun_name = parameters['colmun_name']
    env = parameters['env']

    # execute
    ds = DivideCsv(env)
    ds.run(
        data_for_prediction_file_path,
        predict_data_dir_path,
        colmun_name
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
