#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DMA Collaborative Filtering Model. processing training data."""

from datetime import datetime
import os
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import sys

class DmaCfError(Exception):
    """common error class."""

    pass

class ProcessTrainingData(object):
    """get training data from Redshift, and add sequence number to data."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        unprocessed_data_file_path: str,
        training_data_dir_path: str
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isfile(unprocessed_data_file_path):
                raise DmaCfError('invalid parameter: unprocessed_data_file_path')
            if not os.path.isdir(training_data_dir_path):
                raise DmaCfError('invalid parameter: training_data_dir_path')

        return True

    def __get_action_log(
        self,
        sqlContext: SQLContext,
        unprocessed_data_file_path: str
    ) -> DataFrame:
        """get data."""
        df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(unprocessed_data_file_path)

        return df

    def run(
        self,
        unprocessed_data_file_path: str,
        training_data_dir_path: str
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            unprocessed_data_file_path,
            training_data_dir_path
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('process_training_data')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # get data
        df = self.__get_action_log(sqlContext, unprocessed_data_file_path)

        # make sequence number of users
        unique_users_rdd = df.rdd.map(lambda l: l[0]).distinct().zipWithIndex()
        unique_users_df = sqlContext.createDataFrame(
            unique_users_rdd,
            ('user', 'unique_user_id')
        )

        # make sequence number of items
        unique_items_rdd = df.rdd.map(lambda l: l[1]).distinct().zipWithIndex()
        unique_items_df = sqlContext.createDataFrame(
            unique_items_rdd,
            ('item', 'unique_item_id')
        )

        # add sequence number of users, sequence number of items to data
        df = df.join(
            unique_users_df,
            df['user'] == unique_users_df['user'],
            'inner'
        ).drop(unique_users_df['user'])
        df = df.join(
            unique_items_df,
            df['item'] == unique_items_df['item'],
            'inner'
        ).drop(unique_items_df['item'])

        # save
        ymd = datetime.today().strftime('%Y%m%d')
        saved_data_file_path = training_data_dir_path \
            + 'cf_training_data_%s.csv' % (ymd)
        df.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(saved_data_file_path)

        # copy directory
        copied_data_file_path = training_data_dir_path + 'cf_training_data.csv'
        df.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(copied_data_file_path)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'unprocessed_data_file_path',
        'training_data_dir_path'
    ]
    usage = 'Usage:\n' \
        + '    process_training_data_script.py\n' \
        + '        --unprocessed_data_file_path=<unprocessed_data_file_path>\n' \
        + '        --training_data_dir_path=<training_data_dir_path>\n' \
        + '        [--env=<env>]\n' \
        + '    process_training_data_script.py -h | --help'

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
    unprocessed_data_file_path = parameters['unprocessed_data_file_path']
    training_data_dir_path = parameters['training_data_dir_path']
    env = parameters['env']

    # execute
    ptd = ProcessTrainingData(env)
    ptd.run(unprocessed_data_file_path, training_data_dir_path)

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
