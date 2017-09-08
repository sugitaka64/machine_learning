#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DMA Collaborative Filtering Model. predict score from collaborative filtering model."""

from datetime import datetime
import os
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
import sys

class DmaCfError(Exception):
    """common error class."""

    pass

class CreatePredictedScore(object):
    """predict score from collaborative filtering model."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        model_file_path: str,
        predict_data_dir_path: str,
        user_data_file_path: str,
        item_data_file_path: str,
        processed_training_data_file_path: str,
        data_limit: int=-1
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isdir(model_file_path):
                raise DmaCfError('invalid parameter: model_file_path')
            if not os.path.isdir(predict_data_dir_path):
                raise DmaCfError('invalid parameter: predict_data_dir_path')
            if not os.path.isfile(user_data_file_path):
                raise DmaCfError('invalid parameter: user_data_file_path')
            if not os.path.isfile(item_data_file_path):
                raise DmaCfError('invalid parameter: item_data_file_path')
            if not os.path.isdir(processed_training_data_file_path):
                raise DmaCfError('invalid parameter: processed_training_data_file_path')

        try:
            data_limit = int(data_limit)
        except ValueError:
            raise DmaCfError('invalid parameter: data_limit')
        except TypeError:
            raise DmaCfError('invalid parameter: data_limit')

        return True

    def run(
        self,
        model_file_path: str,
        predict_data_dir_path: str,
        user_data_file_path: str,
        item_data_file_path: str,
        processed_training_data_file_path: str,
        data_limit: int=-1
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            model_file_path,
            predict_data_dir_path,
            user_data_file_path,
            item_data_file_path,
            processed_training_data_file_path,
            data_limit
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('create_predicted_score')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # load user data
        users_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='false')\
            .load(user_data_file_path)
        users_id_rdd = users_df.rdd.map(lambda l: Row(user_id=l[0]))
        users_id_df = sqlContext.createDataFrame(users_id_rdd)

        # load item data
        items_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='false')\
            .load(item_data_file_path)
        items_id_rdd = items_df.rdd.map(lambda l: Row(item_id=l[0]))
        items_id_df = sqlContext.createDataFrame(items_id_rdd)

        # cross join user_id and item_id
        joined_df = users_id_df.join(items_id_df)
        joined_df.cache()

        # delete unnecessary variables
        del(users_df)
        del(users_id_rdd)
        del(users_id_df)
        del(items_df)
        del(items_id_rdd)
        del(items_id_df)

        # load training data
        custom_schema = StructType([
            StructField('user', StringType(), True),
            StructField('item', StringType(), True),
            StructField('rating', FloatType(), True),
            StructField('unique_user_id', IntegerType(), True),
            StructField('unique_item_id', IntegerType(), True),
        ])
        training_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(processed_training_data_file_path, schema=custom_schema)
        # users
        unique_users_rdd = training_df.rdd.map(lambda l: [l[0], l[3]])
        unique_users_df = sqlContext.createDataFrame(
            unique_users_rdd,
            ('user', 'unique_user_id')
        ).dropDuplicates()
        unique_users_df.cache()
        # items
        unique_items_rdd = training_df.rdd.map(lambda l: [l[1], l[4]])
        unique_items_df = sqlContext.createDataFrame(
            unique_items_rdd,
            ('item', 'unique_item_id')
        ).dropDuplicates()
        unique_items_df.cache()

        # delete unnecessary variables
        del(training_df)
        del(unique_users_rdd)
        del(unique_items_rdd)

        # add unique user id
        joined_df = joined_df.join(
            unique_users_df,
            joined_df['user_id'] == unique_users_df['user'],
            'inner'
        ).drop(unique_users_df['user'])

        # add unique item id
        joined_df = joined_df.join(
            unique_items_df,
            joined_df['item_id'] == unique_items_df['item'],
            'inner'
        ).drop(unique_items_df['item'])

        # load model
        model = ALSModel.load(model_file_path)

        # predict score
        predictions = model.transform(joined_df)
        all_predict_data = predictions\
            .select('user_id', 'item_id', 'prediction')\
            .filter('prediction > 0')

        # save
        ymd = datetime.today().strftime('%Y%m%d')
        # all score
        saved_data_file_path = predict_data_dir_path \
            + 'als_predict_data_all_%s.csv' % (ymd)
        all_predict_data.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(saved_data_file_path)
        copied_data_file_path = predict_data_dir_path + 'als_predict_data_all.csv'
        all_predict_data.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(copied_data_file_path)

        # limited score
        data_limit = int(data_limit)
        if data_limit > 0:
            all_predict_data.registerTempTable('predictions')
            sql = 'SELECT user_id, item_id, prediction ' \
                + 'FROM ( ' \
                + '  SELECT user_id, item_id, prediction, dense_rank() ' \
                + '  OVER (PARTITION BY user_id ORDER BY prediction DESC) AS rank ' \
                + '  FROM predictions ' \
                + ') tmp WHERE rank <= %d' % (data_limit)
            limited_predict_data = sqlContext.sql(sql)
        else:
            limited_predict_data = all_predict_data

        saved_data_file_path = predict_data_dir_path \
            + 'als_predict_data_limit_%s.csv' % (ymd)
        limited_predict_data.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(saved_data_file_path)
        copied_data_file_path = predict_data_dir_path + 'als_predict_data_limit.csv'
        limited_predict_data.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(copied_data_file_path)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'model_file_path',
        'predict_data_dir_path',
        'user_data_file_path',
        'item_data_file_path',
        'processed_training_data_file_path'
    ]
    usage = 'Usage:\n' \
        + '    create_predicted_score_script.py\n' \
        + '        --model_file_path=<model_file_path>\n' \
        + '        --predict_data_dir_path=<predict_data_dir_path>\n' \
        + '        --user_data_file_path=<user_data_file_path>\n' \
        + '        --item_data_file_path=<item_data_file_path>\n' \
        + '        --processed_training_data_file_path=<processed_training_data_file_path>\n' \
        + '        [--data_limit=<data_limit>]\n' \
        + '        [--env=<env>]\n' \
        + '    create_predicted_score_script.py -h | --help'

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
    ret['data_limit'] = -1
    ret['env'] = 'local'
    for actual_option in actual_options:
        if '--data_limit=' in actual_option:
            value = actual_option.split('=')[1]
            if value:
                ret['data_limit'] = value
        if '--env=' in actual_option:
            value = actual_option.split('=')[1]
            if value:
                ret['env'] = value

    return ret

if __name__ == '__main__':
    print('%s %s start.' % (datetime.today(), __file__))

    # check parameter
    parameters = __parser()
    if len(parameters) == 0:
        sys.exit(1)
    model_file_path = parameters['model_file_path']
    predict_data_dir_path = parameters['predict_data_dir_path']
    user_data_file_path = parameters['user_data_file_path']
    item_data_file_path = parameters['item_data_file_path']
    processed_training_data_file_path = parameters['processed_training_data_file_path']
    data_limit = parameters['data_limit']
    env = parameters['env']

    # execute
    cps = CreatePredictedScore(env)
    cps.run(
        model_file_path,
        predict_data_dir_path,
        user_data_file_path,
        item_data_file_path,
        processed_training_data_file_path,
        data_limit
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
