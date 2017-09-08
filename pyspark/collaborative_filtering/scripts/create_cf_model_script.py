#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DMA Collaborative Filtering Model. create collaborative filtering model."""

from datetime import datetime
import os
from pyspark.ml.recommendation import ALS
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

class CreateCfModel(object):
    """create collaborative filtering model."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        processed_training_data_file_path: str,
        model_dir_path: str,
        rank: int,
        max_iter: int,
        implicit_prefs: str,
        alpha: float,
        num_user_blocks: int,
        num_item_blocks: int
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isdir(processed_training_data_file_path):
                raise DmaCfError('invalid parameter: processed_training_data_file_path')
            if not os.path.isdir(model_dir_path):
                raise DmaCfError('invalid parameter: model_dir_path')

        try:
            rank = int(rank)
        except ValueError:
            raise DmaCfError('invalid parameter: rank')
        except TypeError:
            raise DmaCfError('invalid parameter: rank')

        try:
            max_iter = int(max_iter)
        except ValueError:
            raise DmaCfError('invalid parameter: max_iter')
        except TypeError:
            raise DmaCfError('invalid parameter: max_iter')

        if implicit_prefs != 'True' and implicit_prefs != 'False':
            raise DmaCfError('invalid parameter: implicit_prefs')

        try:
            alpha = float(alpha)
        except ValueError:
            raise DmaCfError('invalid parameter: alpha')
        except TypeError:
            raise DmaCfError('invalid parameter: alpha')

        try:
            num_user_blocks = int(num_user_blocks)
        except ValueError:
            raise DmaCfError('invalid parameter: num_user_blocks')
        except TypeError:
            raise DmaCfError('invalid parameter: num_user_blocks')

        try:
            num_item_blocks = int(num_item_blocks)
        except ValueError:
            raise DmaCfError('invalid parameter: num_item_blocks')
        except TypeError:
            raise DmaCfError('invalid parameter: num_item_blocks')

        return True

    def run(
        self,
        processed_training_data_file_path: str,
        model_dir_path: str,
        rank: int,
        max_iter: int,
        implicit_prefs: str,
        alpha: float,
        num_user_blocks: int,
        num_item_blocks: int
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            processed_training_data_file_path,
            model_dir_path,
            rank,
            max_iter,
            implicit_prefs,
            alpha,
            num_user_blocks,
            num_item_blocks
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('create_cf_model')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # create model
        als = ALS(
            rank=int(rank),
            maxIter=int(max_iter),
            implicitPrefs=bool(implicit_prefs),
            alpha=float(alpha),
            numUserBlocks=int(num_user_blocks),
            numItemBlocks=int(num_item_blocks),
            userCol='unique_user_id',
            itemCol='unique_item_id'
        )

        # load training data
        custom_schema = StructType([
            StructField('user', StringType(), True),
            StructField('item', StringType(), True),
            StructField('rating', FloatType(), True),
            StructField('unique_user_id', IntegerType(), True),
            StructField('unique_item_id', IntegerType(), True),
        ])
        df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(processed_training_data_file_path, schema=custom_schema)

        # fitting
        model = als.fit(df)

        # save
        ymd = datetime.today().strftime('%Y%m%d')
        saved_data_dir_path = model_dir_path \
            + 'als_model_' \
            + ymd
        model.write().overwrite().save(saved_data_dir_path)

        # copy directory
        copied_data_dir_path = model_dir_path + 'als_model'
        model.write().overwrite().save(copied_data_dir_path)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'processed_training_data_file_path',
        'model_dir_path',
        'rank',
        'max_iter',
        'implicit_prefs',
        'alpha',
        'num_user_blocks',
        'num_item_blocks'
    ]
    usage = 'Usage:\n' \
        + '    create_cf_model_script.py\n' \
        + '        --processed_training_data_file_path=<processed_training_data_file_path>\n' \
        + '        --model_dir_path=<model_dir_path>\n' \
        + '        --rank=<rank>\n' \
        + '        --max_iter=<max_iter>\n' \
        + '        --implicit_prefs=<implicit_prefs>\n' \
        + '        --alpha=<alpha>\n' \
        + '        --num_user_blocks=<num_user_blocks>\n' \
        + '        --num_item_blocks=<num_item_blocks>\n' \
        + '        [--env=<env>]\n' \
        + '    create_cf_model_script.py -h | --help'

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
    processed_training_data_file_path = parameters['processed_training_data_file_path']
    model_dir_path = parameters['model_dir_path']
    rank = parameters['rank']
    max_iter = parameters['max_iter']
    implicit_prefs = parameters['implicit_prefs']
    alpha = parameters['alpha']
    num_user_blocks = parameters['num_user_blocks']
    num_item_blocks = parameters['num_item_blocks']
    env = parameters['env']

    # execute
    ccm = CreateCfModel(env)
    ccm.run(
        processed_training_data_file_path,
        model_dir_path,
        rank,
        max_iter,
        implicit_prefs,
        alpha,
        num_user_blocks,
        num_item_blocks
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
