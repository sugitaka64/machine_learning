#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DODA plus, Prediction for Passing Paper exam.
predict score from random forest model.
"""

from datetime import datetime
import os
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
import sys

class DpPPPError(Exception):
    """common error class."""

    pass

class CreatePredictedScore(object):
    """predict score from random forest model."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        model_file_path: str,
        predict_data_dir_path: str,
        data_for_prediction_file_path: str,
        aodrno_to_wjoid_data_file_path: str,
        lower_score: float=0.3
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isdir(model_file_path):
                raise DpPPPError('invalid parameter: model_file_path')
            if not os.path.isdir(predict_data_dir_path):
                raise DpPPPError('invalid parameter: predict_data_dir_path')
            if not os.path.isdir(data_for_prediction_file_path):
                raise DpPPPError('invalid parameter: data_for_prediction_file_path')
            if not os.path.isfile(aodrno_to_wjoid_data_file_path):
                raise DpPPPError('invalid parameter: aodrno_to_wjoid_data_file_path')

        try:
            lower_score = float(lower_score)
        except ValueError:
            raise DpPPPError('invalid parameter: lower_score')
        except TypeError:
            raise DpPPPError('invalid parameter: lower_score')

        return True

    def run(
        self,
        model_file_path: str,
        predict_data_dir_path: str,
        data_for_prediction_file_path: str,
        aodrno_to_wjoid_data_file_path: str,
        lower_score: float=0.3
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            model_file_path,
            predict_data_dir_path,
            data_for_prediction_file_path,
            aodrno_to_wjoid_data_file_path,
            lower_score
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('create_predicted_score')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # load data for prediction
        custom_schema = StructType([
            StructField('id', IntegerType(), True),
            StructField('i_acrid', StringType(), True),
            StructField('i_aodrid', StringType(), True),
            StructField('i_aodrno', StringType(), True),
            StructField('u_wsfno', StringType(), True),
            StructField('u_wsfid', StringType(), True),
            StructField('i_abestagefr', FloatType(), True),
            StructField('i_abestageto', FloatType(), True),
            StructField('i_prefcat', FloatType(), True),
            StructField('i_ayticmfr', FloatType(), True),
            StructField('i_ayticmto', FloatType(), True),
            StructField('i_ayticmirfr', FloatType(), True),
            StructField('i_ayticmirto', FloatType(), True),
            StructField('i_aysbf', FloatType(), True),
            StructField('i_ayhoc', FloatType(), True),
            StructField('i_aavgothrfl', FloatType(), True),
            StructField('i_aebdy', FloatType(), True),
            StructField('i_asihon', FloatType(), True),
            StructField('i_asl1', FloatType(), True),
            StructField('i_akjorv1', FloatType(), True),
            StructField('i_aempcnt', FloatType(), True),
            StructField('i_aaveage', FloatType(), True),
            StructField('i_amrate', FloatType(), True),
            StructField('i_asijoid', FloatType(), True),
            StructField('i_amusagefr', FloatType(), True),
            StructField('i_amusageto', FloatType(), True),
            StructField('i_amustsykcnt', FloatType(), True),
            StructField('i_assyulid', IntegerType(), True),
            StructField('u_wbrdt', FloatType(), True),
            StructField('u_wpfid', IntegerType(), True),
            StructField('u_wlaid', IntegerType(), True),
            StructField('u_wbrfl', FloatType(), True),
            StructField('u_wjobc', FloatType(), True),
            StructField('u_wlmpdt', FloatType(), True),
            StructField('u_widlk1', FloatType(), True),
            StructField('u_woclk1', FloatType(), True),
            StructField('u_whpai', FloatType(), True),
            StructField('u_wmiai', FloatType(), True),
            StructField('u_wecnt', FloatType(), True),
            StructField('u_wmkid', FloatType(), True),
            StructField('u_widlk', FloatType(), True),
            StructField('u_wincm', FloatType(), True),
            StructField('u_wscnm', FloatType(), True),
            StructField('A_1', IntegerType(), True),
            StructField('A_2', IntegerType(), True),
            StructField('A_3', IntegerType(), True),
            StructField('A_4', IntegerType(), True),
            StructField('A_5', IntegerType(), True),
            StructField('D_3', IntegerType(), True),
            StructField('D_4', IntegerType(), True),
            StructField('D_5', IntegerType(), True),
            StructField('D_6', IntegerType(), True),
            StructField('D_7', IntegerType(), True),
            StructField('D_8', IntegerType(), True),
            StructField('D_9', IntegerType(), True),
            StructField('D_10', IntegerType(), True),
            StructField('D_11', IntegerType(), True),
            StructField('G_1', IntegerType(), True),
            StructField('G_2', IntegerType(), True),
            StructField('G_3', IntegerType(), True),
            StructField('G_4', IntegerType(), True),
            StructField('G_5', IntegerType(), True),
            StructField('G_6', IntegerType(), True),
            StructField('G_7', IntegerType(), True),
            StructField('G_8', IntegerType(), True),
            StructField('H_1', IntegerType(), True)
        ])
        df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(data_for_prediction_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)

        # create vector from dataframe
        features = df.drop(
            'id',
            'i_acrid',
            'i_aodrid',
            'i_aodrno',
            'u_wsfno',
            'u_wsfid'
        )
        vector_assembler = VectorAssembler(inputCols=features.columns, outputCol='features')
        tf = vector_assembler.transform(df)

        # load model
        model = RandomForestClassificationModel.load(model_file_path)

        # predict score
        predictions = model.transform(tf)
        predictions = predictions.select(
            'id',
            'probability'
        )
        predictions = predictions.rdd.map(lambda arr: Row(
            id=arr[0],
            probability=float(arr[1][1]),
        ))
        probability = 'probability >= %f' % (float(lower_score))
        predictions_df = sqlContext.createDataFrame(predictions).filter(probability)
        df = df.join(
            predictions_df,
            df['id'] == predictions_df['id'],
            'inner'
        ).drop(predictions_df['id'])

        # load aodrid to wjoid mapping data
        custom_schema = StructType([
            StructField('mapping_aodrno', StringType(), True),
            StructField('mapping_aodrid', StringType(), True),
            StructField('mapping_wjoid', StringType(), True)
        ])
        mapping_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(aodrno_to_wjoid_data_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)
        # join
        df = df.join(
            mapping_df,
            df['i_aodrid'] == mapping_df['mapping_aodrid'],
            'inner'
        ).drop(mapping_df['mapping_aodrid'])

        # delete unnecessary variables
        del(features)
        del(tf)
        del(predictions_df)
        del(mapping_df)

        # save for mail system
        df.registerTempTable('predictions')
        sql = 'SELECT ' \
            + '  mapping_wjoid AS wjoid, ' \
            + '  u_wsfid AS wsfid, ' \
            + '  probability AS score, ' \
            + '  i_aodrno AS aodrno, ' \
            + '  u_wsfno AS wsfno ' \
            + 'FROM predictions ' \
            + 'ORDER BY i_aodrno, probability DESC'
        predict_data = sqlContext.sql(sql)

        saved_data_file_path = predict_data_dir_path + 'predicts_for_sys.csv'
        predict_data.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(saved_data_file_path)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'model_file_path',
        'predict_data_dir_path',
        'data_for_prediction_file_path',
        'aodrno_to_wjoid_data_file_path',
    ]
    usage = 'Usage:\n' \
        + '    create_predicted_score_script.py\n' \
        + '        --model_file_path=<model_file_path>\n' \
        + '        --predict_data_dir_path=<predict_data_dir_path>\n' \
        + '        --data_for_prediction_file_path=<data_for_prediction_file_path>\n' \
        + '        --aodrno_to_wjoid_data_file_path=<aodrno_to_wjoid_data_file_path>\n' \
        + '        [--lower_score=<lower_score>]\n' \
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
    ret['lower_score'] = 0.3
    ret['env'] = 'local'
    for actual_option in actual_options:
        if '--lower_score=' in actual_option:
            value = actual_option.split('=')[1]
            if value:
                ret['lower_score'] = value
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
    model_file_path = parameters['model_file_path']
    predict_data_dir_path = parameters['predict_data_dir_path']
    data_for_prediction_file_path = parameters['data_for_prediction_file_path']
    aodrno_to_wjoid_data_file_path = parameters['aodrno_to_wjoid_data_file_path']
    lower_score = parameters['lower_score']
    env = parameters['env']

    # execute
    cps = CreatePredictedScore(env)
    cps.run(
        model_file_path,
        predict_data_dir_path,
        data_for_prediction_file_path,
        aodrno_to_wjoid_data_file_path,
        lower_score
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
