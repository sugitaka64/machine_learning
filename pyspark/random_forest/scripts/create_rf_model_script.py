#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DODA plus, Prediction for Passing Paper exam.
create random forest model.
"""

from datetime import datetime
import os
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
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

class CreateRfModel(object):
    """create random forest model."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        processed_training_data_file_path: str,
        model_dir_path: str
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isdir(processed_training_data_file_path):
                raise DpPPPError('invalid parameter: processed_training_data_file_path')
            if not os.path.isdir(model_dir_path):
                raise DpPPPError('invalid parameter: model_dir_path')

        return True

    def run(
        self,
        processed_training_data_file_path: str,
        model_dir_path: str
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            processed_training_data_file_path,
            model_dir_path
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('create_rf_model')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # load training data
        custom_schema = StructType([
            StructField('id', IntegerType(), True),
            StructField('i_acrid', StringType(), True),
            StructField('i_aodrid', StringType(), True),
            StructField('i_aodrno', StringType(), True),
            StructField('u_wsfno', StringType(), True),
            StructField('u_wsfid', StringType(), True),
            StructField('tsuka', IntegerType(), True),
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
            .load(processed_training_data_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)

        # drop columns
        df = df.drop(
            'id',
            'i_acrid',
            'i_aodrid',
            'i_aodrno',
            'u_wsfno',
            'u_wsfid'
        )

        # create vector from dataframe
        features = df.drop('tsuka')
        vector_assembler = VectorAssembler(inputCols=features.columns, outputCol='features')
        tf = vector_assembler.transform(df)

        # index labels, adding metadata to the label column
        string_indexer = StringIndexer(inputCol='tsuka', outputCol='indexed')
        si_model = string_indexer.fit(tf)
        td = si_model.transform(tf)

        # divide training and test data
        (training_data, test_data) = td.randomSplit([0.8, 0.2])

        # create model
        rf = RandomForestClassifier(labelCol='indexed')
        evaluator = MulticlassClassificationEvaluator(
            labelCol='indexed',
            predictionCol='prediction',
            metricName='weightedPrecision'
        )

        # cross validation
        param_grid = ParamGridBuilder()\
            .addGrid(rf.numTrees, [10, 50, 100])\
            .addGrid(rf.featureSubsetStrategy, ['auto'])\
            .addGrid(rf.impurity, ['entropy', 'gini'])\
            .addGrid(rf.maxDepth, [4, 6, 8, 10])\
            .addGrid(rf.maxBins, [32, 64, 128])\
            .build()
        cross_val = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=2
        )

        # compute test error
        model = cross_val.fit(training_data).bestModel
        predictions = model.transform(test_data)
        accuracy = evaluator.evaluate(predictions)
        print('Test Success: %f' % (accuracy))
        print('Test Error  : %f' % (1.0 - accuracy))

        # feature importances
        print('Feature Importances:')
        features_column_names = features.schema.names
        feature_importances = model.featureImportances
        scores = {}
        for i, v in enumerate(features_column_names):
            scores[v] = feature_importances[i]
        for k, v in sorted(scores.items(), reverse=True, key=lambda x: x[1]):
            print('  %s: %f' % (k, v))

        # save model
        saved_data_dir_path = model_dir_path + 'rf_model'
        model.write().overwrite().save(saved_data_dir_path)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'processed_training_data_file_path',
        'model_dir_path'
    ]
    usage = 'Usage:\n' \
        + '    create_rf_model_script.py\n' \
        + '        --processed_training_data_file_path=<processed_training_data_file_path>\n' \
        + '        --model_dir_path=<model_dir_path>\n' \
        + '        [--env=<env>]\n' \
        + '    create_rf_model_script.py -h | --help'

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
    env = parameters['env']

    # execute
    crm = CreateRfModel(env)
    crm.run(
        processed_training_data_file_path,
        model_dir_path
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
