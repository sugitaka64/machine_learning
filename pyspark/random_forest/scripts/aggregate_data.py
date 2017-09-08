#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DODA plus, Prediction for Passing Paper exam.
processing data for prediction.
"""

from collections import defaultdict
from datetime import datetime
import os
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
import sys

class DpPPPError(Exception):
    """common error class."""

    pass

class JoinPass(object):
    """processing data for prediction."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        predicted_data_file_path: str,
        passed_paper_exam_data_file_path: str,
        send_scout_mail_data_file_path: str,
        aodrno_to_wjoid_data_file_path: str,
        applied_data_file_path: str,
        item_data_file_path: str
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isfile(predicted_data_file_path):
                raise DpPPPError('invalid parameter: predicted_data_file_path')
            if not os.path.isfile(passed_paper_exam_data_file_path):
                raise DpPPPError('invalid parameter: passed_paper_exam_data_file_path')
            if not os.path.isfile(send_scout_mail_data_file_path):
                raise DpPPPError('invalid parameter: send_scout_mail_data_file_path')
            if not os.path.isfile(aodrno_to_wjoid_data_file_path):
                raise DpPPPError('invalid parameter: aodrno_to_wjoid_data_file_path')
            if not os.path.isfile(applied_data_file_path):
                raise DpPPPError('invalid parameter: applied_data_file_path')
            if not os.path.isfile(item_data_file_path):
                raise DpPPPError('invalid parameter: item_data_file_path')

        return True

    def __column_check(
        self,
        val_1: str,
        val_2: str
    ) -> str:
        """passed check."""
        if (val_1 is not None) and (val_2 is not None):
            return '1'
        return '0'

    def __calc_distribution(
        self,
        df: DataFrame
    ) -> dict:
        """calculate distribution."""
        d = {}
        score_range = [
            0.00, 0.05, 0.10, 0.15, 0.20,
            0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60,
        ]
        for score in score_range:
            lower = '%.2f' % (score)
            if lower == '0.60':
                upper = 1.00
                index = str(lower) + ' - 1.00'
                d[index] = df.where(
                    (df['predicted_score'] >= lower) & (df['predicted_score'] <= upper)
                ).count()
            else:
                upper = '%.2f' % (score + 0.05)
                index = str(lower) + ' - ' + str(upper)
                d[index] = df.where(
                    (df['predicted_score'] >= lower) & (df['predicted_score'] < upper)
                ).count()

        return d

    def __make_area_cd(
            self,
            pref_cd: str
    ) -> str:
        """make pref code from prefecture code."""
        try:
            pref_cd = int(pref_cd)
            if pref_cd >= 1 and pref_cd <= 7:
                return '2'
            if pref_cd >= 8 and pref_cd <= 14:
                return '3'
            if pref_cd >= 21 and pref_cd <= 24:
                return '4'
            if pref_cd >= 15 and pref_cd <= 20:
                return '5'
            if pref_cd >= 25 and pref_cd <= 30:
                return '6'
            if pref_cd >= 31 and pref_cd <= 39:
                return '7'
            if pref_cd >= 40 and pref_cd <= 47:
                return '8'
            return '0'
        except ValueError:
            return '0'
        except TypeError:
            return '0'

    def run(
        self,
        predicted_data_file_path: str,
        passed_paper_exam_data_file_path: str,
        send_scout_mail_data_file_path: str,
        aodrno_to_wjoid_data_file_path: str,
        applied_data_file_path: str,
        item_data_file_path: str
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            predicted_data_file_path,
            passed_paper_exam_data_file_path,
            send_scout_mail_data_file_path,
            aodrno_to_wjoid_data_file_path,
            applied_data_file_path,
            item_data_file_path
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('aggregate_data')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # load predicted data
        custom_schema = StructType([
            StructField('predicted_wjoid', StringType(), True),
            StructField('predicted_wsfid', StringType(), True),
            StructField('predicted_score', FloatType(), True),
            StructField('predicted_aodrno', StringType(), True),
            StructField('predicted_wsfno', StringType(), True)
        ])
        predicted_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(predicted_data_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)

        # load aodrno to wjoid mapping data
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
        df = predicted_df.join(
            mapping_df,
            predicted_df['predicted_aodrno'] == mapping_df['mapping_aodrno'],
            'left_outer'
        )
        # delete unnecessary variables
        del(predicted_df)
        del(mapping_df)

        # load passed data
        custom_schema = StructType([
            StructField('passed_astfid', StringType(), True),
            StructField('passed_aodrid', StringType(), True)
        ])
        passed_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(passed_paper_exam_data_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)

        # join
        df = df.join(
            passed_df,
            (
                (df['mapping_aodrid'] == passed_df['passed_aodrid'])
                & (df['predicted_wsfid'] == passed_df['passed_astfid'])
            ),
            'left_outer'
        )
        # delete unnecessary variables
        del(passed_df)

        # load send scout mail data
        custom_schema = StructType([
            StructField('scout_mail_wsfid', StringType(), True),
            StructField('scout_mail_wjoid', StringType(), True)
        ])
        scout_mail_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(send_scout_mail_data_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)
        # join
        df = df.join(
            scout_mail_df,
            (
                (df['mapping_wjoid'] == scout_mail_df['scout_mail_wjoid'])
                & (df['predicted_wsfid'] == scout_mail_df['scout_mail_wsfid'])
            ),
            'left_outer'
        )
        # delete unnecessary variables
        del(scout_mail_df)

        # load applied data
        custom_schema = StructType([
            StructField('applied_astfid', StringType(), True),
            StructField('applied_aodrid', StringType(), True)
        ])
        applied_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(applied_data_file_path, schema=custom_schema)\
            .na\
            .fill(0.0)
        # join
        df = df.join(
            applied_df,
            (
                (df['mapping_aodrid'] == applied_df['applied_aodrid'])
                & (df['predicted_wsfid'] == applied_df['applied_astfid'])
            ),
            'left_outer'
        )
        # delete unnecessary variables
        del(applied_df)

        # load item data
        items_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(item_data_file_path)\
            .na\
            .fill(0.0)
        # join
        df = df.join(
            items_df,
            df['predicted_aodrno'] == items_df['i_aodrno'],
            'left_outer'
        )
        # delete unnecessary variables
        del(items_df)

        df.cache()

        # make udf
        f_column_check = udf(lambda l_1, l_2: self.__column_check(l_1, l_2), StringType())
        f_make_area_cd = udf(lambda l: self.__make_area_cd(l), StringType())

        # passed check
        df = df.withColumn(
            'passed_paper_exam',
            f_column_check(df['passed_astfid'], df['passed_aodrid'])
        )
        df = df.withColumn(
            'send_scout_mail',
            f_column_check(df['scout_mail_wsfid'], df['scout_mail_wjoid'])
        )
        df = df.withColumn(
            'applied',
            f_column_check(df['applied_astfid'], df['applied_aodrid'])
        )

        # area code
        df = df.withColumn('area_cd', f_make_area_cd(df['i_atdfkid']))

        df.cache()

        # aggregate

        # all data
        data = {}
        d = self.__calc_distribution(df)
        for k in d.keys():
            data[k] = [d[k]]

        # send scout mail
        tmp_df = df.where(df['send_scout_mail'] == '1')
        d = self.__calc_distribution(tmp_df)
        for k in d.keys():
            data[k].append(d[k])

        # send scout mail -> applied
        tmp_df = df.where(
            (df['send_scout_mail'] == '1') & (df['applied'] == '1')
        )
        d = self.__calc_distribution(tmp_df)
        for k in d.keys():
            data[k].append(d[k])

        # send scout mail -> passed paper exam
        tmp_df = df.where(
            (df['send_scout_mail'] == '1') & (df['passed_paper_exam'] == '1')
        )
        d = self.__calc_distribution(tmp_df)
        for k in d.keys():
            data[k].append(d[k])

        # not send scout mail -> applied
        tmp_df = df.where(
            (df['send_scout_mail'] != '1') & (df['applied'] == '1')
        )
        d = self.__calc_distribution(tmp_df)
        for k in d.keys():
            data[k].append(d[k])

        # not send scout mail -> passed paper exam
        tmp_df = df.where(
            (df['send_scout_mail'] != '1') & (df['passed_paper_exam'] == '1')
        )
        d = self.__calc_distribution(tmp_df)
        for k in d.keys():
            data[k].append(d[k])

        print('all')
        for k, v in sorted(data.items()):
            if int(v[2]) == 0:
                div_1 = 0
            else:
                div_1 = int(v[3]) / int(v[2])
            if int(v[4]) == 0:
                div_2 = 0
            else:
                div_2 = int(v[5]) / int(v[4])
            t = [v[0], v[1], v[2], v[3], div_1, v[4], v[5], div_2]

            print('%s\t%s' % (k, '\t'.join(map(str, t))))

        # count by each item
        for column in ['i_assyulid', 'area_cd']:
            print('')
            print(column)

            if column == 'i_assyulid':
                master = ['%02d' % (n) for n in range(1, 16)]
            elif column == 'area_cd':
                master = ['2', '3', '4', '5', '6', '7', '8', '0']
            else:
                master = []

            data = defaultdict(list)
            for m in master:
                tmp_data = {}
                column_df = df.where(df[column] == m)

                # send scout mail -> applied
                tmp_df = column_df.where(
                    (column_df['send_scout_mail'] == '1') & (column_df['applied'] == '1')
                )
                d = self.__calc_distribution(tmp_df)
                for k in d.keys():
                    tmp_data[k] = [d[k]]

                # send scout mail -> passed paper exam
                tmp_df = column_df.where(
                    (column_df['send_scout_mail'] == '1') & (column_df['passed_paper_exam'] == '1')
                )
                d = self.__calc_distribution(tmp_df)
                for k in d.keys():
                    tmp_data[k].append(d[k])

                # not send scout mail -> applied
                tmp_df = column_df.where(
                    (column_df['send_scout_mail'] != '1') & (column_df['applied'] == '1')
                )
                d = self.__calc_distribution(tmp_df)
                for k in d.keys():
                    tmp_data[k].append(d[k])

                # not send scout mail -> passed paper exam
                tmp_df = column_df.where(
                    (column_df['send_scout_mail'] != '1') & (column_df['passed_paper_exam'] == '1')
                )
                d = self.__calc_distribution(tmp_df)
                for k in d.keys():
                    tmp_data[k].append(d[k])

                for k, v in tmp_data.items():
                    if int(v[0]) == 0:
                        div_1 = 0
                    else:
                        div_1 = int(v[1]) / int(v[0])
                    if int(v[2]) == 0:
                        div_2 = 0
                    else:
                        div_2 = int(v[3]) / int(v[2])

                    data[k].append(v[0])
                    data[k].append(v[1])
                    data[k].append(div_1)
                    data[k].append(v[2])
                    data[k].append(v[3])
                    data[k].append(div_2)

            for k, v in sorted(data.items()):
                print('%s\t%s' % (k, '\t'.join(map(str, v))))

        # delete unnecessary variables
        del(column_df)

        # job with experience
        data = defaultdict(list)
        i = 1
        while i <= 2:
            if i == 1:
                must_assyulid_df = df.where(df['i_must_assyulid'].isNotNull())
            elif i == 2:
                must_assyulid_df = df.where(df['i_must_assyulid'].isNull())
            else:
                break

            # send scout mail -> applied
            tmp_df = must_assyulid_df.where(
                (must_assyulid_df['send_scout_mail'] == '1')
                & (must_assyulid_df['applied'] == '1')
            )
            d = self.__calc_distribution(tmp_df)
            for k in d.keys():
                data[k].append(d[k])

            # send scout mail -> passed paper exam
            tmp_df = must_assyulid_df.where(
                (must_assyulid_df['send_scout_mail'] == '1')
                & (must_assyulid_df['passed_paper_exam'] == '1')
            )
            d = self.__calc_distribution(tmp_df)
            for k in d.keys():
                data[k].append(d[k])

            # not send scout mail -> applied
            tmp_df = must_assyulid_df.where(
                (must_assyulid_df['send_scout_mail'] != '1')
                & (must_assyulid_df['applied'] == '1')
            )
            d = self.__calc_distribution(tmp_df)
            for k in d.keys():
                data[k].append(d[k])

            # not send scout mail -> passed paper exam
            tmp_df = must_assyulid_df.where(
                (must_assyulid_df['send_scout_mail'] != '1')
                & (must_assyulid_df['passed_paper_exam'] == '1')
            )
            d = self.__calc_distribution(tmp_df)
            for k in d.keys():
                data[k].append(d[k])

            i += 1

        print('')
        print('job with experience')
        for k, v in sorted(data.items()):
            if int(v[0]) == 0:
                div_1 = 0
            else:
                div_1 = int(v[1]) / int(v[0])
            if int(v[2]) == 0:
                div_2 = 0
            else:
                div_2 = int(v[3]) / int(v[2])

            if int(v[4]) == 0:
                div_3 = 0
            else:
                div_3 = int(v[5]) / int(v[4])
            if int(v[6]) == 0:
                div_4 = 0
            else:
                div_4 = int(v[7]) / int(v[6])

            t = [v[0], v[1], div_1, v[2], v[3], div_2, v[4], v[5], div_3, v[6], v[7], div_4]

            print('%s\t%s' % (k, '\t'.join(map(str, t))))

        # delete unnecessary variables
        del(must_assyulid_df)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'predicted_data_file_path',
        'passed_paper_exam_data_file_path',
        'send_scout_mail_data_file_path',
        'aodrno_to_wjoid_data_file_path',
        'applied_data_file_path',
        'item_data_file_path'
    ]
    usage = 'Usage:\n' \
        + '    aggregate_data.py\n' \
        + '        --predicted_data_file_path=<predicted_data_file_path>\n' \
        + '        --passed_paper_exam_data_file_path=<passed_paper_exam_data_file_path>\n' \
        + '        --send_scout_mail_data_file_path=<send_scout_mail_data_file_path>\n' \
        + '        --aodrno_to_wjoid_data_file_path=<aodrno_to_wjoid_data_file_path>\n' \
        + '        --applied_data_file_path=<applied_data_file_path>\n' \
        + '        --item_data_file_path=<item_data_file_path>\n' \
        + '        [--env=<env>]\n' \
        + '    aggregate_data.py -h | --help'

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
    predicted_data_file_path = parameters['predicted_data_file_path']
    passed_paper_exam_data_file_path = parameters['passed_paper_exam_data_file_path']
    send_scout_mail_data_file_path = parameters['send_scout_mail_data_file_path']
    aodrno_to_wjoid_data_file_path = parameters['aodrno_to_wjoid_data_file_path']
    applied_data_file_path = parameters['applied_data_file_path']
    item_data_file_path = parameters['item_data_file_path']
    env = parameters['env']

    # execute
    jp = JoinPass(env)
    jp.run(
        predicted_data_file_path,
        passed_paper_exam_data_file_path,
        send_scout_mail_data_file_path,
        aodrno_to_wjoid_data_file_path,
        applied_data_file_path,
        item_data_file_path
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
