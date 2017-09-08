#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DODA plus, Prediction for Passing Paper exam.
processing training data.
"""

from datetime import datetime
import os
from pyspark.sql.functions import udf
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType
import sys

class DpPPPError(Exception):
    """common error class."""

    pass

class ProcessTrainingData(object):
    """get training data from csv, and process data."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        unprocessed_data_file_path: str,
        training_data_dir_path: str,
        school_master_file_path: str
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isfile(unprocessed_data_file_path):
                raise DpPPPError('invalid parameter: unprocessed_data_file_path')
            if not os.path.isdir(training_data_dir_path):
                raise DpPPPError('invalid parameter: training_data_dir_path')
            if not os.path.isfile(school_master_file_path):
                raise DpPPPError('invalid parameter: school_master_file_path')

        return True

    def __convert_pref_cd(
            self,
            pref_cd: str
    ) -> str:
        """convert prefecture code."""
        try:
            pref_cd = int(pref_cd)
            if pref_cd >= 1 and pref_cd <= 7:
                return '1'
            if pref_cd >= 8 and pref_cd <= 10:
                return '2'
            if pref_cd >= 15 and pref_cd <= 20:
                return '3'
            if pref_cd >= 31 and pref_cd <= 39:
                return '5'
            if pref_cd >= 41 and pref_cd <= 47:
                return '6'
            if pref_cd in (24, 25, 29, 30):
                return '4'
            if pref_cd in (11, 12, 13, 14, 21, 22, 23, 26, 27, 28, 40, 99):
                return str(pref_cd)
            else:
                return '0'
        except ValueError:
            return '0'
        except TypeError:
            return '0'

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

    def __convert_date_into_serial(
        self,
        date: str
    ) -> str:
        """convert date into serial."""
        try:
            dt = datetime.strptime(date, '%Y/%m/%d') - datetime(1899, 12, 31)
            return str(dt.days + 1)
        except ValueError:
            try:
                dt = datetime.strptime(date, '%Y/%m/%d %H:%M:%S') - datetime(1899, 12, 31)
                return str(dt.days + 1)
            except ValueError:
                return '0'
        except TypeError:
            return '0'

    def __fill_zero(
            self,
            val: str
    ) -> str:
        """fill missing value with 0."""
        if val is None:
            return '0'
        return val

    def __match_or_not(
            self,
            val_1: float,
            val_2: float
    ) -> str:
        """match."""
        if (val_1 is None) or (val_2 is None) or (val_1 == 0.0) or (val_2 == 0.0):
            return '0'
        if val_1 == val_2:
            return '2'
        return '1'

    def __get_rate_of_annual_salary(
            self,
            val_1: str,
            val_2: str
    ) -> str:
        """get rate of annual_salary."""
        try:
            diff = float(val_1) - float(val_2)
            if diff <= -1000000.0:
                return '1'
            if diff <= -500000.0:
                return '2'
            if diff <= 0.0:
                return '3'
            if diff <= 500000.0:
                return '4'
            if diff <= 1000000.0:
                return '5'
            if diff > 1000000.0:
                return '6'
            return '0'
        except ValueError:
            return '0'
        except TypeError:
            return '0'

    def run(
        self,
        unprocessed_data_file_path: str,
        training_data_dir_path: str,
        school_master_file_path: str
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            unprocessed_data_file_path,
            training_data_dir_path,
            school_master_file_path
        )

        # make spark context
        spark = SparkSession\
            .builder\
            .appName('process_training_data')\
            .config('spark.sql.crossJoin.enabled', 'true')\
            .config('spark.debug.maxToStringFields', 500)\
            .getOrCreate()
        sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

        # load item and user data
        raw_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(unprocessed_data_file_path)

        # make udf
        f_make_area_cd \
            = udf(lambda l: self.__make_area_cd(l), StringType())
        # make area code
        raw_df = raw_df.withColumn('i_area_cd', f_make_area_cd(raw_df['i_atdfkid']))

        # extraction condition
        raw_df = raw_df.where(
            # experience business type
            (
                (raw_df.i_assyulid.isNotNull())
                & (raw_df.i_assyulid == raw_df.u_experience_assyulid)
            )
            # desired business type
            & (
                (raw_df.i_assyulid.isNotNull())
                & (
                    (raw_df.i_assyulid == raw_df.u_desired_assyulid1)
                    | (raw_df.i_assyulid == raw_df.u_desired_assyulid2)
                    | (raw_df.i_assyulid == raw_df.u_desired_assyulid3)
                )
            )
            # area
            & (
                (raw_df.i_atdfkid.isNotNull())
                & (
                    (raw_df.i_atdfkid == raw_df.u_atdfkid1)
                    | (raw_df.i_atdfkid == raw_df.u_atdfkid2)
                    | (raw_df.i_atdfkid == raw_df.u_atdfkid3)
                    | (raw_df.u_aaraid1 == '1')
                    | (raw_df.u_aaraid2 == '1')
                    | (raw_df.u_aaraid3 == '1')
                    | (raw_df.i_area_cd == raw_df.u_aaraid1)
                    | (raw_df.i_area_cd == raw_df.u_aaraid2)
                    | (raw_df.i_area_cd == raw_df.u_aaraid3)
                )
            )
        )

        # load school master
        school_df = sqlContext\
            .read\
            .format('csv')\
            .options(header='true')\
            .load(school_master_file_path)
        school_df = school_df.dropDuplicates(['wscnm'])

        # cross join school data
        df = raw_df.join(
            school_df,
            raw_df['u_wscnm'] == school_df['wscnm'],
            'left_outer'
        ).drop(school_df['wscnm'])

        # delete unnecessary variables
        del(raw_df)
        del(school_df)

        # make udf
        f_convert_pref_cd \
            = udf(lambda l: self.__convert_pref_cd(l), StringType())
        f_convert_date_into_serial \
            = udf(lambda l: self.__convert_date_into_serial(l), StringType())
        f_fill_zero = udf(lambda l: self.__fill_zero(l), StringType())
        f_match_or_not = udf(lambda l_1, l_2: self.__match_or_not(l_1, l_2), StringType())
        f_get_rate_of_annual_salary \
            = udf(lambda l_1, l_2: self.__get_rate_of_annual_salary(l_1, l_2), StringType())

        # convert pref code
        df = df.withColumn('i_prefcat', f_convert_pref_cd(df['i_atdfkid']))
        df = df.withColumn('u_wpfid', f_convert_pref_cd(df['u_wpfid']))

        # convert date
        df = df.withColumn('u_wbrdt', f_convert_date_into_serial(df['u_wbrdt']))
        df = df.withColumn('u_wlmpdt', f_convert_date_into_serial(df['u_wlmpdt']))
        df = df.withColumn('i_aebdy', f_convert_date_into_serial(df['i_aebdy']))

        # fill missing value with 0
        df = df.withColumn('u_wscnm', f_fill_zero(df['hensachi']))

        # whether match or not
        df = df.withColumn('A_1', f_match_or_not(df['u_widlk'], df['u_widlk1']))
        df = df.withColumn('A_2', f_match_or_not(df['u_widsk'], df['u_widsk1']))
        df = df.withColumn('A_3', f_match_or_not(df['u_woclk'], df['u_woclk1']))
        df = df.withColumn('A_4', f_match_or_not(df['u_wocmk'], df['u_wocmk1']))
        df = df.withColumn('A_5', f_match_or_not(df['u_wocsk'], df['u_wocsk1']))
        df = df.withColumn('D_3', f_match_or_not(df['i_assyulid'], df['u_woclk1']))
        df = df.withColumn('D_4', f_match_or_not(df['i_assyumid'], df['u_wocmk1']))
        df = df.withColumn('D_5', f_match_or_not(df['i_assyusid'], df['u_wocsk1']))
        df = df.withColumn('D_6', f_match_or_not(df['i_assyulid'], df['u_woclk2']))
        df = df.withColumn('D_7', f_match_or_not(df['i_assyumid'], df['u_wocmk2']))
        df = df.withColumn('D_8', f_match_or_not(df['i_assyusid'], df['u_wocsk2']))
        df = df.withColumn('D_9', f_match_or_not(df['i_assyulid'], df['u_woclk3']))
        df = df.withColumn('D_10', f_match_or_not(df['i_assyumid'], df['u_wocmk3']))
        df = df.withColumn('D_11', f_match_or_not(df['i_assyusid'], df['u_wocsk3']))
        df = df.withColumn('H_1', f_match_or_not(df['i_prefcat'], df['u_wpfid']))

        # diff annual salary
        df = df.withColumn('G_1', f_get_rate_of_annual_salary(df['u_whpai'], df['u_wincm']))
        df = df.withColumn('G_2', f_get_rate_of_annual_salary(df['u_wmiai'], df['u_wincm']))
        df = df.withColumn('G_3', f_get_rate_of_annual_salary(df['u_whpai'], df['i_ayticmfr']))
        df = df.withColumn('G_4', f_get_rate_of_annual_salary(df['u_whpai'], df['i_ayticmto']))
        df = df.withColumn('G_5', f_get_rate_of_annual_salary(df['u_wmiai'], df['i_ayticmfr']))
        df = df.withColumn('G_6', f_get_rate_of_annual_salary(df['u_wmiai'], df['i_ayticmto']))
        df = df.withColumn('G_7', f_get_rate_of_annual_salary(df['u_wincm'], df['i_ayticmfr']))
        df = df.withColumn('G_8', f_get_rate_of_annual_salary(df['u_wincm'], df['i_ayticmto']))

        # add index
        dr = df.rdd.zipWithIndex()

        # regenerate dataframe
        regenerate = dr.map(lambda arr: Row(
            id=arr[1],
            i_acrid=arr[0]['i_acrid'],
            i_aodrid=arr[0]['i_aodrid'],
            i_aodrno=arr[0]['i_aodrno'],
            u_wsfno=arr[0]['u_wsfno'],
            u_wsfid=arr[0]['u_wsfid'],
            tsuka=arr[0]['tsuka'],
            i_abestagefr=arr[0]['i_abestagefr'],
            i_abestageto=arr[0]['i_abestageto'],
            i_prefcat=arr[0]['i_prefcat'],
            i_ayticmfr=arr[0]['i_ayticmfr'],
            i_ayticmto=arr[0]['i_ayticmto'],
            i_ayticmirfr=arr[0]['i_ayticmirfr'],
            i_ayticmirto=arr[0]['i_ayticmirto'],
            i_aysbf=arr[0]['i_aysbf'],
            i_ayhoc=arr[0]['i_ayhoc'],
            i_aavgothrfl=arr[0]['i_aavgothrfl'],
            i_aebdy=arr[0]['i_aebdy'],
            i_asihon=arr[0]['i_asihon'],
            i_asl1=arr[0]['i_asl1'],
            i_akjorv1=arr[0]['i_akjorv1'],
            i_aempcnt=arr[0]['i_aempcnt'],
            i_aaveage=arr[0]['i_aaveage'],
            i_amrate=arr[0]['i_amrate'],
            i_asijoid=arr[0]['i_asijoid'],
            i_amusagefr=arr[0]['i_amusagefr'],
            i_amusageto=arr[0]['i_amusageto'],
            i_amustsykcnt=arr[0]['i_amustsykcnt'],
            i_assyulid=arr[0]['i_assyulid'],
            u_wbrdt=arr[0]['u_wbrdt'],
            u_wpfid=arr[0]['u_wpfid'],
            u_wlaid=arr[0]['u_wlaid'],
            u_wbrfl=arr[0]['u_wbrfl'],
            u_wjobc=arr[0]['u_wjobc'],
            u_wlmpdt=arr[0]['u_wlmpdt'],
            u_widlk1=arr[0]['u_widlk1'],
            u_woclk1=arr[0]['u_woclk1'],
            u_whpai=arr[0]['u_whpai'],
            u_wmiai=arr[0]['u_wmiai'],
            u_wecnt=arr[0]['u_wecnt'],
            u_wmkid=arr[0]['u_wmkid'],
            u_widlk=arr[0]['u_widlk'],
            u_wincm=arr[0]['u_wincm'],
            u_wscnm=arr[0]['u_wscnm'],
            A_1=arr[0]['A_1'],
            A_2=arr[0]['A_2'],
            A_3=arr[0]['A_3'],
            A_4=arr[0]['A_4'],
            A_5=arr[0]['A_5'],
            D_3=arr[0]['D_3'],
            D_4=arr[0]['D_4'],
            D_5=arr[0]['D_5'],
            D_6=arr[0]['D_6'],
            D_7=arr[0]['D_7'],
            D_8=arr[0]['D_8'],
            D_9=arr[0]['D_9'],
            D_10=arr[0]['D_10'],
            D_11=arr[0]['D_11'],
            G_1=arr[0]['G_1'],
            G_2=arr[0]['G_2'],
            G_3=arr[0]['G_3'],
            G_4=arr[0]['G_4'],
            G_5=arr[0]['G_5'],
            G_6=arr[0]['G_6'],
            G_7=arr[0]['G_7'],
            G_8=arr[0]['G_8'],
            H_1=arr[0]['H_1'],
        ))
        del(dr)

        # select columns and sort
        df = sqlContext.createDataFrame(regenerate, samplingRatio=0.2)
        df = df.select(
            'id',
            'i_acrid',
            'i_aodrid',
            'i_aodrno',
            'u_wsfno',
            'u_wsfid',
            'tsuka',
            'i_abestagefr',
            'i_abestageto',
            'i_prefcat',
            'i_ayticmfr',
            'i_ayticmto',
            'i_ayticmirfr',
            'i_ayticmirto',
            'i_aysbf',
            'i_ayhoc',
            'i_aavgothrfl',
            'i_aebdy',
            'i_asihon',
            'i_asl1',
            'i_akjorv1',
            'i_aempcnt',
            'i_aaveage',
            'i_amrate',
            'i_asijoid',
            'i_amusagefr',
            'i_amusageto',
            'i_amustsykcnt',
            'i_assyulid',
            'u_wbrdt',
            'u_wpfid',
            'u_wlaid',
            'u_wbrfl',
            'u_wjobc',
            'u_wlmpdt',
            'u_widlk1',
            'u_woclk1',
            'u_whpai',
            'u_wmiai',
            'u_wecnt',
            'u_wmkid',
            'u_widlk',
            'u_wincm',
            'u_wscnm',
            'A_1',
            'A_2',
            'A_3',
            'A_4',
            'A_5',
            'D_3',
            'D_4',
            'D_5',
            'D_6',
            'D_7',
            'D_8',
            'D_9',
            'D_10',
            'D_11',
            'G_1',
            'G_2',
            'G_3',
            'G_4',
            'G_5',
            'G_6',
            'G_7',
            'G_8',
            'H_1'
        )

        # save
        saved_data_file_path = training_data_dir_path + 'training_data.csv'
        df.write\
            .format('csv')\
            .mode('overwrite')\
            .options(header='true')\
            .save(saved_data_file_path)

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'unprocessed_data_file_path',
        'training_data_dir_path',
        'school_master_file_path'
    ]
    usage = 'Usage:\n' \
        + '    process_training_data_script.py\n' \
        + '        --unprocessed_data_file_path=<unprocessed_data_file_path>\n' \
        + '        --training_data_dir_path=<training_data_dir_path>\n' \
        + '        --school_master_file_path=<school_master_file_path>\n' \
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
    school_master_file_path = parameters['school_master_file_path']
    env = parameters['env']

    # execute
    ptd = ProcessTrainingData(env)
    ptd.run(unprocessed_data_file_path, training_data_dir_path, school_master_file_path)

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
