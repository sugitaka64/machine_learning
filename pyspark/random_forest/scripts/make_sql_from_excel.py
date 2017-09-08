#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DODA plus, Prediction for Passing Paper exam.
make sql files from excel file.
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import sys

class DpPPPError(Exception):
    """common error class."""

    pass

class MakeSqlFromExcel(object):
    """make sql files from excel file."""

    def __init__(self, env='local'):
        """init."""
        self.env = env

    def __param_check(
        self,
        input_excel_file_path: str,
        output_dir_path: str
    ) -> bool:
        """check parameter."""
        if self.env == 'local':
            if not os.path.isfile(input_excel_file_path):
                raise DpPPPError('invalid parameter: input_excel_file_path')
            if not os.path.isdir(output_dir_path):
                raise DpPPPError('invalid parameter: output_dir_path')

        return True

    def run(
        self,
        input_excel_file_path: str,
        output_dir_path: str
    ) -> bool:
        """execute."""
        # check parameter
        self.__param_check(
            input_excel_file_path,
            output_dir_path
        )

        # read excel file
        df = pd.read_excel(
            input_excel_file_path,
            dtype=np.unicode
        )
        df = df[
            [
                '案件No',
                '案件ID',
                '制作ID',
            ]
        ].rename(
            columns={
                '案件No': 'aodrno',
                '案件ID': 'aodrid',
                '制作ID': 'wjoid',
            }
        )

        # save aodrid_to_wjoid.csv
        output_file_path = output_dir_path + 'aodrid_to_wjoid.csv'
        df.to_csv(output_file_path, index=False, header=True)

        # make sql
        sql_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/../sql/'

        # target_items.sql
        l = []
        for aodrno in df['aodrno']:
            if aodrno != '' and aodrno != 'nan':
                l.append('        \'%s\'' % (aodrno))
        aodrnos = ',\n'.join(map(str, l))
        input_file_path = sql_file_dir + 'target_items.sql'
        sql = open(input_file_path).read() % (aodrnos)
        output_file_path = output_dir_path + 'target_items.sql'
        f = open(output_file_path, 'w')
        f.write(sql)
        f.close()

        # applied.sql
        l = []
        for aodrid in df['aodrid']:
            if aodrid != '' and aodrid != 'nan':
                l.append('      %s' % (aodrid))
        aodrids = ',\n'.join(map(str, l))
        input_file_path = sql_file_dir + 'applied.sql'
        sql = open(input_file_path).read() % (aodrids)
        output_file_path = output_dir_path + 'applied.sql'
        f = open(output_file_path, 'w')
        f.write(sql)
        f.close()

        # passed_paper_exam.sql
        input_file_path = sql_file_dir + 'passed_paper_exam.sql'
        sql = open(input_file_path).read() % (aodrids)
        output_file_path = output_dir_path + 'passed_paper_exam.sql'
        f = open(output_file_path, 'w')
        f.write(sql)
        f.close()

        # send_scout_mail.sql
        l = []
        for wjoid in df['wjoid']:
            if wjoid != '' and wjoid != 'nan':
                l.append('      %s' % (wjoid))
        wjoids = ',\n'.join(map(str, l))
        input_file_path = sql_file_dir + 'send_scout_mail.sql'
        sql = open(input_file_path).read() % (wjoids)
        output_file_path = output_dir_path + 'send_scout_mail.sql'
        f = open(output_file_path, 'w')
        f.write(sql)
        f.close()

        return True

def __parser() -> dict:
    """check parameters."""
    arguments = sys.argv
    predicted_options = [
        'input_excel_file_path',
        'output_dir_path'
    ]
    usage = 'Usage:\n' \
        + '    make_sql_from_excel.py\n' \
        + '        --input_excel_file_path=<input_excel_file_path>\n' \
        + '        --output_dir_path=<output_dir_path>\n' \
        + '        [--env=<env>]\n' \
        + '    make_sql_from_excel.py -h | --help'

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
    input_excel_file_path = parameters['input_excel_file_path']
    output_dir_path = parameters['output_dir_path']
    env = parameters['env']

    # execute
    msfe = MakeSqlFromExcel(env)
    msfe.run(
        input_excel_file_path,
        output_dir_path
    )

    print('%s %s end.' % (datetime.today(), __file__))

    sys.exit(0)
