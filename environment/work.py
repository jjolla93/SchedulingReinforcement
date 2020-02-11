from random import randint
from PIL import Image
import pandas as pd
import numpy as np
import xlsxwriter
import datetime
import scipy
import copy

project = None
blocks = None
days = None
activities = None
zero_point = None

class Work(object):
    def __init__(self, work_id=None, block=None, start_date=None, finish_date=None, lead_time=1, earliest_start=-1, latest_finish=-1, max_days=4):
        self.id = str(work_id)
        self.block = block
        self.start_date_plan = start_date
        self.finish_date_plan = finish_date
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        self.start_date_lr = None
        if lead_time == -1:
            self.lead_time = randint(1, 1 + max_days // 3)
        else:
            self.lead_time = lead_time


def import_blocks_schedule(filepath, backward=True):

    df_rev = pd.read_excel(filepath)
    #df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    global project
    project = 3095
    df = df_rev[df_rev['호선'] == project]

    masking = [a or b or c or d for a, b, c, d in zip(df['공정'] == 4, df['공정'] == 6, df['공정'] == 7, df['공정'] == 8)]
    df_schedule = df[masking]
    if backward:
        df_schedule.sort_values(by=['납기일', '호선', '블록그룹', '계획착수일', '블록'], inplace=True, ascending=False)
    else:
        df_schedule.sort_values(by=['납기일', '호선', '블록그룹', '계획착수일', '블록'], inplace=True)
    df_schedule.reset_index(drop=True, inplace=True)

    df_schedule['계획착수일'] = pd.to_datetime(df_schedule['계획착수일'], format='%Y%m%d')
    df_schedule['계획완료일'] = pd.to_datetime(df_schedule['계획완료일'], format='%Y%m%d')
    df_schedule['납기일'] = pd.to_datetime(df_schedule['납기일'], format='%Y%m%d')
    initial_date = df_schedule['계획착수일'].min()
    df_schedule['계획착수일'] = (df_schedule['계획착수일'] - initial_date).dt.days
    df_schedule['계획완료일'] = (df_schedule['계획완료일'] - initial_date).dt.days
    df_schedule['납기일'] = (df_schedule['납기일'] - initial_date).dt.days

    works = []
    block = 0
    max_days = df_schedule['납기일'].max() + 1

    while len(df_schedule) != 0:
        first_row = df_schedule.loc[0]
        masking = [a and b for a, b in zip(df_schedule['호선'] == first_row['호선'],
                                           df_schedule['블록그룹'] == first_row['블록그룹'])]
        temp = df_schedule[masking]
        block_num = len(temp)

        while len(temp) != 0:
            if len(temp) == 1:
                works.append(Work(work_id=temp.loc[0]['액티비티코드'],
                                  block=block,
                                  start_date=temp.loc[0]['계획착수일'],
                                  finish_date=temp.loc[0]['계획완료일'],
                                  lead_time=temp.loc[0]['계획완료일']-temp.loc[0]['계획착수일']+1,
                                  latest_finish=temp.loc[0]['납기일'],
                                  max_days=max_days))
                temp.drop([0], inplace=True)
                temp.reset_index(drop=True, inplace=True)
            else:
                if temp.loc[0]['계획착수일'] != temp.loc[1]['계획착수일']:
                    works.append(Work(work_id=temp.loc[0]['액티비티코드'],
                                      block=block,
                                      start_date=temp.loc[0]['계획착수일'],
                                      finish_date=temp.loc[0]['계획완료일'],
                                      lead_time=temp.loc[0]['계획완료일']-temp.loc[0]['계획착수일']+1,
                                      latest_finish=temp.loc[0]['납기일'],
                                      max_days=max_days))
                    temp.drop([0], inplace=True)
                    temp.reset_index(drop=True, inplace=True)
                else:
                    works.append(Work(work_id=temp.loc[0]['액티비티코드']+temp.loc[1]['액티비티코드'][-4:],
                                      block=block,
                                      start_date=temp.loc[0]['계획착수일'],
                                      finish_date=temp.loc[0]['계획완료일'],
                                      lead_time=temp.loc[0]['계획완료일']-temp.loc[0]['계획착수일']+1,
                                      latest_finish=temp.loc[0]['납기일'],
                                      max_days=max_days))
                    temp.drop([0, 1], inplace=True)
                    temp.reset_index(drop=True, inplace=True)

        df_schedule.drop([_ for _ in range(block_num)], inplace=True)
        df_schedule.reset_index(drop=True, inplace=True)
        block += 1

    global activities
    activities = works[:]
    global zero_point
    zero_point = initial_date
    global blocks
    blocks = block
    global days
    days = max_days

    return works, block, max_days


def calculate_deviation(state):
    s = copy.copy(state)
    s[s == -1] = 0

    loads = np.sum(s, axis=0)
    start = (np.where(loads != 0))[0]
    deviation = np.std(loads[start[0]:(start[-1] + 1)])

    return deviation


def make_image(filepath, name, state):
    s1 = copy.copy(state)
    s2 = copy.copy(state)
    s1[s1 > 0] = 1
    s2[s2 == -1] = 0
    loads = np.sum(s2, axis=0)
    color_map = {
        0: [0, 0, 0],  # black
        1: [0, 0, 255],  # blue
        -1: [255, 0, 0],  # red
    }
    colored_image = np.zeros([state.shape[0] + 1, state.shape[1], 3])
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            colored_image[i + 1, j] = color_map[int(s1[i, j])]
    for i in range(len(loads)):
        if loads[i] == 0.0:
            colored_image[0, i] = [0, 0, 0]
        else:
            grey = max(0, 255 - 0.025 * 255 * loads[i])
            colored_image[0, i] = [grey, grey, grey]

    big_image = scipy.misc.imresize(colored_image, [state.shape[0] * 30, state.shape[1] * 30], interp='nearest')
    image = Image.fromarray(big_image.astype('uint8'), 'RGB')
    image.save(filepath + '_' + name + '.png')


def export_blocks_schedule(filepath):
    plan_state = np.full([blocks, days], 0)
    state = np.full([blocks, days], 0)
    for activity in activities:
        plan_state[activity.block, activity.start_date_plan:(activity.finish_date_plan + 1)] += 1
        plan_state[activity.block, activity.latest_finish] = -1
        state[activity.block, activity.start_date_lr:(activity.start_date_lr + activity.lead_time)] += 1
        state[activity.block, activity.latest_finish] = -1

    make_image(filepath, 'plan', plan_state)
    make_image(filepath, 'a3c', state)

    deviation_plan = calculate_deviation(plan_state)
    deviation = calculate_deviation(state)

    workbook = xlsxwriter.Workbook(filepath + '_{0}호선_{1}표준편차.xlsx'.format(project, round(deviation, 3)))
    worksheet = workbook.add_worksheet()

    row, col = 0, 0
    column_names = ['액티비티코드', '계획착수일_계획', '계획완료일_계획', '계획착수일', '계획완료일', '공기', '납기일']
    for column_name in column_names:
        worksheet.write(row, col, column_name)
        col +=1

    for i in range(len(activities)):
        row += 1
        col = 0
        worksheet.write(row, col, activities[i].id)
        worksheet.write(row, col + 1, (pd.to_datetime(zero_point) + datetime.timedelta(days=int(activities[i].start_date_plan))).strftime('%Y%m%d'))
        worksheet.write(row, col + 2, (pd.to_datetime(zero_point) + datetime.timedelta(days=int(activities[i].finish_date_plan))).strftime('%Y%m%d'))
        worksheet.write(row, col + 3, (pd.to_datetime(zero_point) + datetime.timedelta(days=int(activities[i].start_date_lr))).strftime('%Y%m%d'))
        worksheet.write(row, col + 4, (pd.to_datetime(zero_point) + datetime.timedelta(days=int(activities[i].start_date_lr)+int(activities[i].lead_time)-1)).strftime('%Y%m%d'))
        worksheet.write(row, col + 5, activities[i].lead_time)
        worksheet.write(row, col + 6, (pd.to_datetime(zero_point) + datetime.timedelta(days=int(activities[i].latest_finish))).strftime('%Y%m%d'))

    row, col = 0, 11
    worksheet.write(row, col, '표준편차')

    row, col = 1, 10
    worksheet.write(row, col, '계획 데이터')
    worksheet.write(row + 1, col, '결과 데이터')

    row, col = 1, 11
    worksheet.write(row, col, deviation_plan)
    worksheet.write(row + 1, col, deviation)

    row, col = 5, 10
    worksheet.write(row, col, '계획 데이터')
    worksheet.write(row + 15, col, '결과 데이터')
    worksheet.insert_image(row + 1, col, filepath + '_plan' + '.png', {'x_scale': 0.3, 'y_scale': 0.3})
    worksheet.insert_image(row + 16, col, filepath + '_a3c' + '.png', {'x_scale': 0.3, 'y_scale': 0.3})

    workbook.close()