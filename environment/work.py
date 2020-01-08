from random import randint
import pandas as pd


class Work(object):
    def __init__(self, work_id=None, block=None, lead_time=1, earliest_start=-1, latest_finish=-1, max_days=4):
        self.id = str(work_id)
        self.block = block
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        if lead_time == -1:
            self.lead_time = randint(1, 1 + max_days // 3)
        else:
            self.lead_time = lead_time


def import_blocks_schedule(filepath):

    df_rev = pd.read_excel(filepath)
    #df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df = df_rev[df_rev['호선'] == 2962]

    masking = [a or b or c or d for a, b, c, d in zip(df['공정'] == 4, df['공정'] == 6, df['공정'] == 7, df['공정'] == 8)]
    df_schedule = df[masking]
    df_schedule.sort_values(by=['납기일', '호선', '블록그룹', '계획착수일', '블록'], inplace=True)
    df_schedule.reset_index(drop=True, inplace=True)

    df_schedule['계획착수일'] = pd.to_datetime(df_schedule['계획착수일'], format='%Y%m%d')
    df_schedule['납기일'] = pd.to_datetime(df_schedule['납기일'], format='%Y%m%d')
    initial_date = df_schedule['계획착수일'].min()
    df_schedule['계획착수일'] = (df_schedule['계획착수일'] - initial_date).dt.days
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
                                  block=block, #temp.loc[0]['블록그룹'],
                                  lead_time=temp.loc[0]['계획공기'],
                                  latest_finish=temp.loc[0]['납기일'],
                                  max_days=max_days))
                temp.drop([0], inplace=True)
                temp.reset_index(drop=True, inplace=True)
            else:
                if temp.loc[0]['계획착수일'] != temp.loc[1]['계획착수일']:
                    works.append(Work(work_id=temp.loc[0]['액티비티코드'],
                                      block=block, #temp.loc[0]['블록그룹'],
                                      lead_time=temp.loc[0]['계획공기'],
                                      latest_finish=temp.loc[0]['납기일'],
                                      max_days=max_days))
                    temp.drop([0], inplace=True)
                    temp.reset_index(drop=True, inplace=True)
                else:
                    works.append(Work(work_id=temp.loc[0]['액티비티코드']+temp.loc[1]['액티비티코드'][-4:],
                                      block=block, #temp.loc[0]['블록그룹']+temp.loc[1]['블록그룹'][-2:],
                                      lead_time=temp.loc[0]['계획공기'],
                                      latest_finish=temp.loc[0]['납기일'],
                                      max_days=max_days))
                    temp.drop([0, 1], inplace=True)
                    temp.reset_index(drop=True, inplace=True)

        df_schedule.drop([_ for _ in range(block_num)], inplace=True)
        df_schedule.reset_index(drop=True, inplace=True)
        block += 1

    return works


if __name__ == '__main__':
    works = import_blocks_schedule('./data/191227_납기일 추가.xlsx')
    print(len(works))