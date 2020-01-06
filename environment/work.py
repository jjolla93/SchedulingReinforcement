from random import randint


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


import pandas as pd

# 원 데이터
df_raw = pd.read_excel('./data/191227_납기일 추가.xlsx')
# 가공 데이터
df_proc = pd.DataFrame(columns=['액티비티코드', '호선', '블록', '블록그룹', '공정', '계획착수일', '계획완료일', '계획공기', '계획공수',
                                '중량', '블록단계', '납기일'])

# 전체 data 개수
data_num = len(df_raw)

# P나 S 가공 전에 저장해줄 딕셔너리
dict_temp = {}
'''
{ block_group : { proj_num : { block : 시리즈 } } } 
'''

# 데이터 읽어오기
for i in range(data_num):
    temp = df_raw.loc[i]  # 한 줄씩 읽어주기

    block_group = temp['블록그룹']
    proj_num = temp['호선']
    block_name = temp['블록']

    # 공정 4, 6, 7, 8만 추출
    if (temp['공정'] == 4) or (temp['공정'] == 6) or (temp['공정'] == 7) or (temp['공정'] == 8):
        # 가공을 위한 P나 S를 저장하기 위함 -> { 블록 그룹 : { 호선 번호 : { 블록 : } } }
        if block_group not in dict_temp.keys():
            dict_temp[block_group] = {}
        if proj_num not in dict_temp[block_group].keys():
            dict_temp[block_group][proj_num] = {}

        # P나 S가 아닐 경우 df_proc에 바로 저장 / P나 S일 경우 딕셔너리에 저장 후, 가공, 그리고 다시 df_proc에 넣어줌
        if temp['블록단계'] == '조립':  # 블록 끝이 P0 / S0 / A0 / C0
            if (temp['블록'][-2:] != 'P0') and (temp['블록'][-2:] != 'S0'):  # A / C 일 때는 바로 저장(가공 X)
                df_proc = df_proc.append((pd.DataFrame(temp)).T)
            else:  # P / S 일 때에는 딕셔너리에 저장
                dict_temp[block_group][proj_num][block_name] = temp
        else:  # 블록단계 = 단위 / 블록 끝이 P / S / A
            if (temp['블록'][-1] != 'P') and (temp['블록'][-1] != 'S'):  # A / C 일 때는 바로 저장(가공 X)
                df_proc = df_proc.append((pd.DataFrame(temp)).T)
            else:  # P / S 일 때에는 딕셔너리에 저장
                dict_temp[block_group][proj_num][block_name] = temp

# P, S 가공
for block_group in dict_temp.keys():
    for proj_num in dict_temp[block_group].keys():

        temp_list = []  # 처리한 데이터의 key 값을 저장하는 list
        for block_name in dict_temp[block_group][proj_num].keys():
            if block_name not in temp_list:  # temp_list에 없는 블록만 작업 수행
                temp_list.append(block_name)
                block_1 = dict_temp[block_group][proj_num][block_name]

                # 짝인 블록 이름 설정해주기
                if block_1['블록단계'] == '조립':  # 블록 : (블록그룹) + P0 / S0
                    if block_name[-2] == 'P':  # 블록 : (블록그룹) + P0 --> 짝 : (블록그룹) + S0
                        block_name_2 = block_name[:-2] + 'S0'
                    else:  # 블록 : (블록그룹) + S0 --> 짝 : (블록그룹) + P0
                        block_name_2 = block_name[:-2] + 'P0'
                else:  # 단위 -> 블록 : (블록그룹) + _P / _S
                    if block_name[-1] == 'P':  # 블록 : (블록그룹) + _P --> 짝 : (블록그룹) + _S
                        block_name_2 = block_name[:-1] + 'S'
                    else:  # 블록 : (블록그룹) + _S --> 짝 : (블록그룹) + _P
                        block_name_2 = block_name[:-1] + 'P'

                if block_name_2 not in dict_temp[block_group][proj_num].keys():  # 짝인 블록이 존재하지 않을 때
                    df1 = dict_temp[block_group][proj_num][block_name]
                    df_proc = df_proc.append((pd.DataFrame(df1)).T)
                else:  # 만약 짝인 블록이 존재 한다면,
                    temp_list.append(block_name_2)

                    # block_name_2를 블록 이름으로 가지는 블록 정보(dictionary 형태)
                    block_2 = dict_temp[block_group][proj_num][block_name_2]
                    block_1 = dict(block_1)
                    block_2 = dict(block_2)

                    # df_proc에 저장할 임의의 dataframe -> 계획 공수, 중량, 이름 합치기
                    df_PS = block_2
                    df_PS['계획공수'] = block_1['계획공수'] + block_2['계획공수']
                    df_PS['중량'] = block_1['중량'] + block_2['중량']
                    if block_1['블록단계'] == '조립':
                        df_PS['블록'] = block_group + 'P0S0'
                    else:
                        df_PS['블록'] = block_group + block_1['블록'][-2:] + block_2['블록'][-2:]

                    # dictionary -> 데이터프레임으로 df_proc에 저장
                    df_proc = df_proc.append((pd.DataFrame.from_dict(df_PS, orient='index')).T)

df_proc.to_excel('./data/data.xlsx', sheet_name='Sheet1', index=False)