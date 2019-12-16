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