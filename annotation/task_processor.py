from jiwer import wer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from decimal import Decimal
import toloka.client as toloka


class TaskProcessor:
    def __init__(self, toloka_client, pool_id, skill_id, groud_truth, threshold_wer=40, window=5):
        self.toloka_client = toloka_client
        self.pool_id = pool_id
        self.skill_id = skill_id
        self.char_set = set(list(' qwertyuiopasdfghjklzxcvbnm\''))
        self.gt = self._load_gt(groud_truth)
        self.steps = 0
        self.window = window
        self.writer = SummaryWriter()
        
        self.user_wer_history = dict()
        self.user_skills = dict()
        self.total_accepted = 0
        self.total_rejected = 0
        self.threshold_wer = threshold_wer
        
    def process(self):
        request_for_result = toloka.search_requests.AssignmentSearchRequest(
            status=toloka.assignment.Assignment.SUBMITTED,
            pool_id=self.pool_id,
        )
        
        to_reject = set()
        to_accept = set()
        
        users_made_action = set()
        
        for assignment in self.toloka_client.get_assignments(request_for_result):
            assignment_id = assignment.id
            user_id = assignment.user_id
            
            was_playing = True
            for i, solution in enumerate(assignment.solutions):
                if not self._process_solution(solution, user_id, assignment.tasks[i].input_values['audio']):
                    was_playing = False
                    break
            if not was_playing:
                to_reject.add((assignment_id, 'Audio does not play'))
            else:
                to_accept.add((assignment_id, 'Accepted'))
                users_made_action.add(user_id)

        self._set_skills(users_made_action)
        self._reject_and_accept(to_reject, to_accept)
        self._send_stats(len(to_reject), len(to_accept))
        self.steps += 1
        print(f'{self.pool_id}: Rejected: {len(to_reject)}\tAccepted: {len(to_accept)}')
        
    def _process_solution(self, one_solution, user_id, task):
        solution = one_solution.output_values
        if solution['playing'] == 'no':
            return False
        else:
            transciption = self._make_text_clear(solution['transcription'])
            gt = self.gt[task]

            if user_id not in self.user_wer_history:
                self.user_wer_history[user_id] = []

            self.user_wer_history[user_id].append(wer(gt.split(), transciption.split()))
            
            return True
        
    def _make_text_clear(self, text):
        return ''.join([c for c in text.strip().lower() if c in self.char_set])
    
    def _load_gt(self, path):
        gt = dict()
        with open(path) as f:
            lines = f.readlines()
            for line in lines:   
                audio, gt_text = line.split('\t')
                gt[audio] = self._make_text_clear(gt_text)
        return gt
    
    def _set_skills(self, users_made_action):
        for user_id in users_made_action:
            if len(self.user_wer_history[user_id]) >= self.window:
                skill_value = np.mean(self.user_wer_history[user_id][-5:]) * 100
                self.user_skills[user_id] = skill_value
                request = toloka.user_skill.SetUserSkillRequest(skill_id=self.skill_id, user_id=user_id, value=Decimal(100 - skill_value))
                try:
                    self.toloka_client.set_user_skill(request)
                except:
                    sleep(10)
                    continue
    
    def _send_stats(self, rejected, accepted):
        self.total_accepted += accepted
        self.total_rejected += rejected
        self.writer.add_scalar("Accepted/rejected", float(self.total_rejected), self.steps)
        self.writer.add_scalar("Accepted/accepted", float(self.total_accepted), self.steps)
        if len(self.user_skills) > 0:
            skills = np.array(list(self.user_skills.values()))
            self.writer.add_scalar("Skills/wer", np.mean(skills), self.steps)
            self.writer.add_histogram("Skills distribution", skills, self.steps)
            self.writer.add_scalar("Users/banned", float(np.sum(skills > self.threshold_wer)), self.steps)
        self.writer.add_scalar("Users/total", float(len(self.user_wer_history)), self.steps)
    
    def _reject_and_accept(self, to_reject, to_accept):
        for assignment_id, reason in to_reject:
            try:
                self.toloka_client.reject_assignment(assignment_id, reason)
            except:
                sleep(10)
                continue
        for assignment_id, reason in to_accept:
            try:
                self.toloka_client.accept_assignment(assignment_id, reason)
            except:
                sleep(10)
                continue