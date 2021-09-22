import numpy as np

class Env(object):
    def __init__(self, theta, initial_c):
        super(Env, self).__init__()
        self.c1 = [initial_c]*2
        self.c1_record = [self.c1[:]]
        self.c2 = [initial_c]*2                 
        self.c2_record = [self.c2[:]]           
        self.acc1_record = [0.0]                
        self.acc2_record = [0.0]                
        self.r1_record = [0.0]
        self.theta = theta
        self.embedding1 = None
        self.embedding2 = None
        self.stop = [False, False]

    def get_observation(self):
        select_num = ', '.join([format(num,'.3f') for num in self.c1])
        observation = select_num
        return observation

    def step(self, actions, hgnn, data, ace):
        '''
        Get reward and next state based on current state and action;
        Train model at the same time
        '''
        last_c2 = self.c2_record[-1]  
        last_c1 = self.c1_record[-1]
        
        if actions[0] == 1:
            hgnn.change_curv(last_c2)
            self.c1 = last_c2
        else:
            hgnn.change_curv(last_c1)
            self.c1 = last_c1
        self.c1_record.append(self.c1[:])

        # c' = 1 / (theta * abs(k) + (1-theta) * 1/c)
        action_1 = actions[1] // 2 if not self.stop[0] else 0
        action_2 = actions[1] % 2 if not self.stop[1] else 0
        if action_1+action_2 > 0:
            kappa = ace.estimation(self.embedding2)
            if action_1 == 1:
                self.c2[0] = 1 / ((1-self.theta) / self.c2[0] + self.theta * abs(kappa)) 
            if action_2 == 1:
                self.c2[1] = 1 / ((1-self.theta) / self.c2[1] + self.theta * abs(kappa)) 
        self.c2_record.append(self.c2[:])

        # HGNN Agent
        embeddings1 = hgnn.encode(data['features'], data['adj_train_norm'])
        train_metrics1 = hgnn.compute_metrics(embeddings1, data, 'train')
        curr_acc1 = train_metrics1[hgnn.key_param]
        reward1 = 100 * (curr_acc1 - self.acc1_record[-1])
        self.acc1_record.append(curr_acc1)
        self.r1_record.append(reward1)

        # ACE Agent
        self.embeddings2 = ace.mapping(self.embedding1, self.c1_record[-2], self.c2) 
        train_metrics2 = ace.compute_metrics(self.embeddings2, data, 'train')
        curr_acc2 = train_metrics2[ace.key_param]
        reward2 = 100 * (curr_acc2 - self.acc2_record[-1])
        self.acc2_record.append(curr_acc2)

        self.embedding1 = embeddings1.clone().detach()

        return (reward1, reward2), train_metrics1
