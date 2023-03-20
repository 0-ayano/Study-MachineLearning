
import numpy as np

def shortest_path(start):
    path = [start]
    p_pos = start
    n_pos = p_pos#
    while(n_pos != 8):
        n_pos = np.argmax(Q[p_pos,])
        path.append(n_pos)
        p_pos = n_pos
    return path

gamma = 0.9
alpha = 0.1

reward = np.array([[0,1,0,1,0,0,0,0,0],
                   [1,0,1,0,1,0,0,0,0],
                   [0,1,0,0,0,0,0,0,0],
                   [1,0,0,0,0,0,1,0,0],
                   [0,1,0,0,0,1,0,1,0],
                   [0,0,0,0,1,0,0,0,10000],
                   [0,0,0,1,0,0,0,1,0],
                   [0,0,0,0,1,0,1,0,0],
                   [0,0,0,0,0,1,0,0,0]])

Q = np.array(np.zeros([9,9]))


for i in range(10000):
    p_state = np.random.randint(0,9)
    n_actions = []
    for j in range(9):
        if reward[p_state,j] >= 1:
            n_actions.append(j)
    n_state = np.random.choice(n_actions)
    Q[p_state,n_state] = (1-alpha)*Q[p_state,n_state]+alpha*(reward[p_state,n_state]+gamma*Q[n_state,np.argmax(Q[n_state,])])

print(shortest_path(0))