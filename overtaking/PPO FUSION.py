import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from mydynamic import vehicle_dynamic
import socket
import pickle
# create socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect to the server
client_socket.connect(('127.0.0.1', 8000))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh(),

        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        #cov_mat=cov_mat/100
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        m_state_dict = torch.load('./PPO_continuous_solved_{}.pth')
        self.policy_old.load_state_dict(m_state_dict)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards=rewards.float()

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            #loss.mean().backward()
            loss=loss.mean()
            #print(loss)
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        m_state_dict1 = torch.load('./PPO_continuous_solved_{}.pth')
        self.policy_old.load_state_dict(m_state_dict1)




# Simulation plot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

def myplot(xground, yground):
    fig, ax = plt.subplots()
    ax.plot(xground, yground)
    ax.plot([0, 200], [0, 0], '-g', linewidth=1.2)
    ax.plot([0, 200], [5, 5], '-b', linewidth=1.2)
    ax.plot([0, 200], [-5, -5], '-b', linewidth=1.2)
    ellipse = Ellipse(xy=(30, 0), width=12, height=6, edgecolor='r', fc='None')
    ax.add_patch(ellipse)

    def update(frame):
        # Calculate new center coordinates for the ellipse
        x = 30 + frame * 0.2  # Move the ellipse 0.2 units along the X axis in each frame
        ellipse.set_center((x, 0))
        return ellipse,

    ani = FuncAnimation(fig, update, frames=len(xground), interval=200, blit=True)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()




def main():
    ############## Hyperparameters ##############
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 8000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 500  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr=0.0003
    betas = (0.9, 0.999)

    random_seed = None

    # creating environment
    state_dim = 16# env.observation_space.shape[0]
    action_dim =2 # env.action_space.shape[0]

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    time_step = 0


    #test part
    loss=0
    actionlistx=[]
    actionlisty=[]

    for i in range(200):
        x = 0
        y = 0
        vx = 3
        vy = 0
        fai = 0
        faidot = 0
        xb = 30
        yb = 0
        vbx = 1
        vby = 0
        faib = 0
        faibdot = 0
        reward = 0
        xb1 = -50
        yb1 = 0
        vbx1 = 0
        vby1 = 0

        reward = 0
        state = [x, y, vx, vy, fai, faidot, xb, yb, vbx, vby,xb1,yb1,vbx1,vby1, faib, faibdot]
        state = np.array(state)
        print("now is",i,)
        tempactlistx=[]
        tempactlisty=[]
        for t in range(max_timesteps):
            xb = xb +0.2
            if (x - xb) * (x - xb) / 4 + (y - yb) * (y - yb) < 9:
                loss=loss+1
                print("crash!!!")
                tempactlistx = []
                tempactlisty = []
                break

            if (x - xb1) * (x - xb1) / 4 + (y - yb1) * (y - yb1) < 16:
                loss = loss + 1
                print("crash!!!!!!!")
                tempactlistx = []
                tempactlisty = []
                break

            if abs(y)>20:
                loss=loss+1
                print("outside the road!OOOO")
                tempactlistx = []
                tempactlisty = []
                break





            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)

            # exit to MPC
            data = [action[0], action[1]]  # Pack two numbers into a list
            data_str = pickle.dumps(data)  # Serialize data into byte stream
            client_socket.send(data_str)


            # state, reward, done, _ = env.step(action)
            # print("t",t)
            vx = vx + abs(action[0])
            next_state = vehicle_dynamic(y, vy, fai, faidot, vx, action[1])
            fai = fai + next_state[2]
            vy = next_state[0] + next_state[1]
            faidot = next_state[2]
            fai = fai + faidot
            x = x + vx * 1
            y = y + vy * 1
            tempactlistx.append(x)
            tempactlisty.append(-y)
            #Entrance state X0 from MPC to PPO
            data_str = client_socket.recv(1024)
            if not data_str:
                break
            # deserialized data
            data = pickle.loads(data_str)
            # operate on the data
            result = [num for num in data]
            # print result
            print(f"Received data: {data}, Result: {result}")

            xx = result[0]
            yy = result[1]
            # end of entrance

            state = [x, y, vx, vy, fai, faidot, xb, yb, vbx, vby,xb1,yb1,vbx1,vby1, faib, faibdot]
            state = np.array(state)
            print("my car x,y",x,y)
            print("my car vx,vy", vx, vy)
            print("ax,ay",action[0],action[1])
            done = 0
            if x > 200 or abs(y) > 300:
                done = 1
            if done:
                break



        if tempactlistx!=[]:
            actionlistx.append(tempactlistx)
        if tempactlisty != []:
            actionlisty.append(tempactlisty)
    winrate=(200-loss)/200
    print("rate",winrate)

    print(actionlistx)
    print(actionlisty)


if __name__ == '__main__':
    main()

