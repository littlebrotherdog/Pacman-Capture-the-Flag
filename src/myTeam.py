# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import tensorflow.compat.v1 as tf
import game
import numpy as np
import sys
import os
import wandb
import random
from util import nearestPoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Replay memory
from collections import deque

# Neural nets
from DQN import *
from game import Actions
from heuristicTeam import OffensiveAgent,terminator



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='terminator', **kwargs):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  addtional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    # return [eval(DQN_agent),eval(DQN_agent))]  # maybe like this
    print(f"PLayer 1: {first} red")
    print(f"Player 2: {second} orange")
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex, **kwargs), eval(second)(secondIndex, **kwargs)]


##########
# Agents #
##########

class DQN_agent(CaptureAgent):
    """
  DQN agent
  """
    def __init__(self, index, *args, **kwargs):
        CaptureAgent.__init__(self, index)
        # 从用户给的arguments中读取参数
        load_model = False # 如果 'load_model' 为 True，则从检查点文件加载模型
        if load_model:
            with open("saves/checkpoint") as f:
                data = f.readline()
                f.close()
            load = data[24:-2]  # 快速固定位置从checkpoint读取模型
            load = f"saves/{load}"
            print(load)
        else:
            load = None
        params = {
            # 模型备份
            'load_file': load,
            'save_file': "v1",
            'save_interval': 10000,  # 最初为 100000

            # 训练参数
            'TRAIN': True,
            'train_start': 10000,  # 开始训练前的步数 | 最初为 5000
            'batch_size': 32,  # 经验回放区一个batch大小 | 最初为 32
            'mem_size': 100000,  # 经验回放区大小

            'discount': 0.95,  # 折扣率(gamma)
            'lr': 0.0001,  # 学习率

            # epsilon的值(epsilon-greedy)
            'eps': 0.3,  # 开始值
            'eps_final': 0.01,  # 结束值
            'eps_step': 100000,  # 开始到结束的步数(linear)

            # 状态矩阵数量
            'STATE_MATRICES': 14,

            # 是否使用gpu
            'GPU': False
        }

        self.params = params
        self.params['num_training'] = kwargs.pop('numTraining', 0)

        # 从用户给的arguments中读取参数
        self.params['width'] = 34  # TODO gameState.data.layout.width
        self.params['height'] = 18  # TODO gameState.data.layout.height
        if self.params['TRAIN']:
            self.params['eps'] = 0.
            params['eps'] =0.
        # 开始Tensorflow
        # TODO Add GPU
        if self.params['GPU']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.qnet = DQN(self.params)

        # 创建时间戳
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q 和 cost
        self.Q_global = []
        self.cost_disp = 0

        # 开始
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.
        self.last_food_difference = 0

        self.replay_mem = deque()
        self.last_scores = deque()
        self.last_food_difference = deque()

        wandb.login()
        run = wandb.init(
            # 此处设置你的项目名
            project="pacman",
            # 此处配置需要Wandb帮你记录和track的参数
            config={
                "learning_rate": self.params['lr'],
                "episode": self.params['num_training']
            })

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''

        CaptureAgent.registerInitialState(self, gameState)

        '''
     Your initialization code goes here, if you need any.
     '''

        # 重置reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0
        self.last_food_difference = 0

        # 初始状态的flag
        self.first_state = True

        # 重置state
        self.last_state = None
        self.current_state = self.getStateMatrices(gameState)

        # 重置actions
        self.last_action = None

        # 重置一些变量
        self.terminal = None
        self.won = False
        self.Q_global = []
        self.delay = 0

        # 重置食物
        self.ourFood = self.CountOurFood(gameState)
        self.theirFood = self.CountTheirFood(gameState)

        # 回到中心
        self.atCenter = False # 在DQN之前回到中心
        self.center_counter = 0  # 经过多少个位置

        center_point = self.getCenterPos(gameState)
        self.ASTARPATH = self.getCenterPos(gameState)

        self.frame = 0
        self.numeps += 1

    def isWall(self,gameState,pos:tuple):
        grid = gameState.data.layout.walls
        return grid[pos[0]][pos[1]]

    def getCenterPos(self,gameState):
        width = self.params['width']
        height = self.params['height']
        # ASTAR

        if gameState.isOnRedTeam(self.index):
            pos_x = int(width / 2) - 1
            for i in range(1000):
              pos_y = random.randint(int(height / 4), int(0.75 * height))

              center = (pos_x,pos_y)
              if not self.isWall(gameState,center):
                  return deque(self.aStarSearch(gameState.getAgentPosition(self.index), gameState,
                                                [center]))
        else: # blue
            pos_x = int(width / 2) + 1
            for i in range(1000):
                pos_y = random.randint(int(height / 4), int(0.75 * height))
                center = (pos_x, pos_y)
                if not self.isWall(gameState, center):
                    return deque(self.aStarSearch(gameState.getAgentPosition(self.index), gameState,
                                                  [center]))

        #center_red = [(16, 10), (16, 7)]
        #center_blue = [(17, 7), (17, 10)]
        #i = random.randint(0,1)
    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def getMove(self, gameState):
        # Exploit / Explore
        if np.random.rand() > self.params['eps'] or not self.params['TRAIN']:
            # Exploit action
            # 使用神经网络模型（self.qnet）对当前状态进行Q值的预测
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict={self.qnet.x: np.reshape(self.current_state,
                                                   (1, self.params['width'], self.params['height'],
                                                    self.params['STATE_MATRICES'])),
                           self.qnet.q_t: np.zeros(1),
                           self.qnet.actions: np.zeros((1, 4)),
                           self.qnet.terminals: np.zeros(1),
                           self.qnet.rewards: np.zeros(1)})[0]

            # 将预测的Q值中的最大值添加到全局Q值列表（self.Q_global）
            self.Q_global.append(max(self.Q_pred))
            # 找到具有最大Q值的动作索引
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
                #print("Selected move: "+str(move))
            else:
                move = self.get_direction(
                    a_winner[0][0])
                #print("Q_pred: "+str(self.Q_pred)+" Selected move: "+str(move))
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))
            #print("Random move: "+str(move))

        # 保存上一个动作
        self.last_action = self.get_value(move)

        return move

    def CountOurFood(self, gameState):
        foodgrid = CaptureAgent.getFood(self, gameState)
        count = foodgrid.count()
        return count

    def CountTheirFood(self, gameState):
        foodgrid = CaptureAgent.getFoodYouAreDefending(self, gameState)
        count = foodgrid.count()
        return count

    def updateLastReward(self,currentGameState):

        # GameState objects
        lastGameState = self.getCurrentObservation()
        #currentGameState = self.getCurrentObservation()

        if (currentGameState.isOver()):
            print('GAME IS OVER')
            final_score = CaptureAgent.getScore(self, currentGameState)
            if final_score >0:
                self.won = True
            if (self.terminal and self.won):
                return 10000.  # win了给非常多reward
            elif final_score ==0:
                return - 500
            else:
                return -1000  # 输了


        myLastState = lastGameState.getAgentState(self.index)  # Returns AgentState object
        myCurrentState = currentGameState.getAgentState(self.index)  # Returns AgentState object

        # 位置
        xLast, yLast = lastGameState.getAgentPosition(self.index)
        xCurr, yCurr = currentGameState.getAgentPosition(self.index)

        # 分数信息
        lastScore = self.getScore(lastGameState)
        currentScore = self.getScore(currentGameState)
        self.last_score = lastScore  # 不用了
        self.current_score = currentScore  # 不用了

        # 食物和胶囊信息
        lastFood = self.getFood(lastGameState)
        lastFoodDefending = self.getFoodYouAreDefending(lastGameState)
        currentFood = self.getFoodYouAreDefending(currentGameState)
        currentFoodDefending = self.getFoodYouAreDefending(currentGameState)
        self.ourFood = self.CountOurFood(currentGameState)  # 不用了
        self.theirFood = self.CountTheirFood(currentGameState)  # 不用了

        lastCapsules = self.getCapsules(lastGameState)
        lastCapsulesDefending = self.getCapsulesYouAreDefending(lastGameState)
        currentCapsules = self.getCapsules(currentGameState)
        currentCapsulesDefending = self.getCapsulesYouAreDefending(currentGameState)

        # 查看吃豆人状态
        lastFoodCarrying = myLastState.numCarrying
        currentFoodCarrying = myCurrentState.numCarrying

        # 查看吃豆人是不是回来了
        lastFoodReturned = myLastState.numReturned
        currentFoodReturned = myCurrentState.numReturned

        # 算Reward
        reward = 0
        A = currentFoodCarrying - lastFoodCarrying  # 增加了 == 吃了食物, 减少了 = 放下食物 || 被吃了
        B = currentFoodReturned - lastFoodReturned  # 增加了 == 放下食物
        C = len(currentCapsulesDefending) - len(lastCapsulesDefending) # 减少了 == 敌人吃了胶囊
        D = len(currentCapsules) - len(lastCapsules)  # 减少了 == 自己吃了胶囊
        #E = currentFood.count() - lastFood.count()
        F = currentFoodDefending.count() - lastFoodDefending.count() # 减少了 == 我们的食物被吃了, 增加了 ==  我方幽灵吃了对方吃豆人 || 放下了豆子
        G = currentScore - lastScore

        if self.isPacman(currentGameState,self.index):
            reward += 100 # 鼓励进入对方区域

        if A > 0:
            reward += A*200  # 吃食物
        elif A < 0:
            if B > 0:
                reward += B*1000  # 放下食物
            # else:
            #     reward -= 100  # 被吃掉

        if currentGameState.getAgentPosition(self.index) == currentGameState.getInitialAgentPosition(self.index):
                self.atCenter = False
                self.ASTARPATH = self.getCenterPos(currentGameState)
                self.center_counter = 0
                reward -= 100  # 我方被吃回到原点

        # if C < 0:
        #     reward -= 5  # 我方胶囊被吃

        if D < 0:
            reward += 100  # 吃对方胶囊

        # if F < 0:
        #     reward -= F  # 我方食物被吃
        # elif F > 0:
        #     if B == 0:
        #         reward += 20  # 吃掉对方吃豆人，不完全正确，因为其他队员可能会放下豆子

        reward += G

        if reward == 0:
            reward = -1  # 什么都没有发生

        return reward

    def observation_step(self, gameState):
        if self.last_action is not None:
            # 处理当前经历的状态
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(gameState)

            self.last_reward = self.updateLastReward(gameState)  # 更新奖励
            self.ep_rew += self.last_reward

            # 存到经验回放区
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()
            if self.params['TRAIN']:
                # 保存模型
                if (self.params['save_file']):
                    if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params[
                        'save_interval'] == 0:
                        self.qnet.save_ckpt(
                            'saves/model-' + self.params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                        print('Model saved')

                # 训练
                self.train()
                self.params['eps'] =  max(self.params['eps_final'], 1.00 - float(self.cnt) / float(self.params['eps_step']))
        self.local_cnt += 1
        self.frame += 1

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)
        return state

    def final(self, gameState):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(gameState)

        # Print stats
        log_file = open('./logs/' + str(self.general_record_time) + '-l-' + str(self.params['width']) + '-m-' + str(
            self.params['height']) + '-x-' + str(self.params['num_training']) + '.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                       (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

        if self.params['TRAIN']:
            if self.params['num_training'] == self.numeps:
                # Save model
                if (self.params['save_file']):
                        self.qnet.save_ckpt(
                            'saves/model-' + self.params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                        print('Model saved')

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])  # Why random sampling?
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_n = []  # Next states (s')
            batch_t = []  # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)
            #print(self.cnt, self.cost_disp)
            wandb.log({"cnt": self.cnt, "loss": self.cost_disp})


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def chooseAction(self, gameState):
        """
    This will be our main method from where we get the action!
    """

        if not self.atCenter and self.center_counter == len(self.ASTARPATH) - 1:
            self.atCenter = True
            self.center_counter = 0

        #self.atCenter = True # Always true
        if self.atCenter:
            move = self.getMove(gameState)
        else:
            move = self.ASTARPATH[self.center_counter]
            self.center_counter += 1  # get next move

        # Stop moving when not legal
        legal = gameState.getLegalActions(self.index)

        if move not in legal:
            move = Directions.STOP

        # Save last gameState
        return move

    """Adjusted CODE FROM DQN paper TO GET STATE SPACE for CTF"""

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        size = self.params['STATE_MATRICES']
        total = np.zeros((size, size))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 7
        return total

    def isScared(self, gameState, index):
        """
    Says whether or not the given agent is scared
    """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def isGhost(self, gameState, index):
        """
    Returns true ONLY if we can see the agent and it's definitely a ghost
    """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False

        if not self.red:
            return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))
        else:
            return not ((not gameState.isOnRedTeam(index)) ^ (position[0] >= gameState.getWalls().width / 2))

    def isPacman(self, gameState, index):
        """
    Returns true ONLY if we can see the agent and it's definitely a pacman
    """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False

        if not self.red:
            return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))
        else:
            return not ((not gameState.isOnRedTeam(index)) ^ (position[0] < gameState.getWalls().width / 2))

    def getStateMatrices(self, state):

        """ Return wall, ghosts, food, capsules matrices """

        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell
            return matrix

        def getPacmanMatrix(state, who: str):
            """ Return matrix with the player coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)
            cell = 0  # default

            if who == 'Player':
                pos = state.getAgentPosition(self.index)
                if self.isPacman(state, self.index):
                    cell = 1

            elif who == 'Friend':
                team = self.getTeam(state)

                for agent in team:
                    if agent != self.index:
                        pos = state.getAgentPosition(agent)
                        if self.isPacman(state, agent):  # check if friend is pacman or not
                            cell = 1

            elif who == 'Enemy':
                enemies = self.getOpponents(state)

                for agent in enemies:
                    # TODO use probabilities if the

                    # ! TODO check if food is eaten nearby!
                    pos = state.getAgentPosition(agent)
                    if pos is not None:
                        if self.isPacman(state, agent):
                            cell = 1

            else:
                raise TypeError("Need to specify who to check for")

            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state, who: str):
            """ Return matrix with the player coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            if who == 'Player':
                pos = state.getAgentPosition(self.index)
                cell = 0
                if self.isGhost(state, self.index):  # just ghost
                    cell = 1
            elif who == 'Friend':
                team = self.getTeam(state)
                cell =0
                for agent in team:
                    if agent != self.index:
                        pos = state.getAgentPosition(agent)
                        if self.isGhost(state, agent):  # check if friend is pacman or not
                            cell = 1

            elif who == 'Enemy':
                enemies = self.getOpponents(state)
                cell =0
                for agent in enemies:
                    # TODO use probabilities if the
                    pos = state.getAgentPosition(agent)
                    # ! TODO check if food is eaten nearby!
                    if self.isGhost(state, agent):
                        cell = 1
            else:
                raise TypeError("Need to specify who to check for")

            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state, who: str):
            """ Return matrix with the player coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            cell = 0  # default

            if who == 'Player':
                pos = state.getAgentPosition(self.index)

                if self.isScared(state, self.index):  # if we're scared
                    cell = 1

            elif who == 'Friend':
                team = self.getTeam(state)

                for agent in team:
                    if agent != self.index:
                        pos = state.getAgentPosition(agent)
                        if self.isScared(state, agent):  # check if friend is pacman or not
                            cell = 1

            elif who == 'Enemy':
                enemies = self.getOpponents(state)

                for agent in enemies:
                    # ! TODO check if food is eaten nearby!
                    pos = state.getAgentPosition(agent)
                    if pos is not None:
                        if self.isScared(state, agent):
                            cell = 1

            else:
                raise TypeError("Need to specify who to check for")

            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def GetFoodMatrix(state, who: str):
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            if who == 'Defending':  # ours
                grid = self.getFoodYouAreDefending(state)

            elif who == 'Attacking':  # their
                grid = self.getFood(state)

            else:
                raise TypeError("Need to specify what food you want!")

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell
            #print(matrix)
            return matrix

        def GetCapsulesMatrix(state, who: str):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            if who == 'Defending':
                capsules = self.getCapsulesYouAreDefending(state)
            elif who == 'Attacking':
                capsules = self.getCapsules(state)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1 - i[1], i[0]] = 1

            return matrix

        def predictEnemyMatrix(state):
            self.last_food = GetFoodMatrix(state, 'Defending')
            # Check difference from previous
            # if enemy_isempty on our side

            pass

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        # ? 14 matrices
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((self.params['STATE_MATRICES'], height, width))

        # Player info
        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state, 'Player')  # our
        observation[2] = getGhostMatrix(state, 'Player')
        observation[3] = getScaredGhostMatrix(state, 'Player')

        # teammate info
        observation[4] = getPacmanMatrix(state, 'Friend')
        observation[5] = getGhostMatrix(state, 'Friend')
        observation[6] = getScaredGhostMatrix(state, 'Friend')

        # Enemy info
        observation[7] = getPacmanMatrix(state, 'Enemy')
        observation[8] = getGhostMatrix(state, 'Enemy')
        observation[9] = getScaredGhostMatrix(state, 'Enemy')

        # Food and capsules
        observation[10] = GetFoodMatrix(state, 'Defending')
        observation[11] = GetFoodMatrix(state, 'Attacking')
        observation[12] = GetCapsulesMatrix(state, 'Defending')
        observation[13] = GetCapsulesMatrix(state, 'Attacking')

        """
    We need 
    Opponent ghosts
    Our ghosts

    -- Getourplayer
    -- GetTheirplayer
    -- GetOurFood
    -- GetTheirFood

    -- maybe: Get Our and their ScaredGhost,Ghost, Capsule 

    """

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
    Finds the distance between the agent with the given index and its nearest goalPosition
    """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(
            lambda entry: entry[2] + width * height if entry[0] in avoidPositions else 0 + min(
                util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

        # Keeps track of visited positions
        visited = {currentPosition}

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath


class myOffensiveAgent(DQN_agent):
    """
    Offsenvie Agent overwrite getRewards and final functions.
    """
    def __init__(self, index, **kwargs):
        DQN_agent.__init__(self, index, **kwargs)
    def updateLastReward(self,currentGameState):

        # GameState objects
        lastGameState = self.getCurrentObservation()
        #currentGameState = self.getCurrentObservation()

        if (currentGameState.isOver()):
            print('GAME IS OVER')
            final_score = CaptureAgent.getScore(self, currentGameState)
            if final_score >0:
                self.won = True
            if (self.terminal and self.won):
                return 2000.  # win了给非常多reward
            elif final_score ==0:
                return -500
            else:
                return -1000  # 输了


        myLastState = lastGameState.getAgentState(self.index)  # Returns AgentState object
        myCurrentState = currentGameState.getAgentState(self.index)  # Returns AgentState object

        # 位置
        xLast, yLast = lastGameState.getAgentPosition(self.index)
        xCurr, yCurr = currentGameState.getAgentPosition(self.index)

        # 分数信息
        lastScore = self.getScore(lastGameState)
        currentScore = self.getScore(currentGameState)
        self.last_score = lastScore  # 不用了
        self.current_score = currentScore  # 不用了

        # 食物和胶囊信息
        lastFood = self.getFood(lastGameState)
        lastFoodDefending = self.getFoodYouAreDefending(lastGameState)
        currentFood = self.getFoodYouAreDefending(currentGameState)
        currentFoodDefending = self.getFoodYouAreDefending(currentGameState)
        self.ourFood = self.CountOurFood(currentGameState)  # 不用了
        self.theirFood = self.CountTheirFood(currentGameState)  # 不用了

        lastCapsules = self.getCapsules(lastGameState)
        lastCapsulesDefending = self.getCapsulesYouAreDefending(lastGameState)
        currentCapsules = self.getCapsules(currentGameState)
        currentCapsulesDefending = self.getCapsulesYouAreDefending(currentGameState)

        # 查看吃豆人状态
        lastFoodCarrying = myLastState.numCarrying
        currentFoodCarrying = myCurrentState.numCarrying

        # 查看吃豆人是不是回来了
        lastFoodReturned = myLastState.numReturned
        currentFoodReturned = myCurrentState.numReturned

        # 算Reward
        reward = 0
        A = currentFoodCarrying - lastFoodCarrying  # 增加了 == 吃了食物, 减少了 = 放下食物 || 被吃了
        B = currentFoodReturned - lastFoodReturned  # 增加了 == 放下食物
        C = len(currentCapsulesDefending) - len(lastCapsulesDefending)  # 减少了 == 敌人吃了胶囊
        D = len(currentCapsules) - len(lastCapsules)  # 减少了 == 自己吃了胶囊
        # E = currentFood.count() - lastFood.count()
        F = currentFoodDefending.count() - lastFoodDefending.count()  # 减少了 == 我们的食物被吃了, 增加了 ==  我方幽灵吃了对方吃豆人 || 放下了豆子
        G = currentScore - lastScore

        # MOVE TOWARDS FOOD
        myPos = currentGameState.getAgentState(self.index).getPosition()
        myLastPos = lastGameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(currentGameState).asList()
        food = 0
        minDist = 1000
        for a in foodList:
            dist = self.getMazeDistance(myPos, a)
            if dist < minDist:
                minDist = dist
                food = a
        if len(foodList) > 0:
            if currentGameState.isOnRedTeam(self.index):
                if self.getMazeDistance(myPos, food) < self.getMazeDistance(myLastPos, food) :
                    reward += 0.5
            else:
                if self.getMazeDistance(myPos, (food)) < self.getMazeDistance(myLastPos, food) :
                    reward += 0.5
        # BACK SAFE
        if currentFoodCarrying:
            if currentGameState.isOnRedTeam(self.index):
                if self.getMazeDistance(myPos, (9,9)) < self.getMazeDistance(myLastPos, (9,9)) :
                    reward += 3
            else:
                if self.getMazeDistance(myPos,(27,9)) < self.getMazeDistance(myLastPos, (27,9)) :
                    reward += 3

        # SCORES
        if self.getScore(currentGameState) > lastGameState.getScore():
            reward += self.getScore(currentGameState) - lastGameState.getScore() + 125

        # ATE_FOOD
        foodList = self.getFood(currentGameState).asList()
        prevFood = self.getFood(lastGameState).asList()
        if len(foodList) > len(prevFood):
            reward += len(foodList) - len(prevFood) + 2

        # DIED
        if currentGameState.getAgentPosition(self.index) == currentGameState.getInitialAgentPosition(self.index):
            lastX = lastGameState.getAgentPosition(self.index)[0]
            lastY = lastGameState.getAgentPosition(self.index)[1]
            currentX = currentGameState.getAgentPosition(self.index)[0]
            currentY = currentGameState.getAgentPosition(self.index)[1]
            if not (lastX == currentX + 1 or lastX == currentX - 1) and not (
                    lastY == currentY + 1 or lastY == currentY - 1):
                reward += (-250)

        # ATE_PACMAN
        oldEnemies = [lastGameState.getAgentState(i) for i in self.getOpponents(lastGameState)]
        newEnemies = [currentGameState.getAgentState(i) for i in self.getOpponents(currentGameState)]
        oldPacmen = [a for a in oldEnemies if a.isPacman and a.getPosition() != None]
        newPacmen = [a for a in newEnemies if a.isPacman and a.getPosition() != None]
        if len(oldPacmen) > 0:
            dists = [self.getMazeDistance(lastGameState.getAgentState(self.index).getPosition(), a.getPosition()) for a
                     in oldPacmen]
            for one in oldPacmen:
                if currentGameState.isOnRedTeam(self.index):
                    if self.getMazeDistance(myPos, one) < self.getMazeDistance(myLastPos, one):
                        reward += 1
                else:
                    if self.getMazeDistance(myPos, one) < self.getMazeDistance(myLastPos, one):
                        reward += 1
            if min(dists) == 1:
                if len(newPacmen) == 0 and currentGameState.getAgentState(
                        self.index).getPosition() != currentGameState.getInitialAgentPosition(self.index):
                    reward += 60
                elif len(newPacmen) > 0 and currentGameState.getAgentState(
                        self.index).getPosition() != currentGameState.getInitialAgentPosition(self.index):
                    dists = [self.getMazeDistance(currentGameState.getAgentState(self.index).getPosition(), a.getPosition())
                             for a in oldPacmen]
                    if min(dists) > 2:
                        reward += 60

        return reward
    def final(self, gameState):
        # Next
        self.ep_rew += self.last_reward
        # Do observation
        self.terminal = True
        self.observation_step(gameState)

        # Print stats
        log_file = open('./logs/' + str(self.general_record_time) + '-l-' + str(self.params['width']) + '-m-' + str(
            self.params['height']) + '-x-' + str(self.params['num_training']) + '.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                       (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()
        if self.params['TRAIN']:
            if self.params['num_training'] == self.numeps:
                # Save model
                if (self.params['save_file']):
                    self.qnet.save_ckpt(
                        'saves/model-' + self.params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')


class DefensiveAgent(DQN_agent):
    """
    Offsenvie Agent overwrite getRewards and final functions.
    """
    def __init__(self, index, **kwargs):
        DQN_agent.__init__(self, index, **kwargs)
    def updateLastReward(self,currentGameState):

        # GameState objects
        lastGameState = self.getCurrentObservation()
        #currentGameState = self.getCurrentObservation()

        if (currentGameState.isOver()):
            print('GAME IS OVER')
            final_score = CaptureAgent.getScore(self, currentGameState)
            if final_score >0:
                self.won = True
            if (self.terminal and self.won):
                return 2000.  # win了给非常多reward
            elif final_score ==0:
                return -500
            else:
                return -1000  # 输了


        myLastState = lastGameState.getAgentState(self.index)  # Returns AgentState object
        myCurrentState = currentGameState.getAgentState(self.index)  # Returns AgentState object

        # 位置
        xLast, yLast = lastGameState.getAgentPosition(self.index)
        xCurr, yCurr = currentGameState.getAgentPosition(self.index)

        # 分数信息
        lastScore = self.getScore(lastGameState)
        currentScore = self.getScore(currentGameState)
        self.last_score = lastScore  # 不用了
        self.current_score = currentScore  # 不用了

        # 食物和胶囊信息
        lastFood = self.getFood(lastGameState)
        lastFoodDefending = self.getFoodYouAreDefending(lastGameState)
        currentFood = self.getFoodYouAreDefending(currentGameState)
        currentFoodDefending = self.getFoodYouAreDefending(currentGameState)
        self.ourFood = self.CountOurFood(currentGameState)  # 不用了
        self.theirFood = self.CountTheirFood(currentGameState)  # 不用了

        lastCapsules = self.getCapsules(lastGameState)
        lastCapsulesDefending = self.getCapsulesYouAreDefending(lastGameState)
        currentCapsules = self.getCapsules(currentGameState)
        currentCapsulesDefending = self.getCapsulesYouAreDefending(currentGameState)

        # 查看吃豆人状态
        lastFoodCarrying = myLastState.numCarrying
        currentFoodCarrying = myCurrentState.numCarrying

        # 查看吃豆人是不是回来了
        lastFoodReturned = myLastState.numReturned
        currentFoodReturned = myCurrentState.numReturned

        # 算Reward
        reward = 0
        A = currentFoodCarrying - lastFoodCarrying  # 增加了 == 吃了食物, 减少了 = 放下食物 || 被吃了
        B = currentFoodReturned - lastFoodReturned  # 增加了 == 放下食物
        C = len(currentCapsulesDefending) - len(lastCapsulesDefending)  # 减少了 == 敌人吃了胶囊
        D = len(currentCapsules) - len(lastCapsules)  # 减少了 == 自己吃了胶囊
        E = currentFood.count() - lastFood.count()
        F = currentFoodDefending.count() - lastFoodDefending.count()  # 减少了 == 我们的食物被吃了, 增加了 ==  我方幽灵吃了对方吃豆人 || 放下了豆子
        G = currentScore - lastScore
        if currentGameState.getAgentState(self.index).isPacman:
            reward -= 100
        if E < 0:
            reward -= 10
        if currentGameState.getAgentPosition(self.index) == currentGameState.getInitialAgentPosition(self.index):
                self.atCenter = False
                self.ASTARPATH = self.getCenterPos(currentGameState)
                self.center_counter = 0
                reward -= 100  # 我方被吃回到原点

        # DIED
        if currentGameState.getAgentPosition(self.index) == currentGameState.getInitialAgentPosition(self.index):
            lastX = currentGameState.getAgentPosition(self.index)[0]
            lastY = currentGameState.getAgentPosition(self.index)[1]
            currentX = currentGameState.getAgentPosition(self.index)[0]
            currentY = currentGameState.getAgentPosition(self.index)[1]
            if not (lastX == currentX + 1 or lastX == currentX - 1) and not (
                    lastY == currentY + 1 or lastY == currentY - 1):
                reward += -100

        # ATE_PACMAN
        myPos = currentGameState.getAgentState(self.index).getPosition()
        myLastPos = lastGameState.getAgentState(self.index).getPosition()

        oldEnemies = [lastGameState.getAgentState(i) for i in self.getOpponents(lastGameState)]
        newEnemies = [currentGameState.getAgentState(i) for i in self.getOpponents(currentGameState)]
        oldPacmen = [a for a in oldEnemies if a.isPacman and a.getPosition() != None]
        newPacmen = [a for a in newEnemies if a.isPacman and a.getPosition() != None]
        if len(oldPacmen) > 0:
            dists = [self.getMazeDistance(lastGameState.getAgentState(self.index).getPosition(), a.getPosition()) for a
                     in oldPacmen]
            for one,two in zip(oldPacmen,newPacmen):
                if currentGameState.isOnRedTeam(self.index):
                    if self.getMazeDistance(myPos, two.getPosition()) < self.getMazeDistance(myLastPos, one.getPosition()):
                        reward += 20
                else:
                    if self.getMazeDistance(myPos, two.getPosition()) < self.getMazeDistance(myLastPos, one.getPosition()):
                        reward += 20
            if min(dists) == 1:
                if len(newPacmen) == 0 and currentGameState.getAgentState(
                        self.index).getPosition() != currentGameState.getInitialAgentPosition(self.index):
                    reward += 200
                elif len(newPacmen) > 0 and currentGameState.getAgentState(
                        self.index).getPosition() != currentGameState.getInitialAgentPosition(self.index):
                    dists = [self.getMazeDistance(currentGameState.getAgentState(self.index).getPosition(), a.getPosition())
                             for a in oldPacmen]
                    if min(dists) > 2:
                        reward += 200
        return reward
    def final(self, gameState):
        # Next
        self.ep_rew += self.last_reward
        # Do observation
        self.terminal = True
        self.observation_step(gameState)

        # Print stats
        log_file = open('./logs/' + str(self.general_record_time) + '-l-' + str(self.params['width']) + '-m-' + str(
            self.params['height']) + '-x-' + str(self.params['num_training']) + '.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                       (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()
        if self.params['TRAIN']:
            if self.params['num_training'] == self.numeps:
                # Save model
                if (self.params['save_file']):
                    self.qnet.save_ckpt(
                        'saves/model-' + self.params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')








