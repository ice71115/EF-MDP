# -*- coding: UTF-8 -*-
import csv
import sys
import numpy as np
import copy
Transitions = {}
Cohort = {}
Reward = {}
c=[6,3,2,3,2.5]
user_act=1 #0:sleep 1:watch TV
sensor_number=[1,1,1,1,1]
r=0.3

all_sensor_number=np.sum(sensor_number)
#gamma is the discount factor
if len(sys.argv)>1:
    gamma = float(sys.argv[3])
else:
    gamma = 0.2
sleep_sensor=sensor_number[1]+sensor_number[2]+sensor_number[3]
wTV_sensor=sensor_number[4]+sensor_number[3]+sensor_number[1]
#the maximum error allowed in the utility of any state
if len(sys.argv)>2:
    epsilon = float(sys.argv[4])
else:
    epsilon = 0.001

def read_MDP():
	val=['nor','on_','off']
	all_states=[]
	for i in range(3**all_sensor_number):
		v=i
		state=[]
		for j in range(all_sensor_number):
			state.append(v%3)
			v=int(v/3)
		all_states.append(state)
	#print(all_states)
	
	for ptr_state in all_states:
		act={}
		Nact={}
		#######
		#sleep#
		#######
		regret=0
		s2=0
		for i in range(sensor_number[2-1]):
			if ptr_state[sensor_number[0]+i]==1:
				s2+=0.3/sleep_sensor  #night
			elif ptr_state[sensor_number[0]+i]==2:
				regret=0.1
		s3=0
		for i in range(sensor_number[3-1]):
			if ptr_state[sensor_number[0]+sensor_number[1]+i]==1:
				s3+=1.0/sleep_sensor  #bed
			elif ptr_state[sensor_number[0]+sensor_number[1]+i]==2:
				regret=0.1
		s4=0
		for i in range(sensor_number[4-1]):
			if ptr_state[sensor_number[0]+sensor_number[1]+sensor_number[2]+i]==1:
				s4+=1.0/sleep_sensor  #card
			elif ptr_state[sensor_number[0]+sensor_number[1]+sensor_number[2]+i]==2:
				regret=0.1
		clock=0
		for i in range(sensor_number[1-1]):
			if ptr_state[i]==1:
				clock+=1.0/sensor_number[0]  #clock
			
		#regret=0
		belief_sleep=clock*0.5+(s2+s3+s4)*0.5 -regret if clock!=0 else s2+s3+s4-regret 
		#########
		#watchTV#
		#########
		
		regret=0
		s5=0
		for i in range(sensor_number[5-1]):
			if ptr_state[sensor_number[0]+sensor_number[1]+sensor_number[2]+sensor_number[3]+i]==1:
				s5+=1.0/wTV_sensor  #TV
			elif ptr_state[sensor_number[0]+sensor_number[1]+sensor_number[2]+sensor_number[3]+i]==2:
				regret=0.1
				
		s6=0
		for i in range(sensor_number[4-1]):
			if ptr_state[sensor_number[0]+sensor_number[1]+sensor_number[2]+i]==1:
				s6+=1.0/wTV_sensor  #card
			elif ptr_state[sensor_number[0]+sensor_number[1]+sensor_number[2]+i]==2:
				regret=0.1
				
		s7=0
		for i in range(sensor_number[2-1]):
			if ptr_state[sensor_number[0]+i]==1:
				s7+=0.7/wTV_sensor  #day
			elif ptr_state[sensor_number[0]+i]==2:
				regret=0.1
				

		#regret=0
		belief_wTV=s5+s6+s7-regret
		#print(ptr_state)
		#print(s2,s3,s4,s5,s6,s7)
		#print(belief_sleep,belief_wTV)
		#choose Turn on sensor action
		state_cost=0
		tag=0
		for t in range(5):
			for i in range(sensor_number[t]):
				
				if ptr_state[tag]!=0:
					state_cost+=c[t]
				tag+=1
		state_error=(np.dot(c,sensor_number)-state_cost)/np.dot(c,sensor_number)
		
		#state_error*=2
		#w1=0.5
		#w2=1-w1
		belief=0
		o_sensor=[]
		wTV_LB=0.4
		sleep_LB=0.4
		min_LB=min(wTV_LB,sleep_LB)
		if belief_sleep>belief_wTV:
			belief=belief_sleep
			if belief <sleep_LB:
				belief =0
			else:
				(belief-sleep_LB)/(1-sleep_LB)
			o_sensor=[0,1,2,3]
		elif(belief_sleep<belief_wTV):
			belief=belief_wTV
			if belief <wTV_LB:
				belief =0
			else:
				(belief-wTV_LB)/(1-wTV_LB)
			o_sensor=[4,3,1]
		else:
			belief=belief_sleep
			if belief <min_LB:
				belief =0
			else:
				(belief-min_LB)/(1-min_LB)
			o_sensor=[0,1,2,3,4]
			
		tag=0
		for t in range(5):
			for i in range(sensor_number[t]):
				if ptr_state[tag]==0 and t in o_sensor:
					on=ptr_state[:]
					off=ptr_state[:]
					on[tag]=1
					off[tag]=2
					p_act='Turn on Sensor'+str(tag+1)
					act[p_act]=[(0.5,"("+" ".join(str(ele) for ele in on)+")"),(0.5,"("+" ".join(str(ele) for ele in off)+")")]
					Nact[p_act]=[[1,"("+" ".join(str(ele) for ele in on)+")"],[1,"("+" ".join(str(ele) for ele in off)+")"]]
				tag+=1
		#print(act)
		#state reward
		

		#choose Turn off sensor action
		if belief >0:
			tag=0
			for t in range(5):
				for i in range(sensor_number[t]):
					if ptr_state[tag]!=0:
						nor=ptr_state[:]
						nor[tag]=0
						p_act='Turn off Sensor'+str(tag+1)
						act[p_act]=[(1,"("+" ".join(str(ele) for ele in nor)+")")]
						Nact[p_act]=[[1,"("+" ".join(str(ele) for ele in nor)+")"]]
					tag+=1


		#state reward
		p_state="("+" ".join(str(ele) for ele in ptr_state)+")"
		#Reward[p_state]=((state_error*0.13)+(0.87*belief)) if state_error*belief!=0 else 0
		Reward[p_state]=state_error*belief
		#keep action

		turn_off_sensor=[]
		tag=0
		for t in range(5):
			for i in range(sensor_number[t]):
				if ptr_state[tag] ==0:
					turn_off_sensor.append(tag)
				tag+=1
		p = 1 if len(turn_off_sensor)<all_sensor_number else 1
		p_act='keep'
		act[p_act]=[(p,p_state)]
		Nact[p_act]=[[1,p_state]]
		if p!=1:
			if len(turn_off_sensor)!=all_sensor_number:
				for state in all_states:
					turn_off=[]
					for t in range(all_sensor_number):
						if state[t] ==0:
							turn_off.append(t)
					#s1=set(turn_off_sensor)
					#s2=set(turn_off)
					if turn_off_sensor==turn_off :
						if ptr_state!=state:
							act[p_act].append(((1-p)/(2**(all_sensor_number-len(turn_off))-1),"("+" ".join(str(ele) for ele in state)+")"))
		

					
		Transitions[p_state]=act
		Cohort[p_state]=Nact
		#print("("+" ".join(str(ele) for ele in ptr_state)+")")
		


read_MDP()
#print(Cohort['(0 0 0 0 0)']['keep'][0][1])

class MarkovDecisionProcess:

    """A Markov Decision Process, defined by an states, actions, transition model and reward function."""

    def __init__(self, transition={}, reward={}, gamma=.9):
        #collect all nodes from the transition models
        self.states = transition.keys()
        #initialize transition
        self.transition = transition
        #initialize reward
        self.reward = reward
        #initialize gamma
        self.gamma = gamma

    def R(self, state):
        """return reward for this state."""
        return self.reward[state]

    def actions(self, state):
        """return set of actions that can be performed in this state"""
        return self.transition[state].keys()

    def T(self, state, action):
        """for a state and an action, return a list of (probability, result-state) pairs."""
        return self.transition[state][action]

#Initialize the MarkovDecisionProcess object
#mdp = MarkovDecisionProcess(transition=Transitions, reward=Reward)

def value_iteration():
    """
    Solving the MDP by value iteration.
    returns utility values for states after convergence
    """
    states = mdp.states
    actions = mdp.actions
    T = mdp.T
    R = mdp.R

    #initialize value of all the states to 0 (this is k=0 case)
    V1 = {s: 0 for s in states}
    while True:
        V = V1.copy()
        delta = 0
        for s in states:
            #Bellman update, update the utility values
            V1[s] = R(s) + gamma * max([ sum([p * V[s1] for (p, s1) in T(s, a)]) for a in actions(s)])
            #calculate maximum difference in value
            delta = max(delta, abs(V1[s] - V[s]))

        #check for convergence, if values converged then return V
        if delta < epsilon * (1 - gamma) / gamma:
            return V


def best_policy(V):
    """
    Given an MDP and a utility values V, determine the best policy as a mapping from state to action.
    returns policies which is dictionary of the form {state1: action1, state2: action2}
    """
    states = mdp.states
    actions = mdp.actions
    pi = {}
    for s in states:
        pi[s] = max(actions(s), key=lambda a: expected_utility(a, s, V))
    return pi


def expected_utility(a, s, V):
    """returns the expected utility of doing a in state s, according to the MDP and V."""
    T = mdp.T
    return sum([p * V[s1] for (p, s1) in mdp.T(s, a)])


import random

#sensor_r=range(5)
#for i in range(5):
#	sensor_r[i]=random.choice([0.1,0.15,0.2,0.3])
curr_state="("
for i in range(all_sensor_number):
	curr_state=curr_state+"0"
	if i!=all_sensor_number-1:
		curr_state=curr_state+" "
curr_state=curr_state+")"
#(0 0 .... 0 0) start

#save pi
#pin=np.array(pi)
#np.save('pi.npy',pin)

print(curr_state)
# def env(state,action):
	# s=np.zeros(5)
	# invert=2
	# pick_sensor=1
	# if(user_act==0):#sleep
		# s[0]=random.randint(1, 2)
		# s[1]=1
		# s[2]=1
		# s[3]=1
		# s[4]=2
	# else:#wTV
		# s[4]=1
		# s[3]=1
		# s[1]=1
		# s[0]=2
		# s[2]=2
	# if "Turn" in action:
		# pick_sensor=int(action[-1])
		# invert=1 if s[pick_sensor-1]==2 else 2
	
	# if "Turn on" in action:
		# state=state[:pick_sensor*2-1]+"1"+state[pick_sensor*2:]#1 3 5 7 9
		# for j in range(len(state)):
			# i=(j-1)/2
			# if state[j]=="1" or state[j]=="2":
				# invert=1 if s[i]==2 else 2
				# state=state[:j]+str(int(np.random.choice([s[i],invert],size=1,p=[1-sensor_r[i],sensor_r[i]])[0]))+state[j+1:]
	# elif "Turn off" in action:
		# state=state[:pick_sensor*2-1]+"0"+state[pick_sensor*2:]
		# for j in range(len(state)):
			# i=(j-1)/2
			# if state[j]=="1" or state[j]=="2":
				# invert=1 if s[i]==2 else 2
				# state=state[:j]+str(int(np.random.choice([s[i],invert],size=1,p=[1-sensor_r[i],sensor_r[i]])[0]))+state[j+1:]
	# elif "keep" in action:
		# for j in range(len(state)):
			# i=(j-1)/2
			# if state[j]=="1" or state[j]=="2":
				# invert=1 if s[i]==2 else 2
				# state=state[:j]+str(int(np.random.choice([s[i],invert],size=1,p=[1-sensor_r[i],sensor_r[i]])[0]))+state[j+1:]
	# return state

f = open('./data/value/value r '+str(r)+'.txt','r')
real_sensor=f.readline()
fla = open('./data/label/label r '+str(r)+'.txt','r')
label=fla.readline()

# f = open('./data/value/value train.txt','r')
# real_sensor=f.readline()
# fla = open('./data/label/label train.txt','r')
# label=fla.readline()

def env(state,action):

	real_sensor=f.readline()
	#print(real_sensor)
	#state="(1 1 1 1 1)"
	pick_sensor=1
	
	if "Turn" in action:
		pick_sensor=int(action[-1])
		
	if "Turn on" in action:
		next_state=state[:pick_sensor*2-1]+"1"+state[pick_sensor*2:]#1 3 5 7 9

	elif "Turn off" in action:
		next_state=state[:pick_sensor*2-1]+"0"+state[pick_sensor*2:]
	else:
		next_state=state
	for j in range(len(next_state)):
		if next_state[j]=="1" or next_state[j]=="2":
			next_state=next_state[:j]+real_sensor[j-1]+next_state[j+1:]
#	elif "keep" in action:
	flag=False
	for i in range(len(Cohort[state][action])):
		if Cohort[state][action][i][1]==next_state:
			#print(Cohort[state][action][i])
			Cohort[state][action][i][0]+=1
			flag=True
	if flag==False:
		
		Cohort[state][action].append([1,next_state])
	return next_state
	
	

for	t in range(0,86400-1):
	#print("Phase : "+str(t)+"  state : "+curr_state+"   action : "+pi[curr_state])
	belief_sleep=0
	belief_wTV=0
	#######
	#sleep#
	#######
	s2=0.33*0.3 if curr_state[1*2+1]=='1' else 0 #night
	s3=0.33 if curr_state[2*2+1]=='1' else 0 #bed
	s4=0.33 if curr_state[3*2+1]=='1' else 0 #card
	belief_sleep=0.5+(s2+s3+s4)*0.5 if curr_state[0*2+1]=='1' else s2+s3+s4 
	#########
	#watchTV#
	#########
	s5=0.33 if curr_state[4*2+1]=='1' else 0 #TV
	s6=0.33 if curr_state[3*2+1]=='1' else 0 #card
	s7=0.33*0.7 if curr_state[1*2+1]=='1' else 0 #day
	belief_wTV=s5+s6+s7
	#######
	#sleep#
	#######

	s2=0
	for i in range(sensor_number[2-1]):
		if curr_state[(sensor_number[0]+i)*2+1]=="1":
			s2+=0.3/sleep_sensor  #night

	s3=0
	for i in range(sensor_number[3-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+i)*2+1]=="1":
			s3+=1.0/sleep_sensor  #bed

	s4=0
	for i in range(sensor_number[4-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+sensor_number[2]+i)*2+1]=="1":
			s4+=1.0/sleep_sensor  #card

	clock=0
	for i in range(sensor_number[1-1]):
		if curr_state[(i)*2+1]=="1":
			clock+=1.0/sensor_number[0]  #clock
		

	belief_sleep=clock*0.5+(s2+s3+s4)*0.5 if clock!=0 else s2+s3+s4
	#########
	#watchTV#
	#########
	

	s5=0
	for i in range(sensor_number[5-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+sensor_number[2]+sensor_number[3]+i)*2+1]=="1":
			s5+=1.0/wTV_sensor  #TV

			
	s6=0
	for i in range(sensor_number[4-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+sensor_number[2]+i)*2+1]=="1":
			s6+=1.0/wTV_sensor  #card

			
	s7=0
	for i in range(sensor_number[2-1]):
		if curr_state[(sensor_number[0]+i)*2+1]=="1":
			s7+=0.7/wTV_sensor  #day

			

	#regret=0
	belief_wTV=s5+s6+s7
	#print('識別居民行為 :'),
	# if belief_sleep<belief_wTV and belief_wTV>0.4:
		# #print("watch TV")
		# if "watch TV"in label:
			# hit+=1

	# elif(belief_sleep>0.4):
		# #print("sleep")
		# if "sleep"in label:
			# hit+=1
	# else:
		# print("uncertain")

	#print(en)
	if t%86398==0:
		Transitions=copy.deepcopy(Cohort)
		for s in Cohort:
			for a in Cohort[s]:
				count=0
				for ns_list in Cohort[s][a]:
					count+=ns_list[0]
				for ns in range(len(Cohort[s][a])):
					Transitions[s][a][ns]=(Transitions[s][a][ns][0]/count,Cohort[s][a][ns][1])
		mdp = MarkovDecisionProcess(transition=Transitions, reward=Reward)
		V = value_iteration()
		pi = best_policy(V)
	curr_state=env(curr_state,pi[curr_state])	
	label=fla.readline()
	#print(label)
curr_state="("
for i in range(all_sensor_number):
	curr_state=curr_state+"0"
	if i!=all_sensor_number-1:
		curr_state=curr_state+" "
curr_state=curr_state+")"
f = open('./data/value/value r '+str(r)+' test.txt','r')
real_sensor=f.readline()
fla = open('./data/label/label r '+str(r)+' test.txt','r')
label=fla.readline()

# f = open('./data/value/value test.txt','r')
# real_sensor=f.readline()
# fla = open('./data/label/label test.txt','r')
# label=fla.readline()

hit=0
all_en=0
for	t in range(0,86400-1):
	#print("Phase : "+str(t)+"  state : "+curr_state+"   action : "+pi[curr_state])
	belief_sleep=0
	belief_wTV=0
	#######
	#sleep#
	#######
	s2=0.33*0.3 if curr_state[1*2+1]=='1' else 0 #night
	s3=0.33 if curr_state[2*2+1]=='1' else 0 #bed
	s4=0.33 if curr_state[3*2+1]=='1' else 0 #card
	belief_sleep=0.5+(s2+s3+s4)*0.5 if curr_state[0*2+1]=='1' else s2+s3+s4 
	#########
	#watchTV#
	#########
	s5=0.33 if curr_state[4*2+1]=='1' else 0 #TV
	s6=0.33 if curr_state[3*2+1]=='1' else 0 #card
	s7=0.33*0.7 if curr_state[1*2+1]=='1' else 0 #day
	belief_wTV=s5+s6+s7
	#######
	#sleep#
	#######

	s2=0
	for i in range(sensor_number[2-1]):
		if curr_state[(sensor_number[0]+i)*2+1]=="1":
			s2+=0.3/sleep_sensor  #night

	s3=0
	for i in range(sensor_number[3-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+i)*2+1]=="1":
			s3+=1.0/sleep_sensor  #bed

	s4=0
	for i in range(sensor_number[4-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+sensor_number[2]+i)*2+1]=="1":
			s4+=1.0/sleep_sensor  #card

	clock=0
	for i in range(sensor_number[1-1]):
		if curr_state[(i)*2+1]=="1":
			clock+=1.0/sensor_number[0]  #clock
		

	belief_sleep=clock*0.5+(s2+s3+s4)*0.5 if clock!=0 else s2+s3+s4
	#########
	#watchTV#
	#########
	

	s5=0
	for i in range(sensor_number[5-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+sensor_number[2]+sensor_number[3]+i)*2+1]=="1":
			s5+=1.0/wTV_sensor  #TV

			
	s6=0
	for i in range(sensor_number[4-1]):
		if curr_state[(sensor_number[0]+sensor_number[1]+sensor_number[2]+i)*2+1]=="1":
			s6+=1.0/wTV_sensor  #card

			
	s7=0
	for i in range(sensor_number[2-1]):
		if curr_state[(sensor_number[0]+i)*2+1]=="1":
			s7+=0.7/wTV_sensor  #day

			

	#regret=0
	belief_wTV=s5+s6+s7
	#print('識別居民行為 :'),
	# if belief_sleep<belief_wTV and belief_wTV>0.4:
		# #print("watch TV")
		# if "watch TV"in label:
			# hit+=1

	# elif(belief_sleep>0.4):
		# #print("sleep")
		# if "sleep"in label:
			# hit+=1
	# else:
		# print("uncertain")
	if belief_sleep<belief_wTV:
		#print("watch TV")
		if "watch TV"in label:
			hit+=1
			#print("T watch TV")
		#else:
			#print(label)

	else:
		#print("sleep")
		if "sleep"in label:
			hit+=1
			#print("T sleep")
		#else:
			#print(label)
	#print(curr_state)
	en=0
	tag=0
	for i in range(5):
		for j in range(sensor_number[i]):
			if curr_state[tag*2+1]=='1' or curr_state[tag*2+1]=='2':
				en+=c[i]
			tag+=1
	all_en+=en
	#print(en)
	if t%86398==0:
		Transitions=copy.deepcopy(Cohort)
		for s in Cohort:
			for a in Cohort[s]:
				count=0
				for ns_list in Cohort[s][a]:
					count+=ns_list[0]
				for ns in range(len(Cohort[s][a])):
					Transitions[s][a][ns]=(Transitions[s][a][ns][0]/count,Cohort[s][a][ns][1])
		mdp = MarkovDecisionProcess(transition=Transitions, reward=Reward)
		V = value_iteration()
		pi = best_policy(V)
	#print('action:'+pi[curr_state])
	curr_state=env(curr_state,pi[curr_state])	
	label=fla.readline()
	#print(label)
print(hit/86400.0)
print(all_en/(86400*np.dot(c,sensor_number)))	

#call value iteration
"""

print '    State    -     Value'
for s in V:
    print s, ' - ' , V[s]
pi = best_policy(V)
print '\nOptimal policy is \n   State     -     Action'
for s in pi:
    print s, ' - ' , pi[s]
"""
