import time

class ActionHistory():
    def __init__(self):
      self.event_id = 0
      self.action_id = 0
      self.event = []
      self.action = []

    # new event: 
    # gripper known to be empty in neutral position and requiring new grasp
    # evaluation.
    def new_event(self):
      self.event.append(self.action)         # save previous action
      # ARD: TODO
      # save_event()
      self.action = [] 
      self.event_id += 1
      self.move_id = 0
      return self.event_id

    def new_action(self,action, subaction = None):
      self.event.append(self.action)         # save previous action
      self.action_id += 1
      self.action = []
      self.action.append(["ID", self.event_id, self.action_id, time.time()])
      self.action.append(["ACTION", action, subaction])
      return [self.event_id, self.action_id]

    def action_info(self,info):
      self.action.append(info)

    def get_action_id(self):
      return [self.event_id, self.action_id, self.action]

    def find_action(self, action, info = None):
      # always [["ID",...]["ACTION", ...][... data ...]]
      for e in range(2):
        evnt = self.event[self.event_id - e]
        for a in range(self.action_id):
          act = event[self.action_id - a][1]
          if len(act) == 0 or len(act) != len(action):
            continue
          found = True 
          for i in range(len(act)):
            if (action[i] != act[i]):
              found = False 
              break
          if found:
            act = event[self.action_id - a]
            for i in range[len(act)]:
              if info != None and act[i][0] == info:
                return act[i]
            if info != None:
              return None
            return act
      return None
   
    def save_event(self):
        with h5py.File(self.datapath + '/' + str(i) + '.hdf5', 'w') as file:
            for act in (event[self.event_id]):
              file[act[0]] = act
        print('Saved to %s' % self.datapath  + '/'+ str(i) + '.hdf5')
        # prune history
        for e in range(len(event) - HISTORY_NUM_EVENTS):
          if event[self.event_id - HISTORY_NUM_EVENTS - e] == None:
            break
          event[self.event_id - HISTORY_NUM_EVENTS - e] = None
