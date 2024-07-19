import collections

# CONSTANTS
Max_len = 1
probsAll = [0.0, 0.0, 0.0, 0.0]
number_of_stims = 4
	
class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
      
		
        self.prob_classes = [collections.deque(maxlen=Max_len),
                           collections.deque(maxlen=Max_len),
                           collections.deque(maxlen=Max_len),
                           collections.deque(maxlen=Max_len)]

	
    def get_thresholds(self,that):
        Thresholds = self.setting['Thresholds'].split(':')
	if Thresholds:
            self.maxProbDiffThresh = float(Thresholds[1])
            self.classThresh = float(Thresholds[0])
	else:
	    self.classThresh = 0.5
	    self.maxProbDiffThresh = 0.25
		
	
    def initialize(self):
        self.get_thresholds(self)
        

   
    def get_prob(self, num_input, num_class):
	size  = len(self.input[num_input])
        for i in range(size):
            pm = self.input[num_input].pop()
	    current_time = self.getCurrentTime()
	    current_pm = pm[0]
            self.prob_classes[num_class].append([current_pm, current_time])
            if pm[0] > self.classThresh:
                return True
        return False

    def clear_thresh(self, hits, sent_probs):
        if True in hits:
            for stim in range(number_of_stims):
		for p in self.prob_classes[stim]:
                    state = p[0];
		    if state > self.classThresh:
                        sent_probs[stim] += state
			
	    mp = max(sent_probs)
	    mps = sent_probs.index(mp)
	    del sent_probs[mps]
	    mp2 = max(sent_probs)
	    diff = mp - mp2
	    if(diff >= self.maxProbDiffThresh):
                self.tcp_writer(mps)
		
	return
	
    def tcp_writer(self,mps):
        stimul_set = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime() + 1. / self.getClock())
        stimul_set.append(OVStimulation(33025 + mps, self.getCurrentTime(), 0.))
        self.output[0].append(stimul_set)  
        return
    def process(self):
        probsAll = [0.0, 0.0, 0.0, 0.0]
        hit_inClass = [self.get_prob(num_input=3, num_class=0),
                    self.get_prob(num_input=4, num_class=1),
                    self.get_prob(num_input=5, num_class=2),
                    self.get_prob(num_input=6, num_class=3)]

        self.clear_thresh(hit_inClass, probsAll)
		
	return

    def uninitialize(self):
        return

box = MyOVBox()
