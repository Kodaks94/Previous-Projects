
import pickle
import collections
import numpy as np
from sklearn.metrics import confusion_matrix

class mod_OVBox(OVBox):


    def __init__(self):
        OVBox.__init__(self)
        
        self.time_predicted = 0
        self.gen_label = False
	#meanDetectTimeNewStim
        self.mean_dtns = 0
        self.num_T_class = 0
        self.is_debugged = False
        self.thresh_of_class = 0.5
        self.maximum_probility_diff_in_Thresh = 0.25
        self.debug_pred_nothing = False
        self.neg_class = 0

        self.time_stop_cl = 0
        self.time_start_cl = 0
        self.cl = -1
        
        deque_max_lenght = 1
        self.class_probablities = [collections.deque(maxlen=deque_max_lenght),
                           collections.deque(maxlen=deque_max_lenght),
                           collections.deque(maxlen=deque_max_lenght),
                           collections.deque(maxlen=deque_max_lenght)]

        self.labels_prob_act = []
        self.labels_prob_pred = []
        self.num_stims_class = 0
        self.num_stims_act = 0

        

    

       

        
    def initialize(self):
        thresh_string_list = self.setting['Thresholds'] 
        thresh_list = thresh_string_list.split(':')
        if thresh_list:
            self.maximum_probility_diff_in_Thresh = float(thresh_list[1])
            self.thresh_of_class = float(thresh_list[0])

    
    
	
    def add_to_matrix(self, num_input, num_class):
        probMatrix = self.input[num_input].pop()  # id_probMatrix = 0  stimulated, id_Matrix = 1 is non-stimulated
        self.class_probablities[num_class].append([probMatrix[0], self.getCurrentTime()])
        if probMatrix[0] > self.thresh_of_class:
            return True
        else:
            return False
		 
    def get_classification_probablity(self, inputNr, classNr):
        for i in range(len(self.input[inputNr])):
            if self.add_to_matrix( inputNr, classNr):
                return True 
            
        return False

		
    def calculate_new_attributes(self,stim):
        self.num_stims_act += 1
        self.time_start_cl = stim.date + 1.0
        self.cl = stim.identifier - 33024
        self.time_stop_cl = self.time_start_cl + 7.0
        self.gen_label = True
		
		
    def if_OVStimset(self,act_chunk):
        if type(act_chunk) == OVStimulationSet:
            for i in range(len(act_chunk)):
                stimulation = act_chunk.pop()
                if self.is_debugged:
                    print 'Received stim on input 7', stimulation.identifier - 33025, ' at', stimulation.date, 's'
                if (stimulation.identifier > 33024 and stimulation.identifier < 33030):
                    self.calculate_new_attributes(stimulation)
                    if self.is_debugged:
                        print 'Total classified: ' + str(self.num_T_class)
			
  
	
	
    def probablity_alter(self,all_probablities):
        max_probablity = max(all_probablities)
        max_position = all_probablities.index(max_probablity)
        del all_probablities[max_position]
        maxProb2 = max(all_probablities)
        maxPosDif = max_probablity - maxProb2
        if self.is_debugged:
            print "maxPosDif: " + str(maxPosDif) + "max_position: " + str(max_position)
        return max_probablity, max_position, maxPosDif
		
    def if_predict_condition(self,maxPosDif):
        if (self.cl is not -1) and  maxPosDif >= self.maximum_probility_diff_in_Thresh and (self.time_start_cl <= self.time_predicted) and (self.time_predicted <= self.time_stop_cl):
            return True
        return False
			
								
					
	
	
    def Hit_class_process(self,Hit_class, all_probablities):
        if True in Hit_class:  # Clear first threshold
            for i in range(4):
                for j in self.class_probablities[i]:
                    if j[0] > self.thresh_of_class:
                        all_probablities[i] += j[0]
            if self.is_debugged:
                print "all_probablities: " + str(all_probablities)

            max_probablity , max_position , maxPosDif = self.probablity_alter(all_probablities)

            self.time_predicted = self.getCurrentTime()

            if self.if_predict_condition(maxPosDif):
                self.num_T_class += 1
                self.labels_prob_pred.append(max_position + 1)
                self.labels_prob_act.append(self.cl)
                
                
                temp_mean = self.time_predicted - self.time_start_cl
                if self.gen_label:
                    self.mean_dtns += temp_mean
                    self.gen_label = False
                    self.num_stims_class += 1
                    if self.is_debugged:
                        print "number  of stims is " + str(self.num_stims_class)
                        print "predicted class was " + str(max_position + 1) + " label was: " + str(self.cl)
	# Process loop for classification

 # Mark as unclassified if not classified in allotted time
    def if_gen_lab(a,all_probablities):
        
        
        if  a.gen_label and (a.getCurrentTime() > a.time_stop_cl) : 
            a.neg_class += 1
            a.gen_label = False
			
            if a.is_debugged:
                print "couldnt classify!"

            if (a.cl is not -1) and a.debug_pred_nothing :  
                for i in range(4):
                    for j in a.class_probablities[i]:
                        all_probablities[i] += j[0]

                max_pos = all_probablities.index(max(all_probablities))

                a.labels_prob_act.append(self.cl)
                a.labels_prob_pred.append(max_pos + 1)
                a.num_stims_class += 1
                a.mean_dtns += 7
                a.num_T_class += 1
        return all_probablities

    def process(self):
        for i in range(len(self.input[7])):  #act process
            a = self.input[7].pop()
            self.if_OVStimset(a)
                
        all_probablities = [0.0, 0.0, 0.0, 0.0]
        all_probablities = self.if_gen_lab(all_probablities)

        Hit_class = [self.get_classification_probablity(inputNr=3, classNr=0),
                    self.get_classification_probablity(inputNr=4, classNr=1),
                    self.get_classification_probablity(inputNr=5, classNr=2),
                    self.get_classification_probablity(inputNr=6, classNr=3)]
        self.Hit_class_process(Hit_class,all_probablities)

            

            
        return

    
        
	
    def gather_data_to_write(self):
        file_name = self.setting['Current Subject Nr']
        path_file = self.setting['Results Directory']
        num_subjs = int(self.setting['Nr of subjects'])
       
       
        list_settings = {'ch': self.setting['channels'],
                      'epDur': self.setting['Epoch Duration'],
                      'epInt': self.setting['Epoch Interval'],
                      'freqTol': self.setting['Freq Tol'],
                      'simFreq': self.setting['SimulationFreq'],
                      'num_subjsects': num_subjs,
                      'currentSubjNr': file_name,
                      'classTresh': self.thresh_of_class,
                      'maximum_probility_diff_in_Thresh': self.maximum_probility_diff_in_Thresh}
        return file_name, path_file, num_subjs, list_settings
		
    def write_into_file(self,file_name, path_file, num_subjs, list_settings):
        settingsFileName = ''
        for key, value in list_settings.items():
            if (key != 'simFreq'):
                settingsFileName += str(value)[2:]

        writeFile = path_file + 'prev' + settingsFileName + 'subject' + str(file_name)
        data = {'settings': list_settings,
                'actual': self.labels_prob_act,
                'predicted': self.labels_prob_pred,
                'detectTime': self.mean_dtns,
                'stims': {'stimsNrClassified': self.num_stims_class,
                          'stimsNrActual': self.num_stims_act,
                          'num_T_class': self.num_T_class,
                          'totalTime': self.num_stims_act * 7,
                          'neg_class': self.neg_class}
                }

        pickle.dump(data, open(writeFile, 'wb'))
        print 'File saved to ' + writeFile
		
    # Write results to a file
    def uninitialize(self):
        if self.num_stims_class != 0:
            self.mean_dtns /= self.num_stims_class
            print "mean time for first classification: " + str(self.mean_dtns)
        cmSklearn = confusion_matrix(self.labels_prob_act, self.labels_prob_pred)
        cmSklearn = cmSklearn.astype('float') / cmSklearn.sum(axis=1)[:, np.newaxis]
        print "Confusion matrix -------------------------------"
        print cmSklearn
        returned = confusion_matrix(self.labels_prob_act, self.labels_prob_pred).ravel()
        F1 = returned[0:4]
        F2 = returned[4:8]
        F3 = returned[8:12]
        F4 = returned[12:16]
        ACC = [F1,F2,F3,F4]
        i = [20,15,12,10]
        j = 0
        for v in ACC:
            print("Frequency:",i[j])
            j +=1 
            tn, fp, fn,tp = v
            print("True Negatives: ",tn)
            print("False Positives: ",fp)
            print("False Negatives: ",fn)
            print("True Positives: ",tp)
            Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
            print(("Accuracy {:0.2f}%:").format(Accuracy))
            Precision = tp/(tp+fp) 
            print(("Precision {:0.2f}").format(Precision))
            Recall = tp/(tp+fn) 
            print(("Recall {:0.2f}").format(Recall))
            f1 = (2*Precision*Recall)/(Precision + Recall)
            print("F1 Score {:0.2f}".format(f1))
            Specificity = tn/(tn+fp)
            print("Specificity {:0.2f}".format(Specificity))
        file_name, path_file, num_subjs, list_settings= self.gather_data_to_write()
        self.write_into_file(file_name, path_file, num_subjs, list_settings)
        

box = mod_OVBox()
