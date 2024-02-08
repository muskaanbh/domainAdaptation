import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 2e-5
epochs = 5  #initial train epochs
outer_epochs = 10 #How many times we perform the pseudo labeling
inner_epochs = 3 #Epochs for each pseudo labeling iteration
train_batch_size = 16
valid_batch_size = 16
test_batch_size = 16
unlabeled_batch_size = 16

#Whether to use KD or not
use_kd = False
#Other KD params, these won't be used if use_kd = False
Temp = 1
kd_alpha = 0.5

dropout_rate = 0.3
class_num = 2
bidirectional = True

#Whether to use data aug and what is the ratio
noise_injection = True
noise_injection_rate = 0.4

# Define hyperparameters
train_params = {'batch_size': train_batch_size,
                'shuffle':True,
                'num_workers':0
               }
valid_params = {'batch_size':valid_batch_size,
                'shuffle':False,
                'num_workers':0
               }
test_params = {'batch_size':test_batch_size,
                'shuffle':False,
                'num_workers':0
               }
unlabeled_params = {'batch_size':unlabeled_batch_size,
                    'shuffle':True,
                    'num_workers':0
                   }
