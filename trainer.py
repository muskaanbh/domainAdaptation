import os
import torch
from evaluator import Evaluator
from torch.utils.data import  DataLoader
from util.early_stopping import EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification
from util.dataset import Dataset
from util.criterion import kd_loss
import constant as config
import torch.nn.functional as F 


class Trainer(object):
    def __init__(self, config, model, optimizer, save_path, dev_dataset, test_dataset, alpha=0, temp=1):
        self.config = config
        self.optimizer = optimizer
        self.device = self.config.device
        self.alpha = alpha
        self.temp = temp

        # self.loss = criterion.to(self.device)
        self.evaluator = Evaluator(batch_size=self.config.test_batch_size, alpha=self.alpha, temp=self.temp)
        self.model = model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.train_loader = None
        
        self.valid_loader = DataLoader(dev_dataset, **self.config.valid_params)
        self.test_loader = DataLoader(test_dataset, **self.config.test_params)
        
        self.early_stopping = None
        
        self.save_path = save_path
        self.sup_path = self.save_path +'/supervised'
        self.ssl_path = self.save_path +'/semisupervised'
        
        if not os.path.isabs(self.sup_path):
            self.sup_path = os.path.join(os.getcwd(), self.sup_path)
        if not os.path.exists(self.sup_path):
            os.makedirs(self.sup_path)
        
        if not os.path.isabs(self.ssl_path):
            self.ssl_path = os.path.join(os.getcwd(), self.ssl_path)
        if not os.path.exists(self.ssl_path):
            os.makedirs(self.ssl_path)

        
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
        
    def train_epoch(self, epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        print('train_epoch', epoch)
        
        for _, batch in enumerate(self.train_loader):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)

            targets = batch['labels'].to(self.device, dtype=torch.float)
            outputs = self.model(ids, attention_mask, token_type_ids)
            loss = kd_loss(outputs,targets,self.alpha,self.temp)
            
            tr_loss += loss.item()
            scores = torch.softmax(outputs.data, dim=-1)
            big_val, big_idx = torch.max(scores.data, dim=-1)
            _,big_idx_target = torch.max(targets.data,dim=-1)
            n_correct += self.calculate_accu(big_idx,big_idx_target)
            nb_tr_steps += 1
            nb_tr_examples += big_idx_target.size(0)
            
            # if _ % 1000 == 0:
            #     loss_step = tr_loss/nb_tr_steps
            #     accu_step = (n_correct*100)/nb_tr_examples
            #     print(f"Training Loss per 1000 steps: {loss_step}")
            #     print(f"Training Accuracy per 1000 steps: {accu_step}")
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        
        
    def initial_train(self, label_dataset):
        print('initial train module')
        self.train_loader = DataLoader(label_dataset, **self.config.train_params)
        self.early_stopping = EarlyStopping(patience=3, verbose=True)
        min_dev_loss = 987654321
        
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch)
            dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.valid_loader)
            self.early_stopping(dev_loss)
            
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                torch.save({'model_state_dict':self.model.state_dict(),
                            'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch}, self.sup_path +'/checkpoint.pt')
                
            if epoch % 3 == 0:
                test_loss, test_acc = self.evaluator.evaluate(self.model, self.test_loader, is_test=True)
            
            if self.early_stopping.early_stop:
                print("Eearly Stopping!")
                break

                
    def self_train(self, labeled_dataset, unlabeled_dataset, confidence_threshold=0.8):
        best_accuracy = -1
        min_dev_loss = 987654321
        
        for outer_epoch in range(self.config.outer_epochs):
            # pseudo-labeling
            new_dataset = self.pseudo_labeling(unlabeled_dataset, confidence_threshold)
            
            # update dataset (remove pseudo-labeled from unlabeled dataset and add them into labeled dataset)
            labeled_dataset, unlabeled_dataset = self.update_dataset(labeled_dataset, unlabeled_dataset, new_dataset)

            #TODO:
            #Here we can add the data aug part with a ratio (does not need to augment the entire dataset since it is time-consuming)
            #Look at constant.py, set noise_injection == True to decide whether to use data aug and if yes, what is the ratio
            #After generate the augmented data, store it seperately and only use it for the belowed student model training, do NOT overwrite the original "labeled_dataset"

            self.train_loader = DataLoader(labeled_dataset, **self.config.train_params)
            self.early_stopping = EarlyStopping(patience=3, verbose=True)
            
            #TODO:
            #We can do "del self.model" then initialize self.model = LARGER_BERT and so on

            # retrain model with labeled data + pseudo-labeled data
            for inner_epoch in range(self.config.inner_epochs):
                print('outer_epoch {} inner_epoch {}'.format(outer_epoch, inner_epoch))
                self.train_epoch(inner_epoch)
                dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.valid_loader)
                self.early_stopping(dev_loss)
                
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    torch.save({'model_state_dict':self.model.state_dict(),
                                'optimizer_state_dict':self.optimizer.state_dict(), 'epoch':inner_epoch}, self.ssl_path +'/checkpoint.pt')
                    print("Saving..")
                
                if inner_epoch % 2 == 0:
                    test_loss, test_acc = self.evaluator.evaluate(self.model, self.test_loader, is_test=True)
                    if best_accuracy < test_acc:
                        best_accuracy = test_acc
                        
                if self.early_stopping.early_stop:
                    print("Early Stopping!")
                    break
                    
        print('Best accuracy {}'.format(best_accuracy))
    
    
    def pseudo_labeling(self, unlabeled_dataset, confidence_threshold):
        unlabeled_loader = DataLoader(unlabeled_dataset, **self.config.unlabeled_params)
        self.model.eval()
        new_dataset = {label:[] for label in range(self.config.class_num)}
        
        with torch.no_grad():
            print("Starting pseudo labeling")
            for _, batch in enumerate(unlabeled_loader):
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                targets = batch['labels'].to(self.device, dtype=torch.float)
                
                outputs = self.model(ids, attention_mask, token_type_ids)
                loss = kd_loss(outputs,targets,self.config.kd_alpha,self.config.Temp)
                confidences = torch.softmax(outputs.data, dim=-1)
                big_val, big_idx = torch.max(confidences.data, dim=-1)

                for text_id, label, conf_val, soft_pred, target in zip(ids, big_idx, big_val, confidences.data, targets):
                    pred_label, conf_val, target = label.item(), conf_val.item(), target
                    if not config.use_kd:
                        soft_pred = F.one_hot(torch.argmax(soft_pred), num_classes=2)
                    if conf_val >= confidence_threshold:
                        new_dataset[pred_label].append((text_id, pred_label, soft_pred, target))
        
        num_of_min_dataset = 987654321
        for label, dataset in new_dataset.items():
            if num_of_min_dataset > len(dataset):
                num_of_min_dataset = len(dataset)
                
        for label in new_dataset.keys():
            new_dataset[label] = new_dataset[label][:num_of_min_dataset]
            
        balanced_dataset = []
        for label in new_dataset.keys():
            balanced_dataset.extend(new_dataset[label][:num_of_min_dataset])

        return balanced_dataset

    
    def update_dataset(self, labeled_dataset, unlabeled_dataset, new_dataset):
        '''
        @param new_dataset type list(tuple(text_ids, pred_label, target_label))
        '''
        new_texts = []
        new_labels = []
        for idx in range(len(new_dataset)):
            text_id = new_dataset[idx][0]
            pred_label = new_dataset[idx][2]
            decoded_text = self.tokenizer.decode(text_id)
            decoded_text = decoded_text.replace("[CLS]", "").replace("[SEP]","").replace("[PAD]","").strip()
            new_texts.append(decoded_text)
            new_labels.append(pred_label)
        
        labeled_texts, labeled_labels = self.decode_dataset(labeled_dataset)
        unlabeled_texts, unlabeled_labels = self.decode_dataset(unlabeled_dataset)
        print('labeled {} unlabeled {}'.format(len(labeled_texts), len(unlabeled_texts)))
        print('pseudo-labeled', len(new_texts))
        
        # add pseudo_labeled into labeled dataset
        labeled_texts.extend(new_texts)
        labeled_labels.extend(new_labels)
        
        # remove pseudo-labeled from unlabeled dataset
        for text in new_texts:
            idx = unlabeled_texts.index(text)
            unlabeled_texts.pop(idx)
            unlabeled_labels.pop(idx)
        print('After updated -> labeled {} unlabeled {}'.format(len(labeled_texts), len(unlabeled_texts)))
        
        # encodings -> make dataset
        labeled_dataset = self.encode_dataset(labeled_texts, labeled_labels)
        unlabeled_dataset = self.encode_dataset(unlabeled_texts, unlabeled_labels)
        return labeled_dataset, unlabeled_dataset
        
        
    def encode_dataset(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        labels = [label.tolist() for label in labels]
        dataset = Dataset(encodings, labels)
        labels = torch.tensor(labels)
        return dataset
    
    
    def decode_dataset(self, dataset):
        decoded_texts = []
        labels = []
        for idx in range(len(dataset)):
            text_id = dataset[idx]['input_ids']
            label = dataset[idx]['labels']
            decoded_text = self.tokenizer.decode(text_id)
            decoded_text = decoded_text.replace("[CLS]", "").replace("[SEP]","").replace("[PAD]","").strip()
            decoded_texts.append(decoded_text)
            labels.append(label)
        return decoded_texts, labels
