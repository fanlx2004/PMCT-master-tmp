import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from datetime import datetime
from collections import Counter, defaultdict
import copy
import sys
import argparse
from tools.PMCTNetwork_attention import PMCTNetwork_attention


class BalancedBuffer:

    
    def __init__(self, max_size: int = 20000):
        self.max_size = max_size
        self.buffer_by_label = defaultdict(list)  
        self.label_counts = Counter()  
        self.total_samples = 0
        self.samples_per_label = 0  
    
    def add_sample(self, sample: tuple, label: int):

        if self.total_samples >= self.max_size:
  
            self._remove_random_sample(preferred_label=label)
        
        self.buffer_by_label[label].append(sample)
        self.label_counts[label] += 1
        self.total_samples += 1
    
    def _remove_random_sample(self, preferred_label: Optional[int] = None):

        available_labels = [lbl for lbl, samples in self.buffer_by_label.items() if samples]
        if not available_labels:
            return

        label_to_remove = None
        if preferred_label is not None and preferred_label in self.buffer_by_label and self.buffer_by_label[preferred_label]:
            label_to_remove = preferred_label
        else:
            label_to_remove = random.choice(available_labels)

        samples_list = self.buffer_by_label.get(label_to_remove, [])
        if samples_list:
            idx = random.randrange(len(samples_list))
            del samples_list[idx]
            self.label_counts[label_to_remove] -= 1
            self.total_samples -= 1

            if not samples_list:
                try:
                    del self.buffer_by_label[label_to_remove]
                except KeyError:
                    pass
                try:
                    del self.label_counts[label_to_remove]
                except KeyError:
                    pass
    
    def get_random_sample(self) -> Optional[Tuple[tuple, int]]:

        if self.total_samples == 0:
            return None
        
        available_labels = [label for label, samples in self.buffer_by_label.items() if samples]
        if not available_labels:
            return None

        weights = [len(self.buffer_by_label[label]) for label in available_labels]
        label = random.choices(available_labels, weights=weights, k=1)[0]

        samples_for_label = self.buffer_by_label.get(label, [])
        if samples_for_label:
            idx = random.randrange(len(samples_for_label))
            sample = samples_for_label[idx]
            return sample, label

        return None
    
    def get_batch(self, batch_size: int) -> List[Tuple[tuple, int]]:

        batch = []
        for _ in range(batch_size):
            sample = self.get_random_sample()
            if sample:
                batch.append(sample)
        return batch
    
    def clear(self):
        self.buffer_by_label.clear()
        self.label_counts.clear()
        self.total_samples = 0
        self.samples_per_label = 0
    
    def update_sampling_plan(self, min_label_count: int):
        self.samples_per_label = max(1, min_label_count // 2)


class RandomFileDataset:
    def __init__(self, data_dir: str, batch_size: int = 256, failure_file: str = None):

        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob("*.json"))
        self.batch_size = batch_size
        self.failure_file = Path(failure_file) if failure_file else None
        self.failure_samples = []

        

        self.buffer = BalancedBuffer(max_size=100000)
        

        self.processed_files = set()
    
        self.failure_sample_size = 0

        self.need_full_refresh = True 

        self.batch_counter = 0
        self.batches_per_update = 20  
        self.files_per_update = 150    
        
        self.failure_samples = self._read_failure_file()
        self.failure_sample_size = len(self.failure_samples)
       
    
    def _read_file(self, file_path: Path) -> List:

        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _read_failure_file(self) -> List:

        if not self.failure_file or not self.failure_file.exists():
            return []
        
        try:
            with open(self.failure_file, 'r') as f:
                failures = json.load(f)
            return failures
        except Exception as e:
            return []
    
    def _parse_sample(self, item) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:

        if isinstance(item, dict) and 'st' in item and 'st1' in item and 'rt' in item:

            st, st1, rt = item['st'], item['st1'], item['rt']
        elif isinstance(item, list) and len(item) == 3:

            st, st1, rt = item
        else:
            return None
        
        if (isinstance(st, list) and len(st) == 33 and 
            isinstance(st1, list) and len(st1) == 33 and 
            isinstance(rt, (int, float)) and 1 <= rt <= 11):
            return np.array(st, dtype=np.float32), np.array(st1, dtype=np.float32), int(rt)
        
        return None
    
    def _sample_from_file_balanced(self, file_path: Path) -> List[Tuple[tuple, int]]:
        
        

        try:
            data = self._read_file(file_path)
        except Exception as e:
            return []
        
        # 按类别分组
        samples_by_label = defaultdict(list)
        valid_samples = []
        
        for item in data:
            sample = self._parse_sample(item)
            if sample:
                st, st1, rt = sample
                samples_by_label[rt].append((st, st1, rt))
                valid_samples.append(((st, st1, rt), rt))
        
        if not samples_by_label:
            return []
        
        min_count = min(len(samples) for samples in samples_by_label.values())
        

        self.buffer.update_sampling_plan(min_count)
        

        sampled_data = []
        for label, samples in samples_by_label.items():
            if label == 11:

                desired = max(1, int(1.5 * self.buffer.samples_per_label))
                desired = min(desired, len(samples))
                if len(samples) > desired:
                    selected = random.sample(samples, desired)
                else:
                    selected = samples
            else:

                if len(samples) > 1.5 * self.buffer.samples_per_label:
                    raint = random.randint(int(0.5 * self.buffer.samples_per_label), int(1.5 * self.buffer.samples_per_label))
                    selected = random.sample(samples, raint)
                else:
                    selected = samples 

            for sample in selected:
                sampled_data.append((sample, label))
        
        if len(self.failure_samples) >= 10 :
            failure_data = random.choices(self.failure_samples, k=10)
            for sample in failure_data:
                sample = [sample['st'], sample['st1'], sample['rt']]
                sampled_data.append((sample, sample[-1]))
        
        return sampled_data
    
    def _refresh_buffer_partial(self, num_files: int = 5):

        unprocessed_files = [f for f in self.files if f not in self.processed_files]
        
        if not unprocessed_files:
            self.processed_files.clear()
            unprocessed_files = self.files.copy()
        
        files_to_read = random.sample(unprocessed_files, min(num_files, len(unprocessed_files)))
        
        total_added = 0
        for file_path in files_to_read:

            sampled_data = self._sample_from_file_balanced(file_path)
            for sample, label in sampled_data:
                self.buffer.add_sample(sample, label)
                total_added += 1

            self.processed_files.add(file_path)

        if self.failure_samples:
            num_failure_samples = min(10, len(self.failure_samples)) 
            selected_failures = random.sample(self.failure_samples, num_failure_samples)
            
            for item in selected_failures:
                sample = self._parse_sample(item)
                if sample:
                    st, st1, rt = sample
                    self.buffer.add_sample((st, st1, rt), rt)
                    total_added += 1
        
       
        
    
    
    def _refresh_buffer_full(self):

        self.buffer.clear()
        self.processed_files.clear()
        self.need_full_refresh = False
        self._refresh_buffer_partial(num_files=20)  
    
    def get_sample(self) -> Tuple[np.ndarray, np.ndarray, int]:

        if self.need_full_refresh:
            self._refresh_buffer_full()

        result = self.buffer.get_random_sample()
        
        if result is None:
            self._refresh_buffer_partial()
            result = self.buffer.get_random_sample()
            
            if result is None:
                return np.zeros(33, dtype=np.float32), np.zeros(33, dtype=np.float32), 11
        
        (st, st1, rt), _ = result
        return st, st1, rt
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        st_batch, st1_batch, rt_batch = [], [], []
        
        for _ in range(batch_size):
            st, st1, rt = self.get_sample()
            st_batch.append(st)
            st1_batch.append(st1)
            rt_batch.append(rt)
        
        self.batch_counter += 1
        

        if self.batch_counter >= self.batches_per_update:
            self._refresh_buffer_partial(num_files=self.files_per_update)
            self.batch_counter = 0  
        
        return np.array(st_batch), np.array(st1_batch), np.array(rt_batch)
    
    def refresh_for_validation(self):

        self.need_full_refresh = True
        self._refresh_buffer_full()
        self.batch_counter = 0 



class PMCTTrainer:

    def __init__(self, train_dir: str, val_dir: str, test_dir: str, batch_size: int = 256, 
                 lr: float = 0.001, model_save_dir: str = "./saved_models", patience: int = 200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.failure_file = self.model_save_dir / "failure.json"

        if not self.failure_file.exists():
            with open(self.failure_file, 'w') as f:
                json.dump([], f, indent=4)

        else:
            try:
                with open(self.failure_file, 'r') as f:
                    failures = json.load(f)
               
            except:

                with open(self.failure_file, 'w') as f:
                    json.dump([], f, indent=4)

 
        self.train_dataset = RandomFileDataset(train_dir, batch_size=batch_size, 
                                               failure_file=str(self.failure_file))

        self.val_dataset = RandomFileDataset(val_dir, batch_size=batch_size)

        self.test_dataset = RandomFileDataset(test_dir, batch_size=batch_size)

        self.model = PMCTNetwork_attention().to(self.device)

       
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.lambda_bound = 1.0       
        self.lambda_consistency = 1.0  
        self.lambda_incentive = 0.5    
        self.delta = 1.0            
        
        self.batch_size = batch_size
        

        self.train_losses = []
        self.val_results = []
        
 
        self.best_val_misclassification = float('inf')
        self.best_val_conservativeness = float('inf')
        self.best_epoch = 0
        

        self.best_model_path = self.model_save_dir / "best_model_attention_training.pth"
        self.pac_verified = False
        

        self.patience = patience
        self.patience_counter = 0
        self.early_stop = False
        self.start_time = datetime.now()

        self.total_failures_collected = 0
        self.early_epoch_threshold = 0
        self.best_combined_loss = float('inf')
        self.current_epoch_train_loss = 0.0

    def compute_loss(self, q_st: torch.Tensor, q_st1: torch.Tensor, rt: torch.Tensor) -> Tuple[torch.Tensor, Dict]:

        bound_loss = torch.mean(torch.relu(q_st - rt) ** 2)
        consistency_loss = torch.mean(torch.relu(q_st - q_st1 - 1) ** 2)
        incentive_loss = torch.mean(torch.relu((rt - self.delta) - q_st) ** 2)

        total_loss = (
            self.lambda_bound * bound_loss +
            self.lambda_consistency * consistency_loss +
            self.lambda_incentive * incentive_loss
        )
        
        with torch.no_grad():
            conservativeness = torch.mean(torch.clamp(rt - q_st, min=0)).item()
            avg_pred = torch.mean(q_st).item()
            avg_truth = torch.mean(rt).item()
            

            msc_pred = self.model._get_msc_pred(q_st)
            misclassified = torch.sum(msc_pred > rt.long()).item() / rt.size(0)
        
        loss_dict = {
            'total': total_loss.item(),
            'bound': bound_loss.item(),
            'consistency': consistency_loss.item(),
            'incentive': incentive_loss.item(),
            'conservativeness': conservativeness,
            'avg_pred': avg_pred,
            'avg_truth': avg_truth,
            'misclassified': misclassified
        }
        
        return total_loss, loss_dict
    
    def train_step(self) -> float:

        self.model.train()
        
        st, st1, rt = self.train_dataset.get_batch(self.batch_size)
        
        st_tensor = torch.FloatTensor(st).to(self.device)
        st1_tensor = torch.FloatTensor(st1).to(self.device)
        rt_tensor = torch.FloatTensor(rt).to(self.device)

        q_st, _ = self.model(st_tensor)
        q_st1, _ = self.model(st1_tensor)
        
        q_st1 = q_st1.detach()
        mask = (st1_tensor == 0.0).all(dim=1)
        q_st1[mask] = 0.0
        
        target = torch.clamp(torch.minimum(rt_tensor, q_st1+1), min=1, max=11)
        error = q_st - target
        
        loss = torch.mean(error ** 2)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, dataset: RandomFileDataset, num_batches: int = 50) -> Tuple[Dict, Dict]:

        self.model.eval()
        dataset.refresh_for_validation()
        
        total_samples = 0
        total_correct = 0
        total_misclassified = 0
        total_pred_sum = 0
        total_truth_sum = 0
        loss_summary = {
            'total': 0, 'bound': 0, 'consistency': 0, 'incentive': 0
        }
        
        with torch.no_grad():
            for _ in range(num_batches):
                st, st1, rt = dataset.get_batch(self.batch_size)
                st_tensor = torch.FloatTensor(st).to(self.device)
                st1_tensor = torch.FloatTensor(st1).to(self.device)
                rt_tensor = torch.FloatTensor(rt).to(self.device)
                

                q_st, msc_pred = self.model(st_tensor)
                q_st1, _ = self.model(st1_tensor)
                q_st1 = q_st1.detach()
                mask = (st1_tensor == 0.0).all(dim=1)
                q_st1[mask] = 0.0
                
                target = torch.clamp(torch.minimum(rt_tensor, q_st1+1), min=1, max=11)
                error = q_st - target

                
                loss = torch.mean(error ** 2)
                
                loss_summary['total'] += loss * st.shape[0]

                total_samples += st.shape[0]
                
                total_correct += (msc_pred == rt_tensor.long()).sum().item()
                
                total_misclassified += (msc_pred > rt_tensor.long()).sum().item()
                
                total_conservativeness = torch.relu(rt_tensor.long() - msc_pred).sum().item()
                
                total_pred_sum += torch.sum(q_st).item()
                total_truth_sum += torch.sum(rt_tensor).item()
        
        for key in loss_summary:
            loss_summary[key] /= total_samples
        
        metrics = {
            'accuracy': total_correct / total_samples,
            'misclassification_rate': total_misclassified / total_samples,
            'conservativeness' : total_conservativeness / total_samples,
            'avg_prediction': total_pred_sum / total_samples,
            'avg_truth': total_truth_sum / total_samples
        }
        
        return loss_summary, metrics
    
    def save_best_model(self, metrics: Dict, val_loss: float, epoch: int):
        misclassification_rate = metrics['misclassification_rate']
        conservativeness = metrics['conservativeness']
        train_loss = self.current_epoch_train_loss
        combined_loss = (train_loss + val_loss) / 2

        should_save = False
        
        if epoch <= self.early_epoch_threshold:

            if combined_loss < self.best_combined_loss:
                should_save = True
                improvement = self.best_combined_loss - combined_loss

        else:

            if self.best_val_misclassification == float('inf'):
                should_save = True
                

            elif misclassification_rate < self.best_val_misclassification:
                should_save = True
                improvement = self.best_val_misclassification - misclassification_rate
                

            elif (abs(misclassification_rate - self.best_val_misclassification) < 0.000001 and 
                  conservativeness < self.best_val_conservativeness):
                should_save = True
                improvement = self.best_val_conservativeness - conservativeness
               
        
        if should_save:
            if epoch <= self.early_epoch_threshold:
                old_combined_loss = self.best_combined_loss
                self.best_combined_loss = combined_loss
                self.best_epoch = epoch
                

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'combined_loss': combined_loss,
                    'early_save': True  
                }, self.best_model_path)
                
            else:
                old_misclass = self.best_val_misclassification
                old_conserve = self.best_val_conservativeness
                old_epoch = self.best_epoch
                
                self.best_val_misclassification = misclassification_rate
                self.best_val_conservativeness = conservativeness
                self.best_epoch = epoch
  
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'metrics': metrics,
                    'best_val_misclassification': self.best_val_misclassification,
                    'best_val_conservativeness': self.best_val_conservativeness,
                    'early_save': False 
                }, self.best_model_path)
                
                print(f"✓ save best model to {self.best_model_path}")
                print(f"  misclass={misclassification_rate:.4f} (old: {old_misclass:.4f}), "
                      f"conservativeness={conservativeness:.2f} (old: {old_conserve:.2f})")
                print(f" from epoch: {epoch} (old: {old_epoch})")
            
            self.patience_counter = 0 
        else:
            self.patience_counter += 1
            if epoch <= self.early_epoch_threshold:
                print(f"✗ average loss = {combined_loss:.4f} (best={self.best_combined_loss:.4f})")
                print(f"  train loss: {train_loss:.4f}, test loss: {val_loss:.4f}")
            else:
                print(f"✗  misclass={misclassification_rate:.4f} (best={self.best_val_misclassification:.4f}), "
                      f"conservativeness = {conservativeness:.2f} (best={self.best_val_conservativeness:.2f})")
            print(f"  early stop count: {self.patience_counter}/{self.patience}")
    
    def load_best_model(self) -> bool:
        if not self.best_model_path.exists():
            return False
        
        try:

            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            

            if 'best_val_misclassification' in checkpoint:
                self.best_val_misclassification = checkpoint['best_val_misclassification']
                self.best_val_conservativeness = checkpoint['best_val_conservativeness']
                self.best_epoch = checkpoint.get('epoch', 0)
            
            print(f"✓ load best model: {self.best_model_path}")
            print(f" misclass={self.best_val_misclassification:.4f}, "
                  f"conservativeness={self.best_val_conservativeness:.2f}")
            
            return True
            
        except Exception as e:
            return False
    
    def pac_verification(self, epsilon: float = 0.05, beta: float = 0.001) -> bool:

        self.model.eval()
        self.test_dataset.refresh_for_validation()

        N_i = math.ceil(math.log(beta) / math.log(1 - epsilon))
        print(f"\nstart PAC verification")
        print(f"epsilon={epsilon}, beta={beta}")
        print(f"each MSC category needs {N_i} success samples")
        

        success_counts = [0] * 12  
        total_counts = [0] * 12    

        sampled_msc_classes = set()

        verified_msc_classes = set()
        
        total_attempts = 0
        total_success = 0
        total_failure = 0
        total_skipped = 0 
        

        new_failure_samples = []
        
        

        while True:
            total_attempts += 1
            if total_attempts % 30 == 0:
                self.test_dataset.refresh_for_validation()
                

            st, st1, rt = self.test_dataset.get_sample()

            st_tensor = torch.FloatTensor(st).unsqueeze(0).to(self.device)
            st1_tensor = torch.FloatTensor(st1).unsqueeze(0).to(self.device)
            rt_tensor = torch.FloatTensor([rt]).to(self.device)
            rt_value = rt
            
            with torch.no_grad():
                q_st, msc_pred = self.model(st_tensor)
                q_st1, _ = self.model(st1_tensor)

                predicted_msc = msc_pred.item()

            msc_class = predicted_msc
            

            if msc_class in verified_msc_classes:
                total_skipped += 1
                continue
            
            total_counts[msc_class] += 1

            if msc_class not in sampled_msc_classes:
                sampled_msc_classes.add(msc_class)
                
            verification_result = rt_value >= predicted_msc
            
            if total_attempts <= 100 or total_attempts % 50 == 0:
                if verification_result:

                    success_counts[msc_class] += 1
                    total_success += 1
                    status = "✓"

                    if success_counts[msc_class] >= N_i and msc_class not in verified_msc_classes:
                        verified_msc_classes.add(msc_class)
                        print(f"🎯 MSC{msc_class} pass verification! ({success_counts[msc_class]}/{N_i} success)")
                        print(f"  pass verification: {sorted(verified_msc_classes)}")
                else:

                    total_failure += 1
                    status = "✗"
 
                    failure_sample = {
                        'st': st.tolist(),
                        'st1': st1.tolist(),
                        'rt': int(rt_value),
                        'predicted_msc': int(predicted_msc),
                        'attempt': total_attempts,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    new_failure_samples.append(failure_sample)
                    
                    try:
                        existing_failures = []
                        if self.failure_file.exists():
                            with open(self.failure_file, 'r') as f:
                                existing_failures = json.load(f)
                        
                        new_failure_for_training = {
                            'st': st.tolist(),
                            'st1': st1.tolist(),
                            'rt': int(rt_value)
                        }
                        
                        if new_failure_for_training not in existing_failures:
                            existing_failures.append(new_failure_for_training)
                            self.total_failures_collected += 1
                            with open(self.failure_file, 'w') as f:
                                json.dump(existing_failures, f, indent=4)
                            
                    except Exception as e:
                        pass

            else:

                if verification_result:
                    success_counts[msc_class] += 1
                    total_success += 1
 
                    if success_counts[msc_class] >= N_i and msc_class not in verified_msc_classes:
                        verified_msc_classes.add(msc_class)
                else:
                    total_failure += 1

                    failure_sample = {
                        'st': st.tolist(),
                        'st1': st1.tolist(),
                        'rt': int(rt_value),
                        'predicted_msc': int(predicted_msc)
                    }
                    new_failure_samples.append(failure_sample)
                    
                    try:
                        existing_failures = []
                        if self.failure_file.exists():
                            with open(self.failure_file, 'r') as f:
                                existing_failures = json.load(f)
                        
                        new_failure_for_training = {
                            'st': st.tolist(),
                            'st1': st1.tolist(),
                            'rt': int(rt_value)
                        }
                        
                        if new_failure_for_training not in existing_failures:
                            existing_failures.append(new_failure_for_training)
                            self.total_failures_collected += 1
                            with open(self.failure_file, 'w') as f:
                                json.dump(existing_failures, f, indent=4)
                    except:
                        pass
            
            if not verification_result:

                print("\n" + "=" * 80)
                print(f"✗ fail !")
                print(f"  MSC={predicted_msc}, rt={rt_value}")
                print("=" * 80)
                return False
            
            if sampled_msc_classes.issubset(verified_msc_classes) and len(sampled_msc_classes) > 0:
                print("\n" + "=" * 80)
                print(f"✓ PAC success !")
                
                

                all_predictions = sum(total_counts[1:])
                for msc in range(1, 12):
                    count = total_counts[msc]
                    if all_predictions > 0:
                        percentage = count / all_predictions * 100
                        if msc in verified_msc_classes:
                            marker = "✓✓"
                        elif msc in sampled_msc_classes:
                            marker = "✓ "
                        else:
                            marker = "  "
                        print(f"  {marker} MSC={msc:2d}: {count:4d}次 ({percentage:6.2f}%)")

                print("=" * 80)
                return True

            if (total_attempts - total_skipped) % 200 == 0:
                remaining = []
                completed = []
                for msc in sorted(sampled_msc_classes):
                    if msc in verified_msc_classes:
                        completed.append(f"MSC{msc}")
                    elif success_counts[msc] < N_i:
                        remaining.append(f"MSC{msc}:{success_counts[msc]}/{N_i}")
                    else:
                        remaining.append(f"MSC{msc}:{success_counts[msc]}/{N_i}(waiting)")
                

    
    def train(self, epochs: int, steps_per_epoch: int = 1000, 
        validation_interval: int = 100, pac_verification_interval: int = 10):

        try:
            with open(self.failure_file, 'r') as f:
                failures = json.load(f)

            self.total_failures_collected = len(failures)
        except:
            pass
        
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            if self.pac_verified:
                print(f"\n pass pac, stop.")
                break
            
            for step in range(steps_per_epoch):
                loss = self.train_step()
                epoch_loss += loss

                if (step + 1) % validation_interval == 0:
                    _, val_metrics = self.validate(self.val_dataset, num_batches=10)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    print(f"Epoch {epoch+1:3d}/{epochs}, Step {step+1:4d}/{steps_per_epoch}: "
                        f"Loss: {epoch_loss/(step+1):.4f}, "
                        f"Misclass: {val_metrics['misclassification_rate']:.4f}, "
                        f"Conserve: {val_metrics['conservativeness']:.2f}, "
                        f"Pred/Truth: {val_metrics['avg_prediction']:.1f}/{val_metrics['avg_truth']:.1f}, "
                        f"LR: {current_lr:.6f}, "
                        f"Failures: {self.total_failures_collected}")

            avg_loss = epoch_loss / steps_per_epoch
            self.current_epoch_train_loss = avg_loss 

            self.train_losses.append({
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
            })
            
            loss_summary, val_metrics = self.validate(self.val_dataset, num_batches=1000)
            self.val_results.append({
                'epoch': epoch + 1,
                **val_metrics
            })
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1:3d}/{epochs} finish: ")
            print(f"  train loss: {avg_loss:.4f}")
            print(f"  test loss: {loss_summary['total']:.4f}")
            print(f"  Misclass: {val_metrics['misclassification_rate']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"Conserve: {val_metrics['conservativeness']:.2f}, "
                f"LR: {current_lr:.6f}")
            print("-" * 80)
            

            self.save_best_model(val_metrics, loss_summary['total'], epoch+1)
            self.scheduler.step()
            if self.patience_counter >= self.patience:

                
                if self.load_best_model():
                    is_verified = self.pac_verification()
                    if is_verified:
                        print("✓ success")
                        self.pac_verified = True
                        break
                    else:
                        print("✗ stop training")
                        break
                else:
                    break
            
            if (epoch + 1) % pac_verification_interval == 0:
                is_verified = self.pac_verification()
                if is_verified:
                    self.pac_verified = True
                    break
                else:
                    print("✗ continue training")
            
        end_time = datetime.now()
        training_duration = end_time - self.start_time
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select mode: road or cross")
    parser.add_argument("--mode", choices=["road", "cross"], default="road", help="Operating mode")
    args = parser.parse_args()
    mode = args.mode()

    base = 'data/terasim_data'
    if mode == "road":
        train_dir = f"{base}/train_road"
        val_dir = f"{base}/val_road"
        test_dir = f"{base}/test_road"
        model_save_dir = "model_params/saved_pmct_models_onlyroad"
    else:
        train_dir = f"{base}/train_cross"
        val_dir = f"{base}/val_cross"
        test_dir = f"{base}/test_cross"
        model_save_dir = "model_params/saved_pmct_models_cross"

    
    
    batch_size = 256
    learning_rate = 0.0001
    epochs = 2000
    steps_per_epoch = 500
    patience = 150  
    
    trainer = PMCTTrainer(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        lr=learning_rate,
        model_save_dir=model_save_dir,
        patience=patience
    )
    

    
    if trainer.load_best_model():

        
        _, metrics = trainer.validate(trainer.val_dataset, num_batches=1000)
        
        print(f"\n existing model:")
        print(f"  Misclassification: {metrics['misclassification_rate']:.6f}")
        print(f"  Conservativeess: {metrics['conservativeness']:.6f}")
        print(f"  Acc: {metrics['accuracy']:.6f}")
        print(f"  Avg prediction: {metrics['avg_prediction']:.4f}")
        print(f"  Avg truth: {metrics['avg_truth']:.4f}")
        
        is_verified = trainer.pac_verification()
        
        if is_verified:
            print("PAC passed!")
            sys.exit()
        else:
           print("PAC didn't pass!") 
        
        trainer.best_val_misclassification = metrics['misclassification_rate']
        trainer.best_val_conservativeness = metrics['conservativeness']

    trainer.train(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_interval=5,  
        pac_verification_interval=1  
    )