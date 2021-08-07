import torch
from torch import nn
from transformers import AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix


class BertClassifier(nn.Module):
    def __init__(self, bert_name, num_label=5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_label)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        features = bert_outputs.last_hidden_state[:, 0]
        logits = self.linear(features)
        preds = self.softmax(logits)
        return preds


def train_loop(train_loader, model, loss_fn, optim,
    val_loader=None, log_interval=5, epoch_id=None):
    """One train loop for one epoch.

    Args:
        train_loader: Pytorch dataloader for training.
        model: A BertForSequentialClassification model.
        loss_fn: Loss function.
        optim: Optimizer.
        val_loader: Pytorch dataloader for validation.
        log_interval: Integer. Zero for no logging.
        epoch_id: Optional integer shown in log string.
    Returns:
        A tuple of (train acc, train loss, val acc, val loss),
        where val_acc, val_ls are optional if val_loader is given.
    """

    step_loss = 0
    tot_train_ls, tot_train_acc = 0, 0

    # Training
    model.train()
    for batch_id, batch in enumerate(train_loader):
        optim.zero_grad()
        # Onto device (GPU).
        device = next(model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # Forward.
        preds = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(preds, labels)
        # BP and update.
        loss.backward()
        optim.step()

        # Log the batch.
        tot_train_ls += loss.item()
        tot_train_acc += accuracy_score(
            labels.cpu().flatten(), preds.argmax(axis=1).cpu().flatten())
        step_loss += loss.item()
        if log_interval and (batch_id+1) % log_interval == 0:
            s_epoch = f'Epoch {epoch_id:3} ' if epoch_id is not None else ''
            print('[{}Batch {:>3}/{:3}] train_loss={:.4f}'.format(
                s_epoch, batch_id+1, len(train_loader), 
                step_loss / log_interval
                ))   
            step_loss = 0

    # Log the training epoch.
    avg_train_ls = tot_train_ls / len(train_loader)
    avg_train_acc= tot_train_acc / len(train_loader)

    # Validation
    if val_loader:
        val_acc, val_ls, _ = test_loop(val_loader, model, loss_fn, log_interval=0)
        return avg_train_acc, avg_train_ls, val_acc, val_ls
    else:
        return avg_train_acc, avg_train_ls
    


def test_loop(test_loader, model, loss_fn, log_interval=1):
    """Test loop for validation and evalutaion.

    Args:
        test_loader: Pytorch dataloader.
        model: A BertForSequentialClassification model.
        log_interval: Integer. Zero for no logging on terminal.
    Returns:
        Tuple of (accuracy, loss, confusion matrix).
    """

    tot_test_ls, tot_test_acc = 0, 0
    all_preds, all_labels = [], []

    # Evaluation.
    model.eval()
    for batch_id, batch in enumerate(test_loader):
        with torch.no_grad():
            # Inference.
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            preds = model(input_ids, attention_mask=attention_mask)
            # Save the batch result.
            all_preds.append(preds.argmax(axis=1))
            all_labels.append(labels)
            # Compute metrics of batch.
            batch_acc = accuracy_score(
                labels.cpu().flatten(), preds.argmax(dim=1).cpu().flatten())
            batch_ls = loss_fn(preds, labels).item()
            tot_test_acc += batch_acc
            tot_test_ls += batch_ls
            # Log on terminal.
            if log_interval and (batch_id+1) % log_interval == 0:
                print('[Batch {:>3}/{:3}] batch_acc={:.4f} batch_ls={:.6f}'.format(
                    batch_id+1, len(test_loader), batch_acc, batch_ls)) 

    test_acc = tot_test_acc / len(test_loader)
    test_ls = tot_test_ls / len(test_loader)
    all_preds = torch.hstack(all_preds).cpu()
    all_labels = torch.hstack(all_labels).cpu()
    conf_mat = confusion_matrix(all_labels.cpu(), all_preds.cpu())
    
    return test_acc, test_ls, conf_mat