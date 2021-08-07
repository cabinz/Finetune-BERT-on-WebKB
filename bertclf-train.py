import torch
import numpy as np
from utils.dataloader import load_data, statistics
from utils.dataloader import WebKBDataset
from utils.dataloader import id2label
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, dataloader
from bertclf import BertClassifier
from bertclf import train_loop, test_loop
from utils.vis import draw_hist_loss, draw_hist_acc, draw_confusion_matrix
import argparse
import os, time, sys

# Parse hyperparameters.
parser = argparse.ArgumentParser()
parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
parser.add_argument('--uni_lt', type=str, nargs='+')
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--batch_siz', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--split_id', type=int)
args = parser.parse_args()
# Set random seed and device.
RANDOM_SEED = args.random_seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED) # Sklearn uses numpy's random seed.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Use device:', device)


# Load data.
cat_lt = ['student', 'faculty', 'project', 'course']
label_lt = list(id2label.values())[:4]
print('uni_lt: ', args.uni_lt)
texts, labels = load_data(
    './dataset.tsv', uni_lt=args.uni_lt, cat_lt=cat_lt)
# Split Train-val-test set.
train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
try:
    train_texts, valtest_texts, train_labels, valtest_labels = train_test_split(
        texts, labels, test_size=val_ratio+test_ratio, stratify=labels)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        valtest_texts, valtest_labels, test_size=test_ratio/(test_ratio+val_ratio),
        stratify=valtest_labels)
except:
    train_texts, valtest_texts, train_labels, valtest_labels = train_test_split(
        texts, labels, test_size=val_ratio+test_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        valtest_texts, valtest_labels, test_size=test_ratio/(test_ratio+val_ratio))
# Print statistics of the splitting.
s_stats = statistics(train_labels, val_labels, test_labels)
print(s_stats)


# Tokenization
tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")
test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")


# Create pytorch dataset.
train_dataset = WebKBDataset(train_encodings, train_labels)
val_dataset = WebKBDataset(val_encodings, val_labels)
test_dataset = WebKBDataset(test_encodings, test_labels)


# Deploy model.
model = BertClassifier(args.bert_name)
model.to(device)
# Initialize the loss functon and the optimizer.
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=args.lr)


# Training.
hist_train_ls, hist_val_ls = [],[]
hist_train_acc, hist_val_acc = [], []
mx_train_acc, mx_val_acc = 0, 0
train_loader = DataLoader(train_dataset, batch_size=args.batch_siz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_siz, shuffle=True)
log_interval = len(train_loader) // 10
# Train loops.
for epoch_id in range(args.num_epoch):
    train_acc, train_ls, val_acc, val_ls = train_loop(
        train_loader, model, loss_fn, optim, 
        val_loader=val_loader, epoch_id=epoch_id,
        log_interval=log_interval
    )
    print(f'Epoch {epoch_id} finished with '
        f'train_acc={train_acc:4f}, val_acc={val_acc:4f}, '
        f'train_ls={train_ls:6f}, val_ls={val_ls:6f}')
    hist_train_acc.append(train_acc)
    hist_val_acc.append(val_acc)
    hist_train_ls.append(train_ls)
    hist_val_ls.append(val_ls)
    mx_train_acc = train_acc if train_acc > mx_train_acc else mx_train_acc
    mx_val_acc = val_acc if val_acc > mx_val_acc else mx_val_acc


# Evaluation.
test_loader = DataLoader(test_dataset, batch_size=args.batch_siz, shuffle=True)
test_acc, test_ls, test_conf_mat = test_loop(test_loader, model, loss_fn)


# Create a folder for saving.
bert_ver = 'base' if args.bert_name == 'bert-base-uncased' else 'tiny'
save_dir = './bertclf-{}-save/epoch{}/'.format(
    bert_ver, args.num_epoch
)
fig_dir = save_dir + 'fig/' 
model_dir = save_dir + 'model/'
os.mkdir(save_dir)
os.mkdir(fig_dir)
os.mkdir(model_dir)
# Save the command line and dataset splitting statistical info
with open(save_dir + 'info.txt', 'w+') as file:
    file.write('Hyperparameter settings:\n')
    file.write(str(sys.argv[1:]))
    file.write('\n\nSplitting statistical info:\n')
    file.write(s_stats)
    file.write('\n\nEnd with\n')
    file.write(f'mx_train_acc={mx_train_acc:6f}, '
        f'mx_val_acc={mx_val_acc:6f}, test_acc={test_acc:6f}')
# Save the loss and acc history.
with open(save_dir + 'hist_train_ls.txt', 'w+') as file:
    file.write(str(hist_train_ls))
with open(save_dir + 'hist_val_ls.txt', 'w+') as file:
    file.write(str(hist_val_ls))
with open(save_dir + 'hist_train_acc.txt', 'w+') as file:
    file.write(str(hist_train_acc))
with open(save_dir + 'hist_val_acc.txt', 'w+') as file:
    file.write(str(hist_val_acc))
# Visualization
draw_hist_loss(hist_train_ls, hist_val_ls, 
    save_path=fig_dir + 'train_loss.jpg', linetype='-o')
draw_hist_acc(hist_train_acc, hist_val_acc, 
    save_path=fig_dir + 'train_acc.jpg', linetype='-o')
draw_confusion_matrix(test_conf_mat, label_lt,
    save_path=fig_dir + 'conf_mat.jpg')
# Save the model and hyperparamer setup.
torch.save(model, model_dir + 'model.pt')
tokenizer.save_pretrained(model_dir)
del model
