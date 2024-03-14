import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing. 
# The flag decides if the loss will be calculted over all 
# or just the predicted values.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, sequence_length, d_model]
        pe = self.pe[:, :x.size(1)]  # 获取与输入相同序列长度的位置编码
        return x + pe

       

class TransAm(nn.Module):
    def __init__(self,feature_size=4,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=2,dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def returnnum(normalized_data):
    def inverse_normalize(data, min_value, max_value):
        return (data + 1) * (max_value - min_value) / 2 + min_value
    series = pd.read_csv(data_path, header=0, index_col=0).squeeze('columns').to_numpy()
    series = series[:,-1]
    min_val = min(series)
    max_val = max(series)
    original_data = inverse_normalize(normalized_data, min_val, max_val)
    return original_data
# 假设 input_data 是一个 Numpy 数组
def create_inout_sequences(input_data, tw):
    L, M = input_data.shape
    zeros_array = np.zeros((output_window, M))
    
    # 预先分配内存（这里需要根据实际情况调整维度和大小）
    inout_seq = np.zeros((L-tw, 2, tw, M))  # 示例维度，实际可能不同
    for i in range(L-tw):
        train_seq = np.concatenate((input_data[i:i+tw][:-output_window], zeros_array))
        train_label = input_data[i:i+tw]  # 或其他逻辑
        inout_seq[i] = np.array([train_seq, train_label])  # 直接在预分配的数组中填充数据
    return torch.FloatTensor(inout_seq)


def get_data(data_path):
    # time        = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
    import pandas as pd
    series = pd.read_csv(data_path, header=0, index_col=0).squeeze('columns')
    series = series.to_numpy()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(series)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    '''
    数据集划分todo:vail
    '''
    sampels = 640000
    train_data = amplitude[:sampels,:]
    test_data = amplitude[sampels:,:]
    # sampels = 640000
    # train_data = amplitude[:sampels]
    # test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?
    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack

    return train_sequence.to(device),test_data.to(device)

def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len] 
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    # print(input.shape)
    # print(target.shape)  
    input = torch.squeeze(input, 2)
    target = torch.squeeze(target, 2)
    # print(input.shape)
    # print(target.shape)  
    return input, target


def train(train_data):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)  
        output = torch.squeeze(output, 2)      
        targets = targets[:, :, -1]
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            # look like the model returns static values for the output window
            output = eval_model(data)  
            output = torch.squeeze(output, 2)      
            target = target[:, :, -1]  
            if calculate_loss_over_all_values:                                
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy()
    len(test_result)

    pyplot.plot(test_result,color="red")
    pyplot.plot(truth[:500],color="blue")
    pyplot.plot(test_result-truth,color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    pyplot.close()
    
    return total_loss / i


def predict_future(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    _ , data = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps,1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0     
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)
    

    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png'%steps)
    pyplot.close()
        
# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
# auch zu denen der predict_future
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            output = torch.squeeze(output, 2)
            targets = targets[:, :, -1]
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)


def trainning_main():
    data_path = 'final_data-CH41.csv'
    train_data, val_data = get_data(data_path)

    model = TransAm().to(device)
    criterion = nn.MSELoss()
    lr = 0.005 
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

    best_val_loss = float("inf")
    epochs = 100 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        
        
        if(epoch % 100 == 0):
            val_loss = plot_and_loss(model, val_data,epoch)
            # predict_future(model, val_data,200)
        else:
            val_loss = evaluate(model, val_data)
            
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        # 训练模型...
        optimizer.step()
        scheduler.step()

    print('training is over.')
    print('-' * 89)

    filename = 'best_model' + str(batch_size) + '.pth'
    torch.save(model.state_dict(), filename)
    print(f"模型已保存为：{filename}")

# trainning_main
calculate_loss_over_all_values = False
input_window = 180
output_window = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64 # batch size
data_path = 'final_data-CH41.csv'
train_data, val_data = get_data(data_path)
## tranning
## prediction
model = TransAm().to(device)
model.load_state_dict(torch.load('best_model512.pth'))
model.eval()
data, target = get_batch(val_data, 150000 ,1)

output = model(data) 
target = target[:, :, -1]
output = torch.squeeze(output, 2)
output = torch.squeeze(output, 1)
target = torch.squeeze(target, 1)
output = output[-60:].detach().cpu().numpy()
target = target[-60:].detach().cpu().numpy()
output = returnnum(output)
target = returnnum(target)
# 创建一个新的图像
plt.figure()
# 绘制data曲线，标记为Data
plt.plot(output, label='prediction')
# 绘制target曲线，标记为Target
plt.plot(target, label='true')
# 添加图例
plt.legend()

# 添加标题
plt.title('prediction')

# 添加X轴和Y轴标签
plt.xlabel('5s/time')
plt.ylabel('Temperature')

# 显示图像
plt.show()
