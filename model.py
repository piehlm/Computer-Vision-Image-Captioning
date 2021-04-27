import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
        
    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
       
        #lstm_out, self.hidden = self.lstm(embeddings, self.hidden)  
        #lstm_out, self.hidden = self.lstm(embeddings.view(len(embeddings), 1, -1), self.hidden) 
        lstm_out, self.hidden = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            outputs,states = self.lstm(inputs,states)
            outputs = self.linear(outputs.squeeze(1))
            tar_ind = outputs.max(1)[1]
            if tar_ind.item() == 1:
                break
            res.append(tar_ind.item())
            inputs = self.embed(tar_ind).unsqueeze(1)
        return res
    
