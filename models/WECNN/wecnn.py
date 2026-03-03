import torch
import torch.nn as nn
import torch.nn.functional as F

class WECNN(nn.Module):
  
  def __init__(self, vocab_size, embedding_dim=300, filters=[2,3,4,6,8], num_filters=200):
    super(WECNN, self).__init__()
    
    # Embedding Layer Question Pair & Unique Words
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # Keyword Embdding and Freezing Weights - Section 3.5
    self.keyword_embedding = nn.Embedding(vocab_size, embedding_dim)
    self.keyword_embedding.weight.requires_grad = False
    
    self.filters = filters
    self.num_filters = num_filters 
    
    self.convs = nn.ModuleList([
      nn.Conv1d(in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=h)
      for h in filters
    ])
    
    # Initialization of conv layers with Xavier Uniform
    for conv in self.convs:
      nn.init.xavier_uniform_(conv.weight)
      
    # lambda
    self.lam = nn.Parameter(torch.ones(1))
    
    # Dropout=0.1 as paper specs  
    self.dropout = nn.Dropout(0.1)
  
    self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.7)
    
    # Concatenation layer with n_filters * 3 (cos(c_min), cos(c_max), cos(c_max_i))
    # Output is 2 since they are similar or not
    self.fc = nn.Linear(len(self.filters)*3, 2)
    
  def get_features(self, x, isKeyword=False):
    emb_layer = self.keyword_embedding if isKeyword else self.embedding
    x = self.bn(emb_layer(x).transpose(1, 2))
    
    all_pools = []
    for conv in self.convs:
      c = self.dropout(F.relu(conv(x)))
      
      c_max = torch.max(c, dim=2)[0]
      c_min = torch.min(c, dim=2)[0]
      
      chunks = torch.chunk(c, 4, dim=1)
      c_max_i = torch.stack([torch.max(chunk.view(chunk.size(0), -1), dim=1)[0] 
                             for chunk in chunks])
      all_pools.append(c_max, c_min, c_max_i)
    return all_pools
  
  def cosine_sim(self, f1, f2):
    dot = torch.sum(f1 * f2, dim=1)
    norm = torch.norm(f1, 2, 1) * torch.norm(f2, 2, 1)
    epsilon = 1e-8
    # Not compute division by zero
    return (dot / (norm + epsilon))  + self.lam
  
  def forward_propagation(self, p, q, p_u, q_u, p_k, q_k):
    
    # Get all values for each embeddings  
    feat_p = self.get_features(p)
    feat_p_u = self.get_features(p_u)
    feat_p_k = self.get_features(p_k, True)
    
    feat_q = self.get_features(q)
    feat_q_u = self.get_features(q_u)
    feat_q_k = self.get_features(q_k, True)
    
    combined_scores = []
    for i in range(len(self.filters)):
      for p_feat, q_feat in [(feat_p, feat_q), (feat_p_u, feat_q_u), (feat_p_k, feat_q_k)]:
        combined_scores.append(self.cosine_sim(p_feat[i][0], q_feat[i][0]))
        combined_scores.append(self.cosine_sim(p_feat[i][1], q_feat[i][1]))
        combined_scores.append(self.cosine_sim(p_feat[i][2], q_feat[i][2]))

    x = torch.stack(combined_scores, dim=1)
    
    output = self.fc(x)
    
    return F.softmax(output, dim=1)
  
  def train(self):
    pass
  
  def test(self):
    pass