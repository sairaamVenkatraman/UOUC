# feature extaction from pretrained model: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
#code is modified from https://github.com/Shivanshu-Gupta/Visual-Question-Answering and https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEmbedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImageEmbedding, self).__init__()

        self.fflayer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU())

    def forward(self, image):
        image_embedding = self.fflayer(image)
        return image_embedding

class MutanFusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hv.append(nn.Sequential(do, lin, nn.ReLU()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
         
        self.question_embed = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU()) 
      
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.ReLU()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, img_emb, questions_conved, questions_combined):
        # Pdb().set_trace()
        ques_emb = self.question_embed(questions_conved + questions_combined)
        #ques_emb = ques_emb.mean(dim=1) 
        batch_size = img_emb.size()[0]
        img_emb = img_emb.unsqueeze(1)
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, -1, self.out_dim)
        x_mm = F.relu(x_mm)
        return x_mm

class QuestionEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, max_length = 100):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.scale = torch.sqrt(torch.FloatTensor([0.3])).cuda()
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        #create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        
        #pos = [0, 1, 2, 3, ..., src len - 1]
        
        #pos = [batch size, src len]
        
        #embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = pos_embedded = [batch size, src len, emb dim]
        
        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        #embedded = [batch size, src len, emb dim]
        
        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        #conv_input = [batch size, src len, hid dim]
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, src len]
        
        #begin convolutional blocks...
        
        for i, conv in enumerate(self.convs):
        
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            #conved = [batch size, 2 * hid dim, src len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]
            
            #apply residual connection
            conved = (conved + conv_input) * self.scale

            #conved = [batch size, hid dim, src len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
        
        #...end convolutional blocks
        
        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        #conved = [batch size, src len, emb dim]
        
        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        
        #combined = [batch size, src len, emb dim]
        
        return conved, combined

class AnswerEmbedding(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, question_max_length = 100, max_length = 100):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        self.scale = torch.sqrt(torch.FloatTensor([0.3])).cuda()
        
        self.tok_embedding = nn.Linear(max_length, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.target_encoder = nn.Linear(question_max_length, max_length)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size,
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
      
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        
        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        #permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        
        #conved_emb = [batch size, trg len, emb dim]
        
        combined = (conved_emb + embedded) * self.scale
        
        #combined = [batch size, trg len, emb dim]
                
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        
        #energy = [batch size, trg len, src len]
        
        attention = F.softmax(energy, dim=2)
        
        #attention = [batch size, trg len, src len]
            
        attended_encoding = torch.matmul(attention, encoder_combined)
        
        #attended_encoding = [batch size, trg len, emd dim]
        
        #convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        
        #attended_encoding = [batch size, trg len, hid dim]
        
        #apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        
        #attended_combined = [batch size, hid dim, trg len]
        
        return attention, attended_combined
        
    def forward(self, encoder_conved, encoder_combined):
        
        hidden_target = self.target_encoder(encoder_conved.permute(0, 2, 1).contiguous()).mean(dim=1) 
        #hidden_target = [batch_size, trg_len]  
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
                
        batch_size = hidden_target.shape[0]
        trg_len = hidden_target.shape[1]
            
        #create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        
        #pos = [batch size, trg len]
        

        #embed tokens and positions
        tok_embedded = self.tok_embedding(hidden_target).unsqueeze(1)
        pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = [batch size, trg len, emb dim]
        #pos_embedded = [batch size, trg len, emb dim]
        
        #combine embeddings by elementwise summing        
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        #embedded = [batch size, trg len, emb dim]
        
        #pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        
        #conv_input = [batch size, trg len, hid dim]
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, trg len]
        
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
        
            #apply dropout
            conv_input = self.dropout(conv_input)
        
            #need to pad so decoder can't "cheat"
                    
            #pass through convolutional layer
            conved = conv(conv_input)

            #conved = [batch size, 2 * hid dim, trg len]
            
            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, trg len]
            
            #calculate attention
            attention, conved = self.calculate_attention(embedded, 
                                                         conved, 
                                                         encoder_conved, 
                                                         encoder_combined)
            
            #attention = [batch size, trg len, src len]
            
            #apply residual connection
            conved = (conved + conv_input) * self.scale
            
            #conved = [batch size, hid dim, trg len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))
         
        #conved = [batch size, trg len, emb dim]
            
        output = self.fc_out(self.dropout(conved))
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

class VQAModel(nn.Module):
    def __init__(self, resnet_feature_size=196, size=200, question_max_length=21, question_vocab_size=473, hidden_dim=200, n_layers=5, kernel_size=3, dropout=0.25, mutan_size=200, answer_vocab_size=410, answer_max_length=35):
        super(VQAModel, self).__init__()
        self.image_embedding = ImageEmbedding(resnet_feature_size, size)
        self.question_embedding = QuestionEmbedding(question_vocab_size, size, hidden_dim, n_layers, kernel_size, dropout, question_max_length) 
        self.mutan_size = mutan_size
        self.mutan = MutanFusion(size, mutan_size, num_layers=1)
        self.answer_vocab_size = answer_vocab_size
        self.target_length = answer_max_length
        self.answering = AnswerEmbedding(answer_vocab_size, size, hidden_dim, n_layers, kernel_size, dropout, question_max_length, answer_max_length)
        

    def forward(self, images, questions):
        image_embeddings = self.image_embedding(images)
        #print(image_embeddings.size())
        questions_conved, questions_combined = self.question_embedding(questions)
        #print(questions_conved.size())
        #print(questions_combined.size())
        questions_combined = self.mutan(image_embeddings, questions_conved, questions_combined)
        #print(questions_combined.size())
        predicted_answers, attention = self.answering(questions_conved, questions_combined)
        #print(predicted_answers.size())        
        #assert(False)
        predicted_answers = predicted_answers.view(-1, self.answer_vocab_size, self.target_length)
        return predicted_answers
'''
net = VQAModel()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
x = torch.randn(64, 196)
y = torch.ones(64, 20).long()
net(x, y)
'''

