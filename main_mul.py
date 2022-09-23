import torch 
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import tqdm
from data_prep.sentence_processing import build_vocab,sentence_processing
from data_prep.sentence_dataset_class import ProcessedSentences
from data_prep.sentence_processing import build_vocab,sentence_processing
from transformer_testing.tomislav_transformer import Seq2SeqTransformer
import time
import json
import os

def stringify_series(df):
    df['input_data'] = df['input_data'].astype('string')
    df['output_data'] = df['output_data'].astype('string')
    return df

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
def main(hp):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train = pd.read_json('data/train_data.json')
    df_test = pd.read_json('data/test_data.json')

    token_transform = get_tokenizer('basic_english')

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    train_input_vocab = build_vocab(df_train['input_data'],token_transform,special_symbols)
    train_output_vocab = build_vocab(df_train['output_data'],token_transform,special_symbols)

    train_dataset = ProcessedSentences(
        input_data = df_train['input_data'].values,
        output_data = df_train['output_data'].values,
    )

    test_dataset = ProcessedSentences(
        input_data = df_test['input_data'].values,
        output_data = df_test['output_data'].values
    )

    def collate_fn(batch):
        input_tensor = []
        output_tensor = []
        for input,output in batch:
            input_tensor.append(sentence_processing(input,
                                                 train_input_vocab,
                                                 token_transform,
                                                 special_symbols.index('<bos>'),special_symbols.index('<eos>')
                                                 )
                                )
            output_tensor.append(sentence_processing(output,
                                                 train_output_vocab,
                                                 token_transform,
                                                 special_symbols.index('<bos>'),special_symbols.index('<eos>')
                                                 ))
        src_batch = pad_sequence(input_tensor, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(output_tensor, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    # train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    # test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

    hyper_params = {}
    #with open('hyperparameters.json', 'r') as f:
    #    hyper_params = json.load(f)
    with open(hp, 'r') as f:
        hyper_params = json.load(f)

    torch.manual_seed(0)
    input_vocab_size = len(train_input_vocab)
    output_vocab_size = len(train_output_vocab)

    emb_size = hyper_params['emb_size']
    n_head = hyper_params['n_head']
    ffn_hid_dim = hyper_params['ffn_hid_dim']
    batch_size = hyper_params['batch_size']
    num_encoder_layers = hyper_params['num_encoder_layers']
    num_decoder_layers = hyper_params['num_decoder_layers']

    transformer = Seq2SeqTransformer(
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        n_head,
        input_vocab_size,
        output_vocab_size,
        ffn_hid_dim)

    transformer = transformer.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def train_epoch(model,optimizer):
        model.train()
        losses = 0
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
        for input_sent, output_sent in tqdm.tqdm(train_dataloader):
            input_sent = input_sent.type(torch.LongTensor)
            output_sent = output_sent.type(torch.LongTensor)

            input_sent = input_sent.to(device)
            output_sent = output_sent.to(device)

            output_input = output_sent[:-1,:]

            input_mask, output_mask, input_padding_mask, output_padding_mask = create_mask(input_sent,output_input)
            logits = model(
                input_sent,
                output_input,
                input_mask,
                output_mask,
                input_padding_mask,
                output_padding_mask,
                input_padding_mask)
            optimizer.zero_grad()

            output_out = output_sent[1:,:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), output_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()
        return losses/len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
        for input_sent, output_sent in tqdm.tqdm(test_dataloader):
            input_sent = input_sent.type(torch.LongTensor)
            output_sent = output_sent.type(torch.LongTensor)

            input_sent = input_sent.to(device)
            output_sent = output_sent.to(device)

            output_input = output_sent[:-1,:]
            input_mask, output_mask, input_padding_mask, output_padding_mask = create_mask(input_sent,output_input)
            logits = model(
                input_sent,
                output_input,
                input_mask,
                output_mask,
                input_padding_mask,
                output_padding_mask,
                input_padding_mask)


            output_out = output_sent[1:,:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), output_out.reshape(-1))

            losses += loss.item()
        return losses/len(test_dataloader)


    num_epochs = hyper_params['num_epochs']

    train_losses = []
    val_losses = []
    for epoch in range(1,num_epochs+1):
        train_loss = train_epoch(transformer,optimizer)
        val_loss = evaluate(transformer)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

    results = pd.DataFrame({'train_loss':train_losses,'val_loss':val_losses})
    time_signature = time.strftime("%d_%m-%Y_%H_%M_%S", time.localtime())

    results.to_csv(f'{time_signature}_results.csv')
    print(f'Saved results {time_signature}_results.csv')

    torch.save(transformer.state_dict(), f'{time_signature}_model.pt')
    print(f'Saved model {time_signature}_model.pt')    


    with open(f'{time_signature}_hyperparameters.json', 'w') as f:
        json.dump(hyper_params,f)
    print(f'Saved hyper parameters in {time_signature}_hyperparameters.json')    



    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for _ in range(max_len-1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys
    
    def translate(model: torch.nn.Module, input_sentence: str):
        model.eval()
        src = sentence_processing(input_sentence,
                                  train_input_vocab,
                                  token_transform,
                                  BOS_IDX,
                                  EOS_IDX).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        return " ".join(train_output_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    example_sentences = ['Give me the documents',
                         'Return the files right away',
                         'I need the documents',
                         'I want to return the files',
                         'I want to return the files right away',
                         'Answer the phone right away',
                         ]

    translated_sentences = [translate(transformer, sentence) for sentence in example_sentences]
    translated_df = pd.DataFrame({'sentence':example_sentences, 'translation':translated_sentences})
    translated_df.to_csv(f'{time_signature}_translations.csv')

if __name__ == '__main__':
    hp_folder = 'hp_folder'
    for hp in os.listdir(hp_folder):
        main(os.path.join(hp_folder,hp))
