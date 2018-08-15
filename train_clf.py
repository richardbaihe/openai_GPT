import argparse
import pandas as pd
from process import preprocess_char_wkx
from model import LM_transformer_similar
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--desc', type=str, default='pre_train_lm')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='../chinese_data/atec/char_AB_unk.tsv')
    parser.add_argument('--encoder_path', type=str, default='../chinese_data/pretrain_small/char_vocab')

    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--n_vocab', type=int, default=8000)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=200)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--pre_load', type=bool, default=True)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.2)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--pos_weight', type=float, default=0.8)

    args = parser.parse_args()
    data_origin = pd.read_csv('../chinese_data/atec/atec_nlp_sim_train.csv',sep='\t',index_col=0,names=['A','B','label'])
    data_add = pd.read_csv('../chinese_data/atec/atec_nlp_sim_train_add.csv',sep='\t',index_col=0,names=['A','B','label'])
    data = pd.concat([data_origin,data_add],ignore_index=True)

    preprocess_char_wkx(data,args)
    print('process finished')
    # data = pd.read_csv('data/generated_data/all.tsv', sep='\t', names=['A', 'B', 'label'])
    model = LM_transformer_similar(args)
    #preprocess_char_wkx(data,args)
    #model.train()