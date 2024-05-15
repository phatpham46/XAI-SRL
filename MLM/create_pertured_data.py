

import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm
from mlm_utils.metric_func import cosine_sim, cosine_module
from mlm_utils.transform_func import get_files

# def read_data(readPath):
#     df = pd.read_json(readPath, lines=True)
#     return df

def read_data_dask(readPath):
    df = dd.read_json(readPath, lines=True)
    return df

def load_df():
    df_split_1 = dd.read_parquet('./split_data/split_1.parquet')
    df_split_2 = dd.read_parquet('./split_data/split_2.parquet')
    df_split_3 = dd.read_parquet('./split_data/split_3.parquet')
    return df_split_1, df_split_2, df_split_3

def compute_cosine_similarities(df_predicate, vector_type, df_content, metric = 'cosine'):
    ''' cosine simimlarity matrix -> neg cosine sum, pos cosine sum'''
    similarities = pd.DataFrame(index=df_predicate.index, columns=df_content.index)
    
    if metric == 'cosine':
        for i in df_predicate.index:
            for j in df_content.index:
                vec1 = df_predicate.at[i, '{}_vector'.format(vector_type)]
                vec2 = df_content.at[j, '{}_vector'.format(vector_type)]
                
                similarities.at[i, j] = cosine_sim(vec1, vec2)
    elif metric == 'cosine_module':
        for i in df_predicate.index:
            for j in df_content.index:
                vec1 = df_predicate.at[i, '{}_vector'.format(vector_type)]
                vec2 = df_content.at[j, '{}_vector'.format(vector_type)]
                
                similarities.at[i, j] = cosine_module(vec1, vec2, cosine_sim(vec1, vec2))
    else:
        raise ValueError("Invalid metric")
    
    # Convert to numeric type
    similarities = similarities.apply(pd.to_numeric)
    
    # cosine -1
    min_indices = similarities.idxmin(axis=1)
    df_predicate.loc[:, "neg_{}_{}".format(metric, vector_type)] = df_content.loc[min_indices]['word'].values
    df_predicate.loc[:, "neg_value_{}_{}".format(metric, vector_type)] = similarities.min(axis=1).values
   
    # cosine 0
    pos_cos_sum_indices = np.abs(similarities).idxmin(axis=1)
    df_predicate.loc[:, "pos_{}_{}".format(metric, vector_type)] = df_content.loc[pos_cos_sum_indices]['word'].values
    df_predicate.loc[:, "pos_value_{}_{}".format(metric, vector_type)] = np.abs(similarities).min(axis=1).values  # absolute value
    
    
    # separate neg_value_cosine_sum and pos_value_cosine_value into dataframe with 2 column
    val_df = df_predicate[['neg_value_{}_{}'.format(metric, vector_type), 'neg_{}_{}'.format(metric, vector_type), 'pos_value_{}_{}'.format(metric, vector_type), 'pos_{}_{}'.format(metric, vector_type)]]
    
    
    # drop 2 columns from df_predicate
    df_predicate.drop(['neg_value_{}_{}'.format(metric, vector_type), 'pos_value_{}_{}'.format(metric, vector_type)], axis=1, inplace=True)
    return df_predicate, val_df

def select_noun_word(df_predicate, vector_type, metric = 'cosine'):
    
    # load noun dfs from disk
    noun_dfs = load_df()
    
    pd.options.mode.copy_on_write = True
    val_df1 = compute_cosine_similarities(df_predicate,vector_type, noun_dfs[0].compute(), metric)[1]
    val_df2 = compute_cosine_similarities(df_predicate, vector_type, noun_dfs[1].compute(), metric)[1]
    val_df3 = compute_cosine_similarities(df_predicate, vector_type, noun_dfs[2].compute(), metric)[1]
    
    # merge 3 val_df into one with axis 1 and get the min value of each row
    concat_df = pd.concat([val_df1, val_df2, val_df3], axis=1)
    
    concat_df_neg  = concat_df.filter(like='neg_value_{type}')
    concat_df_neg.columns = ['neg_value_{type}_1', 'neg_value_{type}_2', 'neg_value_{type}_3']
    
    word_cols = concat_df.filter(like='neg_{type}')
    word_cols.columns = ['neg_{type}_1', 'neg_{type}_2', 'neg_{type}_3']
    
    
    min_val_indices = concat_df_neg.columns.get_indexer(concat_df_neg.idxmin(axis=1))
    df_predicate.loc[:, "neg_{}".format(metric)] =  word_cols.apply(lambda row: row.iloc[min_val_indices[row.name]], axis=1)
    
    return df_predicate


def find_new_word(df_predicate, df_verb, df_adj, df_adv):
    pd.options.mode.copy_on_write = True
    
    # filter df with pos tag is NOUN
    predicate_noun = df_predicate[df_predicate['tag_id'] == 1]
    noun = select_noun_word(predicate_noun, 'sum', 'cosine')
    
    # filter df with pos tag is VERB
    predicate_verb = df_predicate[df_predicate['tag_id'] == 2]
    verb = compute_cosine_similarities(predicate_verb, 'sum', df_verb, metric = 'cosine')
    
    
    # filter df with pos tag is ADJ
    predicate_adj = df_predicate[df_predicate['tag_id'] == 3]
    adj = compute_cosine_similarities(predicate_adj, 'sum', df_adj, metric = 'cosine')
    
    
    # filter df with pos tag is ADV
    predicate_adv = df_predicate[df_predicate['tag_id'] == 4]
    adv = compute_cosine_similarities(predicate_adv, 'sum', df_adv, metric = 'cosine')
    
    # concat all dataframes
    res_df = pd.concat([noun, verb, adj, adv], axis=0).sort_index()
   
    return res_df

def get_tag_id(lst):
    nonzero_elements = filter(lambda x: x != 0, lst)
    return next(nonzero_elements, 0)


def main():
    file_paths = {
        "noun": "./data_mlm/process_folder/list_content_word_v2/NOUN.json",
        "verb": "./data_mlm/process_folder/list_content_word_v2/VERB.json",
        "adj": "./data_mlm/process_folder/list_content_word_v2/ADJ.json",
        "adv": "./data_mlm/process_folder/list_content_word_v2/ADV.json",
        "predicate_dir": "./data_mlm/process_folder/word_present_each_file/",
        "wri_dir": "./data_mlm/pertured_data/masked_data_parquet/"
    }

    df_verb = read_data_dask(file_paths["verb"])
    df_adj = read_data_dask(file_paths["adj"])
    df_adv = read_data_dask(file_paths["adv"])   
    
    files = get_files(file_paths["predicate_dir"]) 
    
    for file in tqdm(files):
        print("Processing file...", file)
        df_predicate = read_data_dask(file_paths["predicate_dir"] + file)
        df_predicate['tag_id'] = df_predicate['pos_tag_id'].apply(get_tag_id)
        
        pertured_df = find_new_word(df_predicate, df_verb, df_adj, df_adv)
        pertured_df.to_parquet(file_paths['wri_dir'] + file.replace("mlm_", "").split(".")[0] + ".parquet")